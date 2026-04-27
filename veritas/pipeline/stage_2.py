from typing import Optional

from pydantic import BaseModel, Field
from scrapemm import retrieve

from veritas.common import Prompt, Publisher, Review, SignatoryStatus
from veritas import logger
from veritas.models import gpt_cheap
from veritas.db import db
from veritas.pipeline.util.signatories import save_publisher
from veritas.pipeline.util.stage import Stage
from veritas.util import get_domain
from veritas.util.util import get_quarter_date_range


class Stage2(Stage):
    """Continuously identifies publishers for reviews."""

    id = 2
    name = "Publisher Identification"

    async def step(self, *,
                   start: tuple[int, int] = (2016, 1),
                   end: tuple[int, int] | None = None,
                   batch_size: int = 100) -> None:
        reviews = await self.get_queued_items(start=start, end=end, batch_size=batch_size)
        if reviews:
            await identify_publishers(reviews)

    async def get_queued_items(self, *,
                               start: tuple[int, int],
                               end: tuple[int, int] | None,
                               batch_size: int) -> list[Review]:
        """Fetches a batch of reviews at stage 1 within the given date range."""
        start_date, _ignored = get_quarter_date_range(*start)
        end_date = None
        if end:
            _ignored, end_date = get_quarter_date_range(*end)
        return await db.get_reviews(stage=1, limit=batch_size,
                                    start_date=start_date, end_date=end_date)


async def identify_publishers(reviews: list[Review]):
    """Reads the publishers from the Reviews. If it encounters new publishers, saves them to the DB."""
    for review in reviews:
        publisher = await review.publisher

        # Make sure we know the publisher
        if not publisher:
            try:
                # We need to identify the publisher.
                domain = get_domain(review.raw_publisher_url or review.url)
                assert domain, "Could not identify the publisher's domain."
                publisher = await db.get_publisher_by_url(domain)
                if not publisher:
                    # The publisher is unknown. We need to register it.
                    name = review.raw_publisher_name
                    publisher = await register_new_publisher(domain, name)
                    assert publisher, "Unable to register publisher."
                review.publisher_id = publisher.id
                await review.save_to_db()
            except Exception as e:
                await review.dismiss(f"Could not identify publisher: {e}")

        # Validate publisher signatory status
        if publisher:
            if (publisher.ifcn_status not in [SignatoryStatus.ACTIVE, SignatoryStatus.IN_RENEWAL] and
                    publisher.efcsn_status not in [SignatoryStatus.ACTIVE, SignatoryStatus.IN_RENEWAL]):
                await review.dismiss("Publisher is not an IFCN or EFCSN signatory.")

        await review.set_stage(2)


async def register_new_publisher(domain: str, name: Optional[str] = None) -> Optional[Publisher]:
    """Visits the publisher's homepage and extract all relevant information.
    Saves the new publisher instance to the DB."""
    logger.info(f"Registering new publisher {name or domain}...")

    # Read the publisher's homepage
    try:
        response = await retrieve(f"https://{domain}", show_progress=False)
        scraped = response.content
    except Exception:
        scraped = None

    # Try to identify language and country of this publisher by homepage
    language = country = None
    if scraped:
        class ExtractedPublisher(BaseModel):
            name: str = Field(description="The official name of the publisher.")
            language: str = Field(
                description="The two or three-digit ISO 639-1 language code of the publisher website's language."
            )
            country: str = Field(description="The name of country of the publisher's headquarters.")

        scraped = str(scraped)[:128_000]  # Limit homepage to 128k chars
        prompt = Prompt("veritas/prompts/get_publisher_details.md", domain=domain, homepage=scraped)
        extracted = await gpt_cheap.generate(prompt, response_format=ExtractedPublisher)

        if extracted:
            name = name or extracted.name
            language = extracted.language
            country = extracted.country
        else:
            raise ValueError(f"Could not extract publisher details from {domain}.")

    publisher = Publisher(
        name=name or domain,
        domains={domain},
        language=language,
        country=country,
    )
    await save_publisher(publisher)
    return publisher
