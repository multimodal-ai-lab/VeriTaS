from ezmm import MultimodalSequence
from pydantic import HttpUrl

from veritas.common.base_model import VeritasBaseModel, Dismissable

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from veritas.common.review import Review


class Article(Dismissable, VeritasBaseModel):
    """A publication containing reviews"""

    url: HttpUrl
    publisher_id: int
    review_ids: set[int]

    scraped_page: str | None = None  # The raw page as Markdown
    extracted_article: str | None = None  # The actual, relevant article content
    title: str | None = None

    @property
    def content(self) -> MultimodalSequence:
        """Returns the actual relevant content of the article as a MultimodalSequence.
        Falls back to the raw scrape if extraction is not available."""
        return MultimodalSequence(self.extracted_article or self.scraped_page or "")

    @property
    async def reviews(self) -> list["Review"]:
        from veritas.db import db
        return [await db.get_review_by_id(id) for id in self.review_ids]

    async def dismiss(self, reason: str) -> None:
        """Excludes this article and all associated reviews from VeriTaS for quality control reasons."""
        await super().dismiss(reason)
        # Also dismiss associated reviews
        for review in await self.reviews:
            await review.dismiss(reason)
