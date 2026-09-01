import logging
from datetime import date, datetime
from typing import Any
from typing import TYPE_CHECKING

from asyncpg import UniqueViolationError
from ezmm import MultimodalSequence
from pydantic import HttpUrl

from veritas.common.base_model import VeritasBaseModel, Deferrable, Dismissable
from veritas.common.platforms import PLATFORMS
from veritas.util import get_domain
from veritas.util.url import resolve_archiving_url, unshorten, is_archiving_url

if TYPE_CHECKING:
    from veritas.common.review import Review

logger = logging.getLogger("VeriTaS")


class Appearance(Dismissable, Deferrable, VeritasBaseModel):
    """Places on the web where a claim occurs."""

    url: HttpUrl | None = None  # Either url or archive_url must be present
    archive_url: HttpUrl | None = None  # WaybackMachine URL etc., if available

    published: date | datetime | None = None
    author_name: str | None = None  # Name of the claimant if known
    author_url: HttpUrl | None = None  # Link to author profile if available

    # Scraped content (stored separately for original and archived URLs)
    original_scraped_content: MultimodalSequence | None = None
    archived_scraped_content: MultimodalSequence | None = None

    # Success flags indicating whether the respective scrape yielded sufficient content
    original_scrape_ok: bool = False
    archived_scrape_ok: bool = False

    # Which scraping method/engine was used to retrieve this appearance (e.g., aiohttp, firecrawl, yt_dlp)
    scrape_method: str | None = None

    def model_post_init(self, context: Any, /) -> None:
        assert self.url or self.archive_url

    def __eq__(self, other):
        return isinstance(other, Appearance) and (
                self.url == other.url or self.archive_url == other.archive_url
        )

    def __hash__(self):
        return hash(str(self.url) + str(self.archive_url))

    @property
    async def reviews(self) -> list["Review"]:
        from veritas.db import db
        return await db.get_reviews_for_appearance(self.id)

    @property
    def platform(self) -> str:
        """Returns the name of the platform associated with the URL
        of this appearance. If the platform is not recognized, returns
        the domain name."""
        domain = get_domain(self.url)
        return PLATFORMS.get(domain, domain)

    async def defer(self, until: datetime | None = None, *, hours: int | None = None) -> None:
        await super().defer(until, hours=hours)

        # Defer corresponding reviews as well
        for review in await self.reviews:
            await review.defer(until=until, hours=hours)

    @property
    def scraped_content(self) -> MultimodalSequence | None:
        """Backward-compatible accessor. Prefer original content if marked OK,
        otherwise return archived content if marked OK, else None."""
        if self.original_scrape_ok and self.original_scraped_content is not None:
            return self.original_scraped_content
        if self.archived_scrape_ok and self.archived_scraped_content is not None:
            return self.archived_scraped_content
        # Fallback: if flags are not set but content exists, prefer original
        return self.original_scraped_content or self.archived_scraped_content

    @property
    def scrape_ok(self) -> bool:
        return self.original_scrape_ok or self.archived_scrape_ok

    async def save_to_db(self) -> None:
        """Saves the appearance to the DB. If another appearance with the same
        `url` or `archive_url` already exists (unique constraint), merge both
        and keep a single, consolidated row."""
        try:
            await super().save_to_db()
        except UniqueViolationError:
            await self._merge_with_existing()

    async def _merge_with_existing(self) -> None:
        """Merges this appearance with an existing one if it exists.
        Copies the data from the existing appearance to this one whenever
        this one does not have a value set yet."""
        from veritas.db import db

        # Load the existing appearance
        existing = await db.get_appearance_by_url(self.url) or \
                   await db.get_appearance_by_archive_url(self.archive_url)

        if not existing:
            return  # Nothing to merge

        if self.id is not None:
            logger.debug(f"Merging appearance {self.id} with existing {existing.id}.")

        # Perform the merge
        if self.id is None:
            self.id = existing.id
        if self.url is None:
            self.url = existing.url
        if self.archive_url is None:
            self.archive_url = existing.archive_url
        if self.original_scraped_content is None:
            self.original_scraped_content = existing.original_scraped_content
        if self.archived_scraped_content is None:
            self.archived_scraped_content = existing.archived_scraped_content
        if self.scrape_method is None:
            self.scrape_method = existing.scrape_method

        self.original_scrape_ok = is_sufficient_content(self.original_scraped_content)
        self.archived_scrape_ok = is_sufficient_content(self.archived_scraped_content)

        # Ignore dismissed and deferred flags

        # Only if both appearances already have a row in the DB, delete the existing (old) one
        if self.id != existing.id:
            # Remove the existing appearance from the DB
            await db.replace_appearance(to_replace=existing.id, replacement=self.id)

        await super().save_to_db()  # Should run now without any unique constraint violation


async def appearance_from_url(url: HttpUrl | str) -> Appearance | None:
    """If the appearance is known (saved in the DB), returns the existing one,
    otherwise creates a new instance. If the URL is a web archive URL, resolves it to
    obtain the original URL. In that case, adds the archive URL to the Appearance."""
    # Ensure string is properly formatted/escaped and unshorten if it is a tinyurl
    url_unshortened = unshorten(str(HttpUrl(url)))

    # Resolve archive URL
    if is_archiving_url(url_unshortened):
        archive_url = url_unshortened
        result = await resolve_archiving_url(url_unshortened)
        app_url = result.get("original_url") if result else None
        scraped_archive_content = result.get("scraped_content") if result else None
    else:
        archive_url = None
        app_url = url_unshortened
        scraped_archive_content = None

    appearance = Appearance(
        url=app_url,
        archive_url=archive_url,
        archived_scraped_content=scraped_archive_content,
        archived_scrape_ok=is_sufficient_content(scraped_archive_content)
    )

    await appearance.save_to_db()  # Handles unique constraint violation if appearance already exists

    return appearance


def is_sufficient_content(content: MultimodalSequence | None) -> bool:
    """Returns True if content is considered a sufficient scrape result of an appearance
    based on length or media presence and does not show typical removal messages."""
    if not content:
        return False
    app_str = str(content).lower()
    if len(app_str) < 100 and not content.has_images() and not content.has_videos():
        return False
    if "error 404" in app_str or "dns error" in app_str:
        return False
    return True
