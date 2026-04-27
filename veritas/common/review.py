from __future__ import annotations

import logging
from dataclasses import field
from datetime import date, datetime
from typing import TYPE_CHECKING

from pydantic import HttpUrl

from veritas.common.appearance import Appearance
from veritas.common.base_model import VeritasBaseModel, Dismissable, Deferrable
from veritas.common.publisher import Publisher
from veritas.util.parsing import lang_iso_to_name

if TYPE_CHECKING:
    from veritas.common.claim import Claim
    from veritas.common.article import Article

logger = logging.getLogger("VeriTaS")


class Review(Dismissable, Deferrable, VeritasBaseModel):
    """A fact-check of a claim, usually published within an article."""

    url: HttpUrl

    # The raw ClaimReview dictionaries from the three different sources
    google_claim_review: dict | None = None
    datacommons_claim_review: dict | None = None
    direct_claim_review: dict | None = None

    # Raw properties
    raw_claim: str  # Claim as exposed by the ClaimReviews or (if not there) the article
    raw_rating: str | None = None  # Rating as exposed by the ClaimReviews or (if not there) the article
    raw_claimant_name: str | None = None  # The author of the claim as by the ClaimReview data
    raw_claimant_url: HttpUrl | None = None  # URL to the profile of the author of the claim
    raw_claim_date: date | datetime | None = None  # The claim release date as by the ClaimReview data
    raw_publisher_name: str | None = None  # The name of organization (!) that published the review
    raw_publisher_url: HttpUrl | None = None  # The URL to the organization's homepage

    published: date | datetime | None = None
    author_name: str | None = None  # The person (!) who wrote the review
    author_url: HttpUrl | None = None
    modified: date | datetime | None = None  # Last day the review was modified by the publisher
    language: str | None = None  # ISO 639-1 language code

    # Foreign keys
    claim_id: int | None = None  # The claim assessed by this review
    publisher_id: int | None = None  # The organization that released the review
    appearance_ids: set[int] | None = field(
        default_factory=set
    )  # All claim appearances occurring in this review (in text or CR)

    # Meta
    stage: int = 0  # The most recent completed processing stage of the review (6 = finished)

    @property
    async def publisher(self) -> Publisher | None:
        if self.publisher_id:
            from veritas.db import db

            return await db.get_publisher_by_id(self.publisher_id)

    @property
    async def claim(self) -> Claim | None:
        if self.claim_id:
            from veritas.db import db

            return await db.get_claim_by_id(self.claim_id)

    @property
    async def appearances(self) -> list[Appearance]:
        if self.appearance_ids:
            from veritas.db import db

            return [await db.get_appearance_by_id(id) for id in self.appearance_ids]
        return []

    @property
    async def article(self) -> Article | None:
        from veritas.db import db

        return await db.get_article_by_url(self.url)

    async def has_article(self) -> bool:
        from veritas.db import db

        return await db.article_exists(self.url)

    @property
    def language_name(self) -> str | None:
        """Returns the display name of the language of the claim, if available."""
        return lang_iso_to_name(self.language) or self.language

    async def set_stage(self, stage: int) -> None:
        """Updates the completed processing stage ID of this review if it's not dismissed
        and currently not deferred. Saves the review in any case to the DB."""
        if not self.dismissed and not self.deferred:
            if not self.stage <= stage <= self.stage + 1:
                logger.info(f"Performing unusual stage transition {self.stage} -> {stage} for review {self.id}.")
            self.stage = stage
        await self.save_to_db()
