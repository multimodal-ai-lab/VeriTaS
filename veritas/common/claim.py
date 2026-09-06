import logging
from dataclasses import field
from datetime import date, datetime
from typing import Optional

from ezmm import MultimodalSequence

from veritas.common.appearance import Appearance
from veritas.common.base_model import VeritasBaseModel, Dismissable
from veritas.common.review import Review
from veritas.common.verdict import Verdict
from veritas.util.parsing import lang_iso_to_name
from veritas.util.util import date_or_datetime_to_str

logger = logging.getLogger("VeriTaS")


class Claim(Dismissable, VeritasBaseModel):
    data: str  # The claim's text, images, videos, ...; Without quotes around it.
    date: date | datetime | None  # Typically the earliest publication date of the claim
    language: str | None = None  # ISO 639-1 language code (e.g., 'en', 'de')

    # Foreign keys
    appearance_ids: set[int]  # Occurrences of the claim in the web
    review_ids: set[int]  # The reviews that deal with this claim
    verdict_ids: set[int] = field(
        default_factory=set
    )  # Our verdicts about this claim (usually only one)

    media_origin: str | None = None  # Origin of the media used in the claim: 'original' | 'archived' | 'article'

    # Rectification capability
    rectifiable: bool | None = None  # Whether this claim can be meaningfully rectified

    # Validation fields, flagging problems with the claim
    is_ambiguous: bool | None = None  # Whether the claim is unclear with the provided context
    media_expose_verdict: bool | None = None  # Whether the media contain annotations from the fact-checking article
    text_exposes_verdict: bool | None = None  # Whether the claim's text contains (hints to) the verdict
    missing_referenced_media: bool | None = None  # Whether all related media are included in the claim
    check_completed: bool | None = None   # Whether the above checks are done
    is_inconsistent: bool | None = None  # Whether the claim contains contradictions (esp. w.r.t. media)
    is_unshareable: bool | None = None  # Whether the claim would ever be shared by anyone on social media

    is_rectified: bool = False  # Whether this claim is a corrected version of another (original) claim
    variant_id: int | None = None  # The ID of this claim's variant (original/rectified) if exists

    released_quarter: bool = False  # Whether this claim was released in a quarter split
    released_longitudinal: bool = False  # Whether this claim was released in the longitudinal split
    text_embedding: list[float] | None = None  # The OpenAI text-embedding-3-large embedding vector of the claim

    # Gold Evidence Reconstruction (see veritas.gold_evidence). Written exclusively
    # by that pipeline; `dismissed` deliberately stays untouched, so a rejected
    # instance remains a valid VeriTaS claim.
    gold_evidence_status: str | None = None  # pending|extracted|filtered|accepted|rejected
    gold_evidence_reason: str | None = None  # Why the instance was rejected, if it was
    gold_evidence_updated_at: datetime | None = None

    @property
    def released(self) -> bool:
        """Returns True if the claim was released in any split."""
        return self.released_quarter or self.released_longitudinal

    @property
    def is_original(self) -> bool:
        """Returns True if this claim is NOT a rectification of another claim."""
        return not self.is_rectified

    @property
    def is_valid(self) -> bool:
        """Returns True iff all validation checks are passed (no problems were detected)."""
        return bool(self.check_completed) and not any([
            self.is_ambiguous,
            self.is_inconsistent,
            self.media_expose_verdict,
            self.text_exposes_verdict,
            self.missing_referenced_media,
            self.is_unshareable
        ])

    @property
    async def appearances(self) -> list[Appearance]:
        from veritas.db import db

        return [await db.get_appearance_by_id(id) for id in self.appearance_ids]

    @property
    def n_appearances(self) -> int:
        return len(self.appearance_ids)

    @property
    async def reviews(self) -> list[Review]:
        from veritas.db import db

        return [await db.get_review_by_id(id) for id in self.review_ids]

    @property
    async def current_verdict(self) -> Verdict | None:
        from veritas.db import db

        return await db.get_verdict_by_claim_id(self.id)

    @property
    async def claimants(self) -> list[str]:
        """Returns the list of claimants (authors) of the claim if this claim is original."""
        if self.is_original:
            reviews = await self.reviews
            appearances = await self.appearances
            # Gather claimant names from reviews and appearances
            claimants = set(review.raw_claimant_name for review in reviews)
            claimants.update(appearance.author_name for appearance in appearances)
            claimants.discard(None)
            return list(claimants)
        else:
            return []

    @property
    def gold_evidence_rejected(self) -> bool:
        """True if the Gold Evidence Reconstruction could not recover the gold verdict."""
        from veritas.gold_evidence import STATUS_REJECTED

        return self.gold_evidence_status == STATUS_REJECTED

    @property
    async def evidence(self) -> list["Evidence"]:
        """The reconstructed gold evidence of this claim."""
        from veritas.db import db

        return await db.get_evidence_for_claim(self.id)

    @property
    async def variant(self) -> Optional["Claim"]:
        if self.variant_id is None:
            return None
        else:
            from veritas.db import db
            return await db.get_claim_by_id(self.variant_id)

    async def dismiss(self, reason: str, also_dismiss_reviews: bool = True) -> None:
        await super().dismiss(reason)
        # Update reviews accordingly, but only if this is the original claim
        if self.is_original and also_dismiss_reviews:
            for review in await self.reviews:
                await review.dismiss(reason)

    async def take_back_dismissal(self) -> None:
        """Take back the dismissal of this claim."""
        await super().take_back_dismissal()
        for review in await self.reviews:
            await review.take_back_dismissal()

    @property
    def date_str(self) -> str:
        return date_or_datetime_to_str(self.date)

    @property
    def language_name(self) -> Optional[str]:
        """Returns the display name of the language of the claim, if available."""
        return lang_iso_to_name(self.language) or self.language

    @property
    def n_words(self) -> int:
        """Returns the number of words in the claim."""
        return len(self.data.split())

    def as_multimodal_sequence(self) -> MultimodalSequence:
        return MultimodalSequence(self.data)

    async def save_to_db(self):
        if self.text_embedding is None:
            await self.compute_text_embedding()
        await super().save_to_db()

    async def compute_text_embedding(self):
        """Computes the text embedding for the claim."""
        from veritas.models import text_embedder
        text_items = [item for item in MultimodalSequence(self.data) if isinstance(item, str)]
        text = " ".join(text_items).strip()
        self.text_embedding = await text_embedder.embed(text)

    def __str__(self):
        return f'"{self.data}"\nPublished on: {self.date_str}'
