"""Data model for reconstructed gold evidence."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from ezmm import MultimodalSequence
from pydantic import BaseModel, Field

from veritas.common.base_model import Deferrable, Dismissable, VeritasBaseModel
from veritas.util import get_domain


class SourceKind(str, Enum):
    """Type of the publication the evidence was taken from."""

    SOCIAL_MEDIA_POST = "social_media_post"
    NEWS_ARTICLE = "news_article"
    FACT_CHECK = "fact_check"
    SCIENTIFIC_MANUSCRIPT = "scientific_manuscript"
    OFFICIAL_STATEMENT = "official_statement"
    GOVERNMENT_RECORD = "government_record"
    DATABASE = "database"
    ENCYCLOPEDIA = "encyclopedia"
    OFFLINE = "offline"
    TOOL = "tool"
    OTHER = "other"


class ProximityLevel(str, Enum):
    """How close the source is to the reported matter.

    - PRIMARY: first-hand, produced by a party directly involved (the original post,
      an eyewitness recording, an official register, the raw output of a tool).
    - SECONDARY: reports on, analyzes, or aggregates primary material (news coverage,
      expert analysis, a scientific study interpreting data).
    - TERTIARY: compiles secondary material (encyclopedias, overviews, digests).
    """

    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"


class EvidenceRole(str, Enum):
    """How much the evidence contributes to establishing the verdict.

    - ESSENTIAL: the verdict does not hold without it.
    - AUXILIARY: corroborates or refines essential evidence.
    - BACKGROUND: provides context but carries no probative weight on its own.
    """

    ESSENTIAL = "essential"
    AUXILIARY = "auxiliary"
    BACKGROUND = "background"


class EvidenceSource(BaseModel):
    """Where an evidence item comes from: an independently locatable publication,
    or - for tools and offline evidence - a named origin without a locator."""

    name: str  # Name of the publishing organization or person
    kind: SourceKind = SourceKind.OTHER
    #: URL or equivalent stable locator. None only for sources that are not
    #: publications - a tool without a page, or offline evidence such as a phone call.
    locator: str | None
    proximity: ProximityLevel = ProximityLevel.SECONDARY
    raw_content: str | None = None  #: The source's content as scraped with scrapeMM

    @property
    def content(self) -> MultimodalSequence | None:
        """The scraped source content as a MultimodalSequence."""
        return MultimodalSequence(self.raw_content) if self.raw_content else None

    @property
    def domain(self) -> str | None:
        return get_domain(self.locator)


class Faithfulness(BaseModel):
    """Whether the currently retrieved source still supports the proposition.

    `assessment` lies in [-1, 1]: -1 means source and proposition clearly
    contradict, 0 a neutral relation or "not enough information", and 1 that the
    proposition is clearly entailed. Intermediate values reflect uncertainty.
    """

    assessment: float = Field(ge=-1.0, le=1.0)
    #: The model's own reasoning trace, as reported by the provider's API. None if
    #: the model reported none (no reasoning effort, or a non-reasoning model).
    reasoning: str | None = None
    #: The short justification the prompt asked the model to state for its rating.
    justification: str | None = None
    rater: str | None = None


class TemporalValidation(BaseModel):
    """Outcome of the temporal/leakage checks (Spec §3.3)."""

    #: True iff t_e <= t_f. Computed, not predicted.
    before_fact_check: bool
    #: True iff t_e <= t_c. Computed, not predicted.
    before_claim: bool
    #: Whether the source reports on a post-t_c event that directly changes the
    #: (hidden, true) information state and, thus, the claim's veracity.
    later_event: bool = False
    #: The model's own reasoning trace, as reported by the provider's API.
    reasoning: str | None = None
    #: The short justification the model stated alongside its judgement.
    justification: str | None = None
    rater: str | None = None


class Evidence(Dismissable, Deferrable, VeritasBaseModel):
    """An atomic piece of factual information that the professional fact-check
    used to establish its verdict, together with its source and pipeline metadata."""

    claim_id: int
    review_id: int | None = None  # The review whose article this was extracted from
    article_id: int | None = None

    #: The atomic factual information contributed by the source, including media references
    proposition: str

    source: EvidenceSource

    #: t_e - when the evidence became publicly accessible. None if no clear date
    #: is associated with the source (in particular for tools).
    available_since: datetime | None = None

    role: EvidenceRole = EvidenceRole.AUXILIARY
    accessed_at: datetime = Field(default_factory=datetime.now)

    # --- Stage 1 metadata ---
    extraction_reasoning: str | None = None
    extraction_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # --- Stage 2 metadata ---
    accessible: bool | None = None  # §3.1
    faithfulness: Faithfulness | None = None  # §3.2
    temporal_validation: TemporalValidation | None = None  # §3.3

    #: Determined algorithmically, see `veritas.gold_evidence.admissibility`.
    admissible: bool | None = None
    inadmissibility_reason: str | None = None

    # ------------------------------------------------------------------ helpers

    @property
    def proposition_sequence(self) -> MultimodalSequence:
        return MultimodalSequence(self.proposition)

    @property
    def is_multimodal(self) -> bool:
        """Whether the proposition carries media. False if the media it references
        have meanwhile left the ezMM store, so that a cleaned-up medium cannot take
        down a whole analysis export."""
        try:
            seq = self.proposition_sequence
        except (ValueError, AssertionError):
            return False
        return seq.has_images() or seq.has_videos()

    @property
    def filtered(self) -> bool:
        """True once Stage 2 has run to completion on this item."""
        return self.admissible is not None

    def time_to_claim(self, t_c: datetime | None) -> float | None:
        """(t_e - t_c) in days. None if either timestamp is unknown."""
        return _delta_days(self.available_since, t_c)

    def time_to_fact_check(self, t_f: datetime | None) -> float | None:
        """(t_e - t_f) in days. None if either timestamp is unknown."""
        return _delta_days(self.available_since, t_f)

    def __str__(self) -> str:
        # Tools and offline evidence may have no locator at all.
        origin = f"{self.source.name} ({self.source.kind.value})"
        if self.source.locator:
            origin += f", {self.source.locator}"
        return (f"[{self.role.value}/{self.source.proximity.value}] {self.proposition}\n"
                f"  -- {origin}")


def to_naive(value: date | datetime | None) -> datetime | None:
    """Normalizes dates/datetimes to naive UTC datetimes, matching the DB columns.

    Every timestamp entering the reconstruction passes through here, so the rest of
    the package works on naive `datetime` objects only; plain `date` values are
    widened to midnight rather than compared against datetimes."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            from datetime import timezone

            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    raise TypeError(f"Unsupported temporal type: {type(value)}")


def _delta_days(a: date | datetime | None, b: date | datetime | None) -> float | None:
    a, b = to_naive(a), to_naive(b)
    if a is None or b is None:
        return None
    return (a - b).total_seconds() / 86400.0
