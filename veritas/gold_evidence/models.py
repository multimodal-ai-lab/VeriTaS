"""Data model for reconstructed gold evidence."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from enum import Enum

from ezmm import MultimodalSequence
from pydantic import BaseModel, Field

from veritas.common.base_model import Dismissable, VeritasBaseModel
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

    Judged against the `VerdictRationale`, not against the verdict directly:

    - ESSENTIAL: the rationale breaks without the proposition this item asserts.
    - AUXILIARY: relevant evidence that corroborates, contextualizes, explains, or strengthens the fact-check, but the rationale still carries without it..
    - BACKGROUND: provides context but carries no probative weight on its own.
    """

    ESSENTIAL = "essential"
    AUXILIARY = "auxiliary"
    BACKGROUND = "background"


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
    """Where a source sits relative to the two cutoffs (Spec §3.3).

    Both are computed from `t_e`, never predicted. The one judgement that needs a
    model - whether the evidence rests on a later event - is about the proposition
    rather than about any single source and lives on `Evidence.later_event`."""

    #: True iff t_e <= t_f.
    before_fact_check: bool
    #: True iff t_e <= t_c.
    before_claim: bool


class LaterEventCheck(BaseModel):
    """Whether an evidence item rests on a change of the world that happened only
    after the claim was made (Spec §3.3 (3)).

    This is the one place where "the information appeared later" and "the facts
    changed later" must not be confused: an article published after t_c that
    reports where a photo was taken long before t_c describes the world as it
    already was, and is admissible. A bankruptcy filed after t_c, cited as proof
    that the company was bankrupt at t_c, is not."""

    #: True iff the evidence relies on a world state that came into existence
    #: after t_c *and* that state bears on the claim's truthfulness.
    change_detected: bool = False
    #: The model's own reasoning trace, as reported by the provider's API.
    reasoning: str | None = None
    #: The short justification the model stated alongside its judgement.
    justification: str | None = None
    rater: str | None = None


class EvidenceSource(VeritasBaseModel):
    """One place the proposition can be read: an independently locatable
    publication, or - for tools and offline evidence - a named origin without a
    locator.

    Everything Stage 2 decides lives here, because every one of those questions is
    about the *source*, not about the proposition: can it still be retrieved, when
    did it become available, does it still say what it was cited for. An `Evidence`
    item aggregates the sources that report the same proposition, and it survives
    as long as one of them does.
    """

    evidence_id: int | None = None

    name: str  # Name of the publishing organization or person
    kind: SourceKind = SourceKind.OTHER
    #: URL or equivalent stable locator. None only for sources that are not
    #: publications - a tool without a page, or offline evidence such as a phone call.
    locator: str | None
    proximity: ProximityLevel = ProximityLevel.SECONDARY
    raw_content: str | None = None  #: The source's content as scraped with scrapeMM

    #: t_e - when the source became publicly accessible. None if no clear date is
    #: associated with it (in particular for tools).
    available_since: datetime | None = None
    accessed_at: datetime = Field(default_factory=datetime.now)

    # --- Stage 2 outcome ---
    accessible: bool | None = None  # §3.1
    faithfulness: Faithfulness | None = None  # §3.2
    temporal_validation: TemporalValidation | None = None  # §3.3

    #: Determined algorithmically, see `veritas.gold_evidence.admissibility`.
    admissible: bool | None = None
    inadmissibility_reason: str | None = None

    #: Set while the source waits out a rate limit; it is neither admissible nor
    #: rejected until the window has passed.
    deferred_until: datetime | None = None
    #: Why the source could not be used, including error messages.
    dismissed_reason: str | None = None

    @property
    def content(self) -> MultimodalSequence | None:
        """The scraped source content as a MultimodalSequence."""
        return MultimodalSequence(self.raw_content) if self.raw_content else None

    @property
    def domain(self) -> str | None:
        return get_domain(self.locator)

    @property
    def filtered(self) -> bool:
        """True once Stage 2 has run to completion on this source."""
        return self.admissible is not None

    @property
    def deferred(self) -> bool:
        """True while the cooldown after a rate limit is still running."""
        return self.deferred_until is not None and self.deferred_until > datetime.now()

    def defer(self, hours: int) -> None:
        """Postpones this source, extending an existing window but never shortening
        it. Unlike `Deferrable.defer` this does not write to the DB: a source is
        saved with the `Evidence` item that owns it."""
        until = datetime.now() + timedelta(hours=hours)
        if self.deferred_until is None or until > self.deferred_until:
            self.deferred_until = until

    def time_to_claim(self, t_c: datetime | None) -> float | None:
        """(t_e - t_c) in days. None if either timestamp is unknown."""
        return _delta_days(self.available_since, t_c)

    def time_to_fact_check(self, t_f: datetime | None) -> float | None:
        """(t_e - t_f) in days. None if either timestamp is unknown."""
        return _delta_days(self.available_since, t_f)

    def __str__(self) -> str:
        origin = f"{self.name} ({self.kind.value})"
        return f"{origin}, {self.locator}" if self.locator else origin


class Evidence(Dismissable, VeritasBaseModel):
    """An atomic piece of factual information that the professional fact-check used
    to establish its verdict, together with every source that reports it.

    Evidence is the *aggregation layer*: the proposition and its role belong here,
    while retrieval, dating, and faithfulness belong to the individual sources. Two
    outlets reporting the same fact are two sources of one evidence item, not two
    evidence items - so losing one of them costs nothing, and the item is discarded
    only when it has lost every source it had.
    """

    claim_id: int
    review_id: int | None = None  # The review whose article this was extracted from
    article_id: int | None = None

    #: The atomic factual information contributed by the sources, incl. media references
    proposition: str

    #: Every source that reports this proposition. At least one.
    sources: list[EvidenceSource] = Field(default_factory=list)

    role: EvidenceRole = EvidenceRole.AUXILIARY

    # --- Stage 1 metadata ---
    extraction_reasoning: str | None = None
    extraction_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    # --- Stage 2 metadata ---
    #: §3.3 (3). Asked once per item, and only for items that became available
    #: after t_c - evidence that already existed when the claim was made cannot
    #: rest on anything that happened afterwards.
    later_event: LaterEventCheck | None = None

    # ------------------------------------------------------------------ helpers

    @property
    def relies_on_later_event(self) -> bool:
        """Whether §3.3 (3) rules this item out. A property of the proposition, so
        it condemns the item however many sources still report it."""
        return bool(self.later_event and self.later_event.change_detected)

    @property
    def admissible(self) -> bool | None:
        """Whether the proposition still has a usable source.

        True as soon as one source is admissible, False once every source was
        decided and rejected, and None while any source is still unfiltered - an
        item is not decided until its last source is. A later event rules the item
        out regardless of its sources."""
        if self.relies_on_later_event:
            return False
        if not self.sources:
            return False
        if any(source.admissible for source in self.sources):
            return True
        if all(source.filtered for source in self.sources):
            return False
        return None

    @property
    def admissible_sources(self) -> list["EvidenceSource"]:
        return [source for source in self.sources if source.admissible]

    @property
    def inadmissibility_reason(self) -> str | None:
        """Why the item was discarded: the reason of its last remaining source. An
        item with several sources reports the first one, since the reasons differ
        per source and the item needs a single label for the exports."""
        if self.relies_on_later_event:
            return "later_event"
        if self.admissible is not False:
            return None
        reasons = [source.inadmissibility_reason for source in self.sources
                   if source.inadmissibility_reason]
        return reasons[0] if reasons else None

    @property
    def filtered(self) -> bool:
        """True once Stage 2 has run to completion on every source."""
        return bool(self.sources) and all(source.filtered or source.deferred
                                          for source in self.sources)

    @property
    def deferred(self) -> bool:
        """True while any source waits out a rate limit: the item is not decided
        until that source has been retried."""
        return any(source.deferred for source in self.sources)

    @property
    def is_essential(self) -> bool:
        return self.role == EvidenceRole.ESSENTIAL

    @property
    def available_since(self) -> datetime | None:
        """t_e of the item: the *earliest* time at which any of its sources made the
        proposition available. That is when the proposition could first be read."""
        times = [source.available_since for source in self.sources
                 if source.available_since is not None]
        return min(times) if times else None

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
        sources = "\n".join(f"  -- {source}" for source in self.sources)
        return f"[{self.role.value}] {self.proposition}\n{sources}"


class VerdictRationale(VeritasBaseModel):
    """The reasoning that bridges the gap between the evidence and the verdict.

    Not every claim is settled by external evidence: some are settled by arithmetic,
    by a logical contradiction inside the claim, or by what the claim's own image
    plainly shows. The rationale captures that step of the fact-check's argument, so
    that such instances can be analyzed instead of looking evidence-less.

    It may combine or transform the premises that the evidence and the claim already
    provide, but it must not silently introduce a new externally verifiable factual
    premise - such a premise would be evidence, and belongs in `Evidence` where it
    can be retrieved, dated, and checked for faithfulness. It also never states the
    verdict: that is what the sufficiency ensemble has to recover on its own.
    """

    claim_id: int
    review_id: int | None = None  # The review whose article this was extracted from
    article_id: int | None = None

    #: The reasoning itself, including media references, like `Evidence.proposition`.
    rationale: str

    #: The model's own reasoning trace for the extraction call, as reported by the
    #: provider's API. None if the model reported none.
    extraction_reasoning: str | None = None

    @property
    def sequence(self) -> MultimodalSequence:
        return MultimodalSequence(self.rationale)

    def __str__(self) -> str:
        return self.rationale


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
