"""Data model for reconstructed gold evidence."""

from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta
from enum import Enum
from urllib.parse import urlsplit, urlunsplit

from ezmm import MultimodalSequence
from pydantic import BaseModel, Field, field_validator

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
    """How much the evidence contributes to establishing the gold verdict.

    Judged once the evidence list of an article is complete, by asking what
    removing the item from that list would do to the verdict:

    - KEY: establishes a central factual premise underlying the gold verdict.
      Removing it from the list likely breaks the verdict.
    - AUXILIARY: corroborates, qualifies, or strengthens the main justification
      without constituting its principal evidential basis. Removing it would not
      break the verdict.
    - BACKGROUND: provides context for understanding the claim or its
      circumstances without directly contributing to the justification.

    Losing a key item leaves no way to the verdict; losing the others is left to
    the sufficiency ensemble to judge.
    """

    KEY = "key"
    AUXILIARY = "auxiliary"
    BACKGROUND = "background"

    @classmethod
    def _missing_(cls, value):
        """Reads role values written before the roles were renamed. The stored rows
        are left untouched; they are mapped on load instead."""
        if isinstance(value, str):
            alias = LEGACY_ROLE_ALIASES.get(value.strip().lower())
            if alias is not None:
                return cls(alias)
        return None


#: Role values of earlier pipeline versions that may still be stored in the
#: database, mapped to their current equivalent.
LEGACY_ROLE_ALIASES = {"essential": EvidenceRole.KEY.value}

#: Rejection reasons (`claims.gold_evidence_reason`) of earlier pipeline versions,
#: mapped to their current equivalent.
LEGACY_REASON_ALIASES = {"essential_evidence_lost": "key_evidence_lost"}


def normalize_reason(reason: str | None) -> str | None:
    """The current name of a stored rejection reason."""
    if reason is None:
        return None
    return LEGACY_REASON_ALIASES.get(reason, reason)


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
    """Where a cited source sits relative to the two cutoffs (Spec §3.3).

    Both are computed from `t_e` and the claim's reference times, never predicted.
    They live on the `Citation`, not on the `Source`: a source is shared by every
    claim that cites it, and `t_c`/`t_f` differ from claim to claim. The one
    judgement that needs a model - whether the evidence rests on a later event - is
    about the proposition and lives on `Evidence.later_event`."""

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


#: Citation kinds that do not refer to a publication at all, so §3.1 and §3.2 do
#: not apply to them: a tool is an instrument, and offline evidence (a phone call,
#: an interview) never went online. They are never retrieved, need no locator, and
#: are admitted on the extraction alone, subject to the dating policy.
UNRETRIEVABLE_KINDS = (SourceKind.TOOL, SourceKind.OFFLINE)


def normalize_locator(locator: str) -> str:
    """The form of a URL under which it identifies a `Source`.

    Only what is case-insensitive by definition is lowercased - the scheme and the
    host. Paths and queries are case-sensitive on many platforms (video IDs, short
    links), so lowercasing them would merge different publications. A trailing
    slash and the fragment are dropped, since neither changes what is served."""
    locator = locator.strip()
    try:
        parts = urlsplit(locator)
    except ValueError:
        return locator
    if not parts.scheme or not parts.netloc:
        return locator.rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(),
                       parts.path.rstrip("/"), parts.query, ""))


def locator_key(locator: str) -> str:
    """The key a `Source` is stored under: a digest of its normalized locator.
    Long enough to be collision-free across the whole database, unlike the 32-bit
    hashes used for per-claim keys, and short enough to index."""
    return hashlib.sha1(normalize_locator(locator).encode("utf-8")).hexdigest()


class Source(VeritasBaseModel):
    """A publication, identified by its URL, as retrieved through scrapeMM.

    A source is **global**: it exists once, however many evidence items - of however
    many claims - cite it. Everything decided about it depends on the URL alone -
    can it still be retrieved, what does it say, when did it become available, is it
    a professional fact-check - so each of these is established exactly once.
    Everything that depends on *what* it was cited for (proximity, faithfulness, the
    cutoffs of the citing claim) lives on the `Citation` instead.

    Sources are write-once: once retrieved, a source is never refetched implicitly,
    because other claims' decisions rest on what was stored. The only transitions
    are "deferred -> retrieved" and an explicit re-retrieval (`reset_retrieval`),
    after which every citation judged against the old state counts as stale.
    """

    locator: str
    raw_content: str | None = None  #: The content as scraped with scrapeMM

    #: t_e - when the source became publicly accessible. None if no clear date is
    #: associated with it.
    available_since: datetime | None = None
    #: How `available_since` was obtained: 'meta' | 'llm' | None
    dating_method: str | None = None
    #: The scrapeMM method that served the retrieval.
    retrieval_method: str | None = None
    #: When the source was retrieved. None until it has been.
    accessed_at: datetime | None = None

    # --- Stage 2 outcome, decided once per URL ---
    accessible: bool | None = None  # §3.1
    #: Whether the publisher registry lists the source's publisher as an IFCN or
    #: EFCSN signatory (§3.3 (2)). None until looked up.
    is_fact_check: bool | None = None
    #: Why the retrieval failed, including error messages.
    retrieval_error: str | None = None
    #: Set while the source waits out a rate limit or Archive.today's access check;
    #: it is neither accessible nor inaccessible until then.
    deferred_until: datetime | None = None

    @property
    def key(self) -> str:
        return locator_key(self.locator)

    @property
    def content(self) -> MultimodalSequence | None:
        """The scraped source content as a MultimodalSequence."""
        return MultimodalSequence(self.raw_content) if self.raw_content else None

    @property
    def domain(self) -> str | None:
        return get_domain(self.locator)

    @property
    def retrieved(self) -> bool:
        """True once a retrieval attempt reached a decision (accessible or not)."""
        return self.accessible is not None

    @property
    def decided(self) -> bool:
        """True once nothing about the source is left to establish: it was
        retrieved, or it was found to be a fact-check and needs no retrieval."""
        return self.retrieved or bool(self.is_fact_check)

    @property
    def deferred(self) -> bool:
        """True while the cooldown after a rate limit is still running."""
        return self.deferred_until is not None and self.deferred_until > datetime.now()

    def defer(self, hours: int) -> None:
        """Postpones this source, extending an existing window but never shortening
        it. Unlike `Deferrable.defer` this does not write to the DB."""
        until = datetime.now() + timedelta(hours=hours)
        if self.deferred_until is None or until > self.deferred_until:
            self.deferred_until = until

    def reset_retrieval(self) -> None:
        """Forgets everything the last retrieval established, so that the next
        Stage 2 run retrieves the source again. The registry lookup is kept: it
        does not depend on what the page currently serves."""
        self.raw_content = None
        self.available_since = None
        self.dating_method = None
        self.retrieval_method = None
        self.accessed_at = None
        self.accessible = None
        self.retrieval_error = None
        self.deferred_until = None

    def adopt(self, other: "Source") -> None:
        """Takes over the stored state of `other`, the same source as persisted.
        Used when another claim decided the source in the meantime."""
        for field_name in type(self).model_fields:
            setattr(self, field_name, getattr(other, field_name))

    def __str__(self) -> str:
        return self.locator


class Citation(VeritasBaseModel):
    """That an evidence item cites a source for its proposition: the relation
    between `Evidence` and `Source`, together with everything that depends on
    both sides.

    - `name` and `kind` describe the source *as cited*. Two articles may label the
      same URL differently, and the kind is a judgement about the citation: whether
      a page was used as a tool decides whether it is retrieved at all.
    - `proximity` is relative to the proposition: a news report is primary for what
      the outlet itself said and secondary for the event it reports.
    - `faithfulness` asks whether *this* source supports *this* proposition.
    - `temporal_validation` places the source relative to the citing claim's
      cutoffs.

    A citation without a `source` is a source the article never located - a tool
    without a page, an interview, or material it cites without linking.
    """

    evidence_id: int | None = None
    source_id: int | None = None
    #: The cited publication. Shared: every citation of the same URL within one
    #: claim refers to the same instance. Stored in its own table, hence excluded
    #: from the citation's own serialization.
    source: Source | None = Field(default=None, exclude=True)

    name: str  # Name of the publishing organization or person, as cited
    kind: SourceKind = SourceKind.OTHER
    proximity: ProximityLevel = ProximityLevel.SECONDARY
    #: For a tool or offline citation: when its finding became available (the date
    #: of an interview, say), if known. Such a citation is not a publication, so
    #: there is no source to read a publication time off.
    date_as_cited: datetime | None = None

    # --- Stage 2 outcome ---
    faithfulness: Faithfulness | None = None  # §3.2
    temporal_validation: TemporalValidation | None = None  # §3.3
    #: Determined algorithmically, see `veritas.gold_evidence.admissibility`.
    admissible: bool | None = None
    inadmissibility_reason: str | None = None
    #: When the admissibility was last decided. A citation judged before its source
    #: was (re-)retrieved is stale and is judged again.
    judged_at: datetime | None = None
    #: An error that interrupted judging the citation (not the retrieval, which is
    #: recorded on the source).
    error: str | None = None

    @property
    def locator(self) -> str | None:
        return self.source.locator if self.source else None

    @property
    def domain(self) -> str | None:
        return self.source.domain if self.source else None

    @property
    def exempt(self) -> bool:
        """Whether the citation is of a kind that is never retrieved."""
        return self.kind in UNRETRIEVABLE_KINDS

    @property
    def key(self) -> str:
        """Identifies the citation within its evidence item: the source it cites,
        or - for a citation without one - its kind and name."""
        if self.source is not None:
            return self.source.key
        return f"unlocated:{self.kind.value}:{' '.join(self.name.lower().split())}"

    @property
    def available_since(self) -> datetime | None:
        """t_e as far as this citation is concerned. A tool or offline citation is
        not a publication, so the publication date of a page it happens to link
        (the tool's homepage, say) says nothing about when its finding existed; its
        own `date_as_cited` counts instead."""
        if self.exempt:
            return self.date_as_cited
        if self.source is None:
            return None
        return self.source.available_since

    @property
    def deferred(self) -> bool:
        """True while the cited source waits out a rate limit."""
        return not self.exempt and self.source is not None and self.source.deferred

    @property
    def judged(self) -> bool:
        return self.admissible is not None

    @property
    def stale(self) -> bool:
        """True if the source was retrieved after this citation was judged, i.e.
        the judgement rests on content that is no longer the stored one."""
        if not self.judged or self.source is None or self.exempt:
            return False
        accessed_at = self.source.accessed_at
        return (accessed_at is not None and self.judged_at is not None
                and accessed_at > self.judged_at)

    @property
    def filtered(self) -> bool:
        """True once Stage 2 decided this citation against the current source."""
        return self.judged and not self.stale

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
    to establish its verdict, together with every citation of a source reporting it.

    Evidence is the *aggregation layer*: the proposition and its role belong here,
    retrieval and dating to the `Source`, and faithfulness to the `Citation` that
    connects the two. Two outlets reporting the same fact are two citations of one
    evidence item, not two evidence items - so losing one of them costs nothing, and
    the item is discarded only when it has lost every citation it had.
    """

    claim_id: int
    review_id: int | None = None  # The review whose article this was extracted from
    article_id: int | None = None

    #: The atomic factual information contributed by the sources, incl. media references
    proposition: str

    #: Every citation of a source reporting this proposition. At least one. Stored in
    #: their own table, hence excluded from the item's own serialization.
    citations: list[Citation] = Field(default_factory=list, exclude=True)

    role: EvidenceRole = EvidenceRole.AUXILIARY

    @field_validator("role", mode="before")
    @classmethod
    def _read_legacy_role(cls, value):
        """Items stored before the roles were renamed carry the old value."""
        if isinstance(value, str):
            return LEGACY_ROLE_ALIASES.get(value.strip().lower(), value)
        return value

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
    def sources(self) -> list[Source]:
        """The distinct sources the item cites, in citation order."""
        seen, sources = set(), []
        for citation in self.citations:
            source = citation.source
            if source is not None and id(source) not in seen:
                seen.add(id(source))
                sources.append(source)
        return sources

    @property
    def relies_on_later_event(self) -> bool:
        """Whether §3.3 (3) rules this item out. A property of the proposition, so
        it condemns the item however many citations still report it."""
        return bool(self.later_event and self.later_event.change_detected)

    @property
    def admissible(self) -> bool | None:
        """Whether the proposition still has a usable citation.

        True as soon as one citation is admissible, False once every citation was
        decided and rejected, and None while any citation is still undecided - an
        item is not decided until its last citation is. A later event rules the item
        out regardless of its citations."""
        if self.relies_on_later_event:
            return False
        if not self.citations:
            return False
        if any(citation.admissible for citation in self.citations):
            return True
        if all(citation.judged for citation in self.citations):
            return False
        return None

    @property
    def admissible_citations(self) -> list[Citation]:
        return [citation for citation in self.citations if citation.admissible]

    @property
    def inadmissibility_reason(self) -> str | None:
        """Why the item was discarded. An item with several citations reports the
        first reason, since the reasons differ per citation and the item needs a
        single label for the exports."""
        if self.relies_on_later_event:
            return "later_event"
        if self.admissible is not False:
            return None
        reasons = [citation.inadmissibility_reason for citation in self.citations
                   if citation.inadmissibility_reason]
        return reasons[0] if reasons else None

    @property
    def filtered(self) -> bool:
        """True once Stage 2 decided every citation against its current source."""
        return bool(self.citations) and all(citation.filtered for citation in self.citations)

    @property
    def deferred(self) -> bool:
        """True while any cited source waits out a rate limit: the item is not
        decided until that source has been retried."""
        return any(citation.deferred for citation in self.citations)

    @property
    def is_key(self) -> bool:
        return self.role == EvidenceRole.KEY

    @property
    def available_since(self) -> datetime | None:
        """t_e of the item: the *earliest* time at which any of its citations made
        the proposition available. That is when the proposition could first be read."""
        times = [citation.available_since for citation in self.citations
                 if citation.available_since is not None]
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

    def time_to_claim(self, t_c: datetime | None) -> float | None:
        """(t_e - t_c) in days. None if either timestamp is unknown."""
        return _delta_days(self.available_since, t_c)

    def time_to_fact_check(self, t_f: datetime | None) -> float | None:
        """(t_e - t_f) in days. None if either timestamp is unknown."""
        return _delta_days(self.available_since, t_f)

    def __str__(self) -> str:
        citations = "\n".join(f"  -- {citation}" for citation in self.citations)
        return f"[{self.role.value}] {self.proposition}\n{citations}"


class VerdictRationale(VeritasBaseModel):
    """The reasoning that bridges the gap between the evidence and the verdict.

    Not every claim is settled by external evidence: some are settled by arithmetic,
    by a logical contradiction inside the claim, or by what the claim's own image
    plainly shows. The rationale captures that step of the fact-check's argument, so
    that such instances can be analyzed instead of looking evidence-less.

    It carries reasoning and commonsense knowledge only (arithmetic, logic,
    comparisons, visual analysis). It does not restate the evidence, refers to no
    specific evidence item, and introduces no externally available information -
    such information would be evidence, and belongs in `Evidence` where it can be
    retrieved, dated, and checked for faithfulness. It also never states the
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
