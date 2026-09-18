"""Algorithmic admissibility of reconstructed evidence (Spec §3).

Pure functions only - no I/O, no LLM calls - so the decision rule that produces
the headline numbers is fully unit-testable and auditable.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Iterable

from veritas.gold_evidence import (
    CONDITION_CLAIM,
    CONDITION_FACT_CHECK,
    faithfulness_threshold as default_faithfulness_threshold,
    min_extraction_confidence as default_min_confidence,
    undated_policy as default_undated_policy,
)
from veritas.gold_evidence.models import (
    Evidence,
    EvidenceRole,
    EvidenceSource,
    SourceKind,
    to_naive,
)


class InadmissibilityReason(str, Enum):
    """Why a *source* was discarded. Mutually exclusive: the first applicable
    reason in `REASON_ORDER` is recorded. An evidence item inherits the reason of
    its sources once it has lost all of them."""

    NOT_FILTERED = "not_filtered"  # Stage 2 did not complete for this source
    LOW_CONFIDENCE = "low_extraction_confidence"  # §2
    INACCESSIBLE = "inaccessible"  # §3.1
    UNDATED = "undated_source"  # §3.1 / §3.3, see `undated_policy`
    UNFAITHFUL = "unfaithful"  # §3.2
    AFTER_FACT_CHECK = "after_fact_check"  # §3.3 (1), t_e > t_f
    VERDICT_LEAK = "fact_check_source"  # §3.3 (2)
    #: §3.3 (3). Decided per *item* (`Evidence.later_event`), not per source: it is
    #: the proposition that rests on a later event, not one place it can be read.
    LATER_EVENT = "later_event"


#: Evaluation order. The first violated criterion becomes the recorded reason, so
#: the reported reasons partition the rejected items rather than double-counting.
REASON_ORDER = (
    InadmissibilityReason.VERDICT_LEAK,  # decidable from the source alone
    InadmissibilityReason.NOT_FILTERED,
    InadmissibilityReason.LOW_CONFIDENCE,
    InadmissibilityReason.INACCESSIBLE,
    InadmissibilityReason.UNDATED,  # checked after the completeness re-check below
    InadmissibilityReason.UNFAITHFUL,
    InadmissibilityReason.AFTER_FACT_CHECK,
    InadmissibilityReason.LATER_EVENT,
)

#: Source kinds that are not publications at all, so §3.1 and §3.2 do not apply to
#: them: a tool is an instrument, and offline evidence (a phone call, an interview)
#: never went online. Neither has to carry a locator or a publication time; they are
#: admitted on the extraction alone, subject to the dating policy below.
UNRETRIEVABLE_KINDS = (SourceKind.TOOL, SourceKind.OFFLINE)

#: How to treat sources without a determinable publication time (`t_e is None`).
#: - "tool_only" (default): keep only the `UNRETRIEVABLE_KINDS`. Evidence that cannot
#:   be dated cannot be shown to predate the cutoff, so keeping it would bias the
#:   analysis towards "gold verdict recoverable". Tools and offline evidence are
#:   exempt because they are not publications and have no release time to find:
#:   requiring one would discard them categorically rather than on the merits.
#: - "permissive": keep all undated sources.
#: - "strict": discard all undated sources, including tools and offline evidence.
UNDATED_POLICIES = ("tool_only", "permissive", "strict")


def keeps_undated(kind: SourceKind, undated_policy: str = None) -> bool:
    """Whether an undated source of the given kind may remain admissible."""
    if undated_policy is None:
        undated_policy = default_undated_policy
    assert undated_policy in UNDATED_POLICIES, f"Unknown undated policy: {undated_policy}"
    if undated_policy == "permissive":
        return True
    if undated_policy == "strict":
        return False
    return kind in UNRETRIEVABLE_KINDS


def compute_temporal_bounds(
        available_since: datetime | None,
        t_c: datetime | None,
        t_f: datetime | None,
) -> tuple[bool, bool]:
    """Returns `(before_claim, before_fact_check)`, i.e. whether `t_e <= t_c` and
    `t_e <= t_f`. An unknown `t_e` (or an unknown reference time) is *not* counted
    as a violation here; undated sources are handled by `undated_policy` instead."""
    t_e = to_naive(available_since)
    if t_e is None:
        return True, True
    t_c, t_f = to_naive(t_c), to_naive(t_f)
    before_claim = True if t_c is None else t_e <= t_c
    before_fact_check = True if t_f is None else t_e <= t_f
    return before_claim, before_fact_check


def determine_inadmissibility(
        source: EvidenceSource,
        *,
        extraction_confidence: float = 1.0,
        faithfulness_threshold: float = None,
        min_extraction_confidence: float = None,
        undated_policy: str = None,
) -> InadmissibilityReason | None:
    """Returns the reason why the source is inadmissible, or None if it is
    admissible. Admissibility is evaluated against the *loose* cutoff `t_f`; the
    strict condition `E_c` is a pure filter on `before_claim` afterwards (valid
    because `t_c <= t_f` always holds).

    Sources of an `UNRETRIEVABLE_KINDS` kind skip the accessibility and faithfulness
    criteria - there is nothing to retrieve and nothing to re-read - and are decided
    by the extraction confidence and the dating policy alone.

    `extraction_confidence` belongs to the evidence item, not to the source; it is
    passed in so that this stays a pure function of what it is handed.
    """
    if faithfulness_threshold is None:
        faithfulness_threshold = default_faithfulness_threshold
    if min_extraction_confidence is None:
        min_extraction_confidence = default_min_confidence

    kind = source.kind
    exempt = kind in UNRETRIEVABLE_KINDS

    # A source that is itself a professional fact-check leaks the verdict, whether it
    # was extracted as one or identified as one by the publisher registry (§3.3 (2)).
    # This holds regardless of whether the source was ever filtered.
    if kind == SourceKind.FACT_CHECK:
        return InadmissibilityReason.VERDICT_LEAK

    # A source the article never located cannot be retrieved, now or ever. It is
    # recorded rather than dropped at extraction, so that "the fact-check cited
    # something it did not link" shows up in the numbers instead of vanishing.
    if not exempt and not source.locator:
        return InadmissibilityReason.INACCESSIBLE

    if not exempt and source.accessible is None:
        return InadmissibilityReason.NOT_FILTERED

    if extraction_confidence < min_extraction_confidence:
        return InadmissibilityReason.LOW_CONFIDENCE

    if not exempt:
        if not source.accessible:
            return InadmissibilityReason.INACCESSIBLE

        # An accessible source without these was interrupted mid-filtering; report
        # that rather than a substantive reason it was never evaluated for.
        if source.faithfulness is None or source.temporal_validation is None:
            return InadmissibilityReason.NOT_FILTERED

    if source.available_since is None and not keeps_undated(kind, undated_policy):
        return InadmissibilityReason.UNDATED

    if not exempt and source.faithfulness.assessment < faithfulness_threshold:
        return InadmissibilityReason.UNFAITHFUL

    temporal = source.temporal_validation
    if temporal is not None and not temporal.before_fact_check:
        return InadmissibilityReason.AFTER_FACT_CHECK

    return None


def apply_admissibility(source: EvidenceSource, *,
                        extraction_confidence: float = 1.0, **kwargs) -> EvidenceSource:
    """Sets `admissible` and `inadmissibility_reason` on the source in place."""
    reason = determine_inadmissibility(source, extraction_confidence=extraction_confidence,
                                       **kwargs)
    source.admissible = reason is None
    source.inadmissibility_reason = reason.value if reason else None
    return source


def apply_admissibility_to_item(evidence: Evidence, **kwargs) -> Evidence:
    """Decides every source of the item. The item itself derives its admissibility
    from them: it survives as long as one source does."""
    for source in evidence.sources:
        if not source.deferred:
            apply_admissibility(source,
                                extraction_confidence=evidence.extraction_confidence,
                                **kwargs)
    return evidence


def source_in_condition(source: EvidenceSource, condition: str) -> bool:
    """Whether an *admissible* source belongs to the evidence set of the condition.

    - `E_f = {s : s admissible}`            (t_e <= t_f, enforced by admissibility)
    - `E_c = {s : s admissible, t_e <= t_c}`
    """
    if not source.admissible:
        return False
    if condition == CONDITION_FACT_CHECK:
        return True
    if condition == CONDITION_CLAIM:
        temporal = source.temporal_validation
        if temporal is not None:
            return temporal.before_claim
        # Sources exempt from Stage 2 carry no temporal validation. An unknown t_e
        # satisfies both cutoffs, exactly as `compute_temporal_bounds` treats it.
        return source.available_since is None
    raise ValueError(f"Unknown condition: {condition}")


def in_condition(evidence: Evidence, condition: str) -> bool:
    """Whether the item still has a source in the condition's evidence set. One
    surviving source is enough: it establishes the proposition on its own."""
    if evidence.relies_on_later_event:
        return False
    return any(source_in_condition(source, condition) for source in evidence.sources)


def select(evidence: Iterable[Evidence], condition: str) -> list[Evidence]:
    """The evidence set for the given condition, ordered by role then confidence."""
    selected = [e for e in evidence if in_condition(e, condition)]
    role_rank = {"essential": 0, "auxiliary": 1, "background": 2}
    return sorted(
        selected,
        key=lambda e: (role_rank.get(e.role.value, 3), -e.extraction_confidence),
    )


def restrict_to_condition(evidence: Iterable[Evidence], condition: str) -> list[Evidence]:
    """The condition's evidence set, each item carrying *only* the sources that
    belong to that condition.

    What the sufficiency ensemble is shown must be exactly what the condition
    provides: an item may be in `E_c` through one source while another of its
    sources only became available later, and naming that later source in the prompt
    would hand the strict condition evidence it does not have."""
    restricted = []
    for item in select(evidence, condition):
        sources = [s for s in item.sources if source_in_condition(s, condition)]
        restricted.append(item.model_copy(update={"sources": sources}))
    return restricted


# ---------------------------------------------------------------------------
# Essential evidence
# ---------------------------------------------------------------------------

def essential(evidence: Iterable[Evidence]) -> list[Evidence]:
    """The items the verdict rationale rests on."""
    return [item for item in evidence if item.is_essential]


def lost_essential(evidence: Iterable[Evidence]) -> list[Evidence]:
    """Essential items that Stage 2 left without a single admissible source.

    This is what disqualifies an instance: the rationale rests on the proposition
    such an item asserts, and no reconstructible source for it survived. Losing
    *some* of an item's sources costs nothing - one surviving source establishes
    the proposition just as well as three would."""
    return [item for item in essential(evidence) if not item.admissible]


def missing_essential(evidence: Iterable[Evidence], condition: str) -> list[Evidence]:
    """Essential items that have no source inside the condition's evidence set.

    Unlike `lost_essential` this is not a defect of the reconstruction: an item may
    be perfectly admissible and still fall outside `E_c` because it only became
    available after the claim. That is precisely the finding the analysis is after,
    and it makes the condition insufficient without asking the ensemble - the
    rationale depends on a proposition this condition cannot support."""
    return [item for item in essential(evidence) if not in_condition(item, condition)]


def in_window(source: EvidenceSource) -> bool:
    """Whether the source falls into the studied interval `t_c < t_e <= t_f`."""
    temporal = source.temporal_validation
    if not temporal or source.available_since is None:
        return False
    return temporal.before_fact_check and not temporal.before_claim
