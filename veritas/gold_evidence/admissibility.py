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
from veritas.gold_evidence.models import Evidence, SourceKind, to_naive


class InadmissibilityReason(str, Enum):
    """Why an evidence candidate was discarded. Mutually exclusive: the first
    applicable reason in `REASON_ORDER` is recorded."""

    NOT_FILTERED = "not_filtered"  # Stage 2 did not complete for this item
    LOW_CONFIDENCE = "low_extraction_confidence"  # §2
    INACCESSIBLE = "inaccessible"  # §3.1
    UNDATED = "undated_source"  # §3.1 / §3.3, see `undated_policy`
    UNFAITHFUL = "unfaithful"  # §3.2
    AFTER_FACT_CHECK = "after_fact_check"  # §3.3 (1), t_e > t_f
    VERDICT_LEAK = "fact_check_source"  # §3.3 (2)
    LATER_EVENT = "later_event"  # §3.3 (3)


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
        evidence: Evidence,
        *,
        faithfulness_threshold: float = None,
        min_extraction_confidence: float = None,
        undated_policy: str = None,
) -> InadmissibilityReason | None:
    """Returns the reason why the evidence is inadmissible, or None if it is
    admissible. Admissibility is evaluated against the *loose* cutoff `t_f`;
    the strict condition `E_claim` is a pure filter on `before_claim` afterwards
    (valid because `t_c <= t_f` always holds).

    Items of an `UNRETRIEVABLE_KINDS` source skip the accessibility and faithfulness
    criteria - there is nothing to retrieve and nothing to re-read - and are decided
    by the extraction confidence and the dating policy alone.
    """
    if faithfulness_threshold is None:
        faithfulness_threshold = default_faithfulness_threshold
    if min_extraction_confidence is None:
        min_extraction_confidence = default_min_confidence

    kind = evidence.source.kind
    exempt = kind in UNRETRIEVABLE_KINDS

    # A source that is itself a professional fact-check leaks the verdict, whether it
    # was extracted as one or identified as one by the publisher registry (§3.3 (2)).
    # This holds regardless of whether the item was ever filtered.
    if kind == SourceKind.FACT_CHECK:
        return InadmissibilityReason.VERDICT_LEAK

    if not exempt and evidence.accessible is None:
        return InadmissibilityReason.NOT_FILTERED

    if evidence.extraction_confidence < min_extraction_confidence:
        return InadmissibilityReason.LOW_CONFIDENCE

    if not exempt:
        if not evidence.accessible:
            return InadmissibilityReason.INACCESSIBLE

        # An accessible item without these was interrupted mid-filtering; report that
        # rather than a substantive reason it was never actually evaluated for.
        if evidence.faithfulness is None or evidence.temporal_validation is None:
            return InadmissibilityReason.NOT_FILTERED

    if evidence.available_since is None and not keeps_undated(kind, undated_policy):
        return InadmissibilityReason.UNDATED

    if not exempt and evidence.faithfulness.assessment < faithfulness_threshold:
        return InadmissibilityReason.UNFAITHFUL

    temporal = evidence.temporal_validation
    if temporal is not None and not temporal.before_fact_check:
        return InadmissibilityReason.AFTER_FACT_CHECK

    if temporal is not None and temporal.later_event:
        return InadmissibilityReason.LATER_EVENT

    return None


def apply_admissibility(evidence: Evidence, **kwargs) -> Evidence:
    """Sets `admissible` and `inadmissibility_reason` on the item in place."""
    reason = determine_inadmissibility(evidence, **kwargs)
    evidence.admissible = reason is None
    evidence.inadmissibility_reason = reason.value if reason else None
    return evidence


def in_condition(evidence: Evidence, condition: str) -> bool:
    """Whether an *admissible* item belongs to the evidence set of the condition.

    - `E_factcheck = {e : e admissible}`           (t_e <= t_f, enforced by admissibility)
    - `E_claim     = {e : e admissible, t_e <= t_c}`
    """
    if not evidence.admissible:
        return False
    if condition == CONDITION_FACT_CHECK:
        return True
    if condition == CONDITION_CLAIM:
        temporal = evidence.temporal_validation
        if temporal is not None:
            return temporal.before_claim
        # Items exempt from Stage 2 carry no temporal validation. An unknown t_e
        # satisfies both cutoffs, exactly as `compute_temporal_bounds` treats it.
        return evidence.available_since is None
    raise ValueError(f"Unknown condition: {condition}")


def select(evidence: Iterable[Evidence], condition: str) -> list[Evidence]:
    """The evidence set for the given condition, ordered by role then confidence."""
    selected = [e for e in evidence if in_condition(e, condition)]
    role_rank = {"essential": 0, "auxiliary": 1, "background": 2}
    return sorted(
        selected,
        key=lambda e: (role_rank.get(e.role.value, 3), -e.extraction_confidence),
    )


def in_window(evidence: Evidence) -> bool:
    """Whether the item falls into the studied interval `t_c < t_e <= t_f`."""
    temporal = evidence.temporal_validation
    if not temporal or evidence.available_since is None:
        return False
    return temporal.before_fact_check and not temporal.before_claim
