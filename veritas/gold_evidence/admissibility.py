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
    UNRETRIEVABLE_KINDS,
    Citation,
    Evidence,
    EvidenceRole,
    SourceKind,
    to_naive,
)


class InadmissibilityReason(str, Enum):
    """Why a *citation* was discarded. Mutually exclusive: the first applicable
    reason in `REASON_ORDER` is recorded. An evidence item inherits the reason of
    its citations once it has lost all of them.

    Most reasons are properties of the cited `Source` and hence the same for every
    citation of it (verdict leak, inaccessible, undated); faithfulness, extraction
    confidence and the cutoffs depend on the citation."""

    NOT_FILTERED = "not_filtered"  # Stage 2 did not complete for this citation
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

# `UNRETRIEVABLE_KINDS` (tool, offline) is defined next to the model and imported
# from here by the other stages: §3.1 and §3.2 do not apply to such citations.

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
        citation: Citation,
        *,
        extraction_confidence: float = 1.0,
        faithfulness_threshold: float = None,
        min_extraction_confidence: float = None,
        undated_policy: str = None,
) -> InadmissibilityReason | None:
    """Returns the reason why the citation is inadmissible, or None if it is
    admissible. Admissibility is evaluated against the *loose* cutoff `t_f`; the
    strict condition `E_c` is a pure filter on `before_claim` afterwards (valid
    because `t_c <= t_f` always holds).

    Citations of an `UNRETRIEVABLE_KINDS` kind skip the accessibility and
    faithfulness criteria - there is nothing to retrieve and nothing to re-read -
    and are decided by the extraction confidence and the dating policy alone.

    `extraction_confidence` belongs to the evidence item; it is passed in so that
    this stays a pure function of what it is handed.
    """
    if faithfulness_threshold is None:
        faithfulness_threshold = default_faithfulness_threshold
    if min_extraction_confidence is None:
        min_extraction_confidence = default_min_confidence

    kind = citation.kind
    exempt = citation.exempt
    source = citation.source

    # A source that is itself a professional fact-check leaks the verdict, whether it
    # was extracted as one or identified as one by the publisher registry (§3.3 (2)).
    # Tools are exempt from the registry: a fact-checker may host the tool it used.
    if kind == SourceKind.FACT_CHECK or (not exempt and source is not None
                                         and source.is_fact_check):
        return InadmissibilityReason.VERDICT_LEAK

    # A source the article never located cannot be retrieved, now or ever. It is
    # recorded rather than dropped at extraction, so that "the fact-check cited
    # something it did not link" shows up in the numbers instead of vanishing.
    if not exempt and source is None:
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
        if citation.faithfulness is None or citation.temporal_validation is None:
            return InadmissibilityReason.NOT_FILTERED

    if citation.available_since is None and not keeps_undated(kind, undated_policy):
        return InadmissibilityReason.UNDATED

    if not exempt and citation.faithfulness.assessment < faithfulness_threshold:
        return InadmissibilityReason.UNFAITHFUL

    temporal = citation.temporal_validation
    if temporal is not None and not temporal.before_fact_check:
        return InadmissibilityReason.AFTER_FACT_CHECK

    return None


def apply_admissibility(citation: Citation, *,
                        extraction_confidence: float = 1.0, **kwargs) -> Citation:
    """Sets `admissible`, `inadmissibility_reason` and `judged_at` in place."""
    reason = determine_inadmissibility(citation, extraction_confidence=extraction_confidence,
                                       **kwargs)
    citation.admissible = reason is None
    citation.inadmissibility_reason = reason.value if reason else None
    citation.judged_at = datetime.now()
    return citation


def apply_admissibility_to_item(evidence: Evidence, **kwargs) -> Evidence:
    """Decides every citation of the item. The item itself derives its
    admissibility from them: it survives as long as one citation does."""
    for citation in evidence.citations:
        if not citation.deferred:
            apply_admissibility(citation,
                                extraction_confidence=evidence.extraction_confidence,
                                **kwargs)
    return evidence


def citation_in_condition(citation: Citation, condition: str) -> bool:
    """Whether an *admissible* citation belongs to the evidence set of the condition.

    - `E_f = {c : c admissible}`            (t_e <= t_f, enforced by admissibility)
    - `E_c = {c : c admissible, t_e <= t_c}`
    """
    if not citation.admissible:
        return False
    if condition == CONDITION_FACT_CHECK:
        return True
    if condition == CONDITION_CLAIM:
        temporal = citation.temporal_validation
        if temporal is not None:
            return temporal.before_claim
        # Citations exempt from Stage 2 carry no temporal validation. An unknown t_e
        # satisfies both cutoffs, exactly as `compute_temporal_bounds` treats it.
        return citation.available_since is None
    raise ValueError(f"Unknown condition: {condition}")


def in_condition(evidence: Evidence, condition: str) -> bool:
    """Whether the item still has a citation in the condition's evidence set. One
    surviving citation is enough: it establishes the proposition on its own."""
    if evidence.relies_on_later_event:
        return False
    return any(citation_in_condition(c, condition) for c in evidence.citations)


def select(evidence: Iterable[Evidence], condition: str) -> list[Evidence]:
    """The evidence set for the given condition, ordered by role then confidence."""
    selected = [e for e in evidence if in_condition(e, condition)]
    role_rank = {"key": 0, "auxiliary": 1, "background": 2}
    return sorted(
        selected,
        key=lambda e: (role_rank.get(e.role.value, 3), -e.extraction_confidence),
    )


def restrict_to_condition(evidence: Iterable[Evidence], condition: str) -> list[Evidence]:
    """The condition's evidence set, each item carrying *only* the citations that
    belong to that condition.

    What the sufficiency ensemble is shown must be exactly what the condition
    provides: an item may be in `E_c` through one citation while another of its
    sources only became available later, and naming that later source in the
    prompt would hand the strict condition evidence it does not have."""
    restricted = []
    for item in select(evidence, condition):
        citations = [c for c in item.citations if citation_in_condition(c, condition)]
        restricted.append(item.model_copy(update={"citations": citations}))
    return restricted


# ---------------------------------------------------------------------------
# Key evidence
# ---------------------------------------------------------------------------

def key_evidence(evidence: Iterable[Evidence]) -> list[Evidence]:
    """The items establishing a central factual premise of the gold verdict:
    removing any of them likely breaks the verdict."""
    return [item for item in evidence if item.is_key]


def lost_key(evidence: Iterable[Evidence]) -> list[Evidence]:
    """Key items that Stage 2 left without a single admissible citation.

    This is what disqualifies an instance: the verdict rests on what such an item
    establishes, and no reconstructible source for it survived. Losing *some* of an
    item's citations costs nothing - one surviving citation establishes the
    proposition just as well as three would. Losing an *auxiliary* item does not
    disqualify either: by definition the verdict does not break without it, and
    whether the rest still carries it is for the sufficiency ensemble to judge."""
    return [item for item in key_evidence(evidence) if not item.admissible]


def lost_auxiliary(evidence: Iterable[Evidence]) -> list[Evidence]:
    """Auxiliary items that Stage 2 left without a single admissible citation. Unlike
    `lost_key`, this does not disqualify an instance: it is reported to show
    how much evidence the instances survived losing."""
    return [item for item in evidence
            if item.role == EvidenceRole.AUXILIARY and item.admissible is False]


def missing_key(evidence: Iterable[Evidence], condition: str) -> list[Evidence]:
    """Key items that have no citation inside the condition's evidence set.

    Unlike `lost_key` this is not a defect of the reconstruction: an item may be
    perfectly admissible and still fall outside `E_c` because it only became
    available after the claim. That is precisely the finding the analysis is after,
    and it makes the condition insufficient without asking the ensemble - the
    verdict likely breaks without the missing item.

    Only key items short-circuit a condition. An auxiliary item that only appeared
    after the claim does not carry the verdict, so treating it as missing would
    count the condition as insufficient for no reason."""
    return [item for item in key_evidence(evidence) if not in_condition(item, condition)]


def in_window(citation: Citation) -> bool:
    """Whether the cited source falls into the studied interval `t_c < t_e <= t_f`
    of the citing claim."""
    temporal = citation.temporal_validation
    if not temporal or citation.available_since is None:
        return False
    return temporal.before_fact_check and not temporal.before_claim
