"""Orchestration of the three Gold Evidence Reconstruction stages for one claim.

The gold verdict is never modified. When sufficient valid evidence cannot be
reconstructed, the instance is *rejected*, which is recorded in the additive
`claims.gold_evidence_*` columns only - `claims.dismissed` stays untouched, so a
rejected instance remains a valid VeriTaS claim.

An *empty* evidence set is not a rejection: a verdict can rest on arithmetic, on a
contradiction inside the claim, or on what the claim's own media shows, and the
verdict rationale carries those cases. What disqualifies an instance is a broken
argument - an essential evidence item that Stage 2 left without a single admissible
source, so that the rationale no longer has a reconstructible basis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from veritas.common import Claim, Verdict
from veritas.db import db
from veritas.gold_evidence import (
    CONDITION_CLAIM,
    CONDITION_FACT_CHECK,
    CONDITIONS,
    STATUS_ACCEPTED,
    STATUS_DEFERRED,
    STATUS_EXTRACTED,
    STATUS_FILTERED,
    STATUS_PENDING,
    STATUS_REJECTED,
    claim_concurrency,
    ensemble_mode as default_ensemble_mode,
    proximity_threshold as default_threshold,
)
from veritas.gold_evidence.admissibility import (
    essential,
    lost_essential,
    missing_essential,
    restrict_to_condition,
)
from veritas.gold_evidence.extraction import extract_evidence
from veritas.gold_evidence.filtering import filter_evidence, get_reference_times
from veritas.gold_evidence.models import Evidence, VerdictRationale
from veritas.gold_evidence.sufficiency import SufficiencyResult, validate_sufficiency
from veritas.models import QuotaExceededError, RateLimitError
from veritas.util import run_with_semaphore

logger = logging.getLogger("VeriTaS")

# Rejection reasons recorded in `claims.gold_evidence_reason`
REJECT_NO_GOLD_VERDICT = "no_gold_verdict"
REJECT_NO_CLAIM_TIME = "no_claim_time"
REJECT_NO_FACT_CHECK_TIME = "no_fact_check_time"
#: Stage 1 produced neither evidence nor a rationale - nothing to analyze at all.
REJECT_NOTHING_EXTRACTED = "nothing_extracted"
#: An essential evidence item lost every one of its sources in Stage 2.
REJECT_ESSENTIAL_EVIDENCE_LOST = "essential_evidence_lost"
REJECT_ENSEMBLE_FAILED = "sufficiency_validation_failed"
REJECT_INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass
class ClaimOutcome:
    """What the reconstruction produced for one claim."""

    claim_id: int
    status: str
    reason: str | None = None
    n_candidates: int = 0
    n_admissible: int = 0
    n_deferred: int = 0
    n_rationales: int = 0
    #: Evidence items the verdict rationale depends on, and how many of them lost
    #: every source in Stage 2. The latter is what rejects an instance.
    n_essential: int = 0
    n_essential_lost: int = 0
    results: dict[str, SufficiencyResult] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.status == STATUS_ACCEPTED

    @property
    def deferred(self) -> bool:
        """The claim was neither accepted nor rejected: part of its evidence is
        waiting for a rate-limited source and the run must revisit it."""
        return self.status == STATUS_DEFERRED

    def recoverable(self, condition: str) -> bool | None:
        result = self.results.get(condition)
        return result.is_close if result else None


async def reconstruct_claim(
        claim: Claim,
        *,
        mode: str = None,
        threshold: float = None,
        re_extract: bool = False,
        re_filter: bool = False,
) -> ClaimOutcome:
    """Runs stages 1-3 for a single claim. Resumable: work already stored in the
    DB is reused unless `re_extract`/`re_filter` force it to be redone."""
    if mode is None:
        mode = default_ensemble_mode
    if threshold is None:
        threshold = default_threshold

    outcome = ClaimOutcome(claim_id=claim.id, status=STATUS_PENDING)

    gold: Verdict | None = await claim.current_verdict
    if gold is None:
        return await _finish(claim, outcome, STATUS_REJECTED, REJECT_NO_GOLD_VERDICT)

    # Both reference times must be known before anything is spent on this claim:
    # without t_c the two conditions collapse into one, and without t_f there is
    # no evidence cutoff at all, so neither instance could be analyzed.
    t_c, t_f = await get_reference_times(claim)
    if t_c is None:
        return await _finish(claim, outcome, STATUS_REJECTED, REJECT_NO_CLAIM_TIME)
    if t_f is None:
        return await _finish(claim, outcome, STATUS_REJECTED, REJECT_NO_FACT_CHECK_TIME)

    # --- Stage 1 -----------------------------------------------------------
    # Read through `db` rather than `claim.evidence`, so that the whole stage
    # orchestration talks to one injectable database handle.
    evidence: list[Evidence] = await db.get_evidence_for_claim(claim.id)
    rationales: list[VerdictRationale] = await db.get_verdict_rationales_for_claim(claim.id)
    if re_extract or not (evidence or rationales):
        extraction = await extract_evidence(claim, replace=re_extract)
        evidence, rationales = extraction.evidence, extraction.rationales
    outcome.n_candidates = len(evidence)
    outcome.n_rationales = len(rationales)

    # An empty evidence set is legitimate as long as a rationale carries the
    # argument; only the total absence of both leaves nothing to analyze.
    if not evidence and not rationales:
        return await _finish(claim, outcome, STATUS_REJECTED, REJECT_NOTHING_EXTRACTED)
    await db.set_gold_evidence_status(claim.id, STATUS_EXTRACTED)

    # --- Stage 2 -----------------------------------------------------------
    # Items waiting out a rate limit are left alone until their window expires.
    pending = [e for e in evidence
               if (re_filter or not e.filtered) and not e.deferred]
    if re_filter:
        # Redoing Stage 2 includes the item-level judgement: the dates it was
        # gated on are about to be re-established.
        for item in pending:
            item.later_event = None
    if pending:
        await filter_evidence(claim, pending)

    outcome.n_deferred = sum(1 for e in evidence if e.deferred)
    if outcome.n_deferred:
        # The evidence set is incomplete, so neither admissibility nor sufficiency
        # can be decided yet. Rejecting here would blame the claim for a throttled
        # source, and that rejection would be recorded permanently.
        logger.debug(f"Claim {claim.id} deferred: {outcome.n_deferred} evidence "
                     f"item(s) wait for a rate-limited source.")
        return await _finish(claim, outcome, STATUS_DEFERRED, None)

    outcome.n_admissible = sum(1 for item in evidence if item.admissible)

    # The argument is broken when a proposition the rationale rests on has no
    # admissible source left. Redundancy saves it: an item survives as long as one
    # of its sources does.
    outcome.n_essential = len(essential(evidence))
    lost = lost_essential(evidence)
    outcome.n_essential_lost = len(lost)
    if lost:
        logger.debug(f"Claim {claim.id} rejected: {len(lost)} essential item(s) lost "
                     f"every source, e.g. {lost[0].proposition[:120]!r}")
        return await _finish(claim, outcome, STATUS_REJECTED, REJECT_ESSENTIAL_EVIDENCE_LOST)
    await db.set_gold_evidence_status(claim.id, STATUS_FILTERED)

    # --- Stage 3 -----------------------------------------------------------
    for condition in CONDITIONS:
        subset = restrict_to_condition(evidence, condition)
        result = await validate_sufficiency(
            claim, subset, gold, condition=condition, mode=mode, threshold=threshold,
            rationales=rationales,
            # A condition that cannot supply an essential proposition cannot support
            # the rationale either, so there is nothing for the ensemble to decide.
            missing_essential=missing_essential(evidence, condition),
        )
        outcome.results[condition] = result
        await db.save_gold_evidence_result(result.to_db_dict())

    decisive = outcome.results[CONDITION_FACT_CHECK]
    if decisive.is_close is None:
        return await _finish(claim, outcome, STATUS_REJECTED, REJECT_ENSEMBLE_FAILED)
    if not decisive.is_close:
        return await _finish(claim, outcome, STATUS_REJECTED, REJECT_INSUFFICIENT_EVIDENCE)
    return await _finish(claim, outcome, STATUS_ACCEPTED, None)


async def _finish(claim: Claim, outcome: ClaimOutcome,
                  status: str, reason: str | None) -> ClaimOutcome:
    outcome.status = status
    outcome.reason = reason
    await db.set_gold_evidence_status(claim.id, status, reason)
    if status == STATUS_REJECTED:
        logger.debug(f"Claim {claim.id} rejected: {reason}")
    return outcome


async def reconstruct_claims(claims: list[Claim], **kwargs) -> list[ClaimOutcome]:
    """Runs the reconstruction for many claims concurrently.

    Quota and rate-limit errors abort the whole run, as elsewhere in the codebase:
    continuing would silently produce results from a degraded ensemble."""
    logger.info(f"Reconstructing gold evidence for {len(claims)} claims...")

    async def _run(claim: Claim) -> ClaimOutcome | None:
        try:
            return await reconstruct_claim(claim, **kwargs)
        except (QuotaExceededError, RateLimitError):
            raise
        except Exception as e:
            logger.error(f"Reconstruction of claim {claim.id} failed: "
                         f"{type(e).__name__}: {e}", exc_info=True)
            return None

    outcomes = await run_with_semaphore([_run(c) for c in claims], limit=claim_concurrency,
                                        show_progress=True,
                                        progress_description="Reconstructing gold evidence")
    return [o for o in outcomes if o is not None]


async def claim_reference_times(claim: Claim):
    """Convenience re-export so scripts need only import from this module."""
    return await get_reference_times(claim)


def summarize(outcomes: list[ClaimOutcome]) -> dict:
    """Aggregate counters for a run, for console output."""
    from collections import Counter

    statuses = Counter(o.status for o in outcomes)
    reasons = Counter(o.reason for o in outcomes if o.reason)
    recoverable = {
        condition: Counter(o.recoverable(condition) for o in outcomes)
        for condition in CONDITIONS
    }
    return {
        "n_claims": len(outcomes),
        "statuses": dict(statuses),
        "rejection_reasons": dict(reasons),
        "n_candidates": sum(o.n_candidates for o in outcomes),
        "n_admissible": sum(o.n_admissible for o in outcomes),
        "n_deferred_evidence": sum(o.n_deferred for o in outcomes),
        "n_rationales": sum(o.n_rationales for o in outcomes),
        "n_essential": sum(o.n_essential for o in outcomes),
        "n_essential_lost": sum(o.n_essential_lost for o in outcomes),
        "recoverable": {
            condition: {str(k): v for k, v in counter.items()}
            for condition, counter in recoverable.items()
        },
        "conditions": {"claim": CONDITION_CLAIM, "fact_check": CONDITION_FACT_CHECK},
    }
