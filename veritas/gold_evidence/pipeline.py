"""Orchestration of the three Gold Evidence Reconstruction stages for one claim.

The gold verdict is never modified. When sufficient valid evidence cannot be
reconstructed, the instance is *rejected*, which is recorded in the additive
`claims.gold_evidence_*` columns only - `claims.dismissed` stays untouched, so a
rejected instance remains a valid VeriTaS claim.

Neither an *empty* evidence set nor the loss of evidence in Stage 2 is a rejection
in itself. A verdict can rest on arithmetic, on a contradiction inside the claim, or
on what the claim's own media shows, and a key item that lost every source may have
been key only to the extractor. Whether what remains suffices to recover the gold
verdict is decided by one judge only - the Stage 3 ensemble. The roles are recorded
for the analysis, not used as a gate.
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
    key_evidence,
    lost_key,
    missing_key,
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
#: None of the claim's reviews has a readable fact-checking article, so there is
#: nothing to reconstruct evidence from. (An article that yields neither evidence
#: nor a rationale is *not* a rejection: the ensemble then judges the claim alone.)
REJECT_NO_ARTICLE = "no_fact_check_article"
REJECT_ENSEMBLE_FAILED = "sufficiency_validation_failed"
REJECT_INSUFFICIENT_EVIDENCE = "insufficient_evidence"

#: No longer assigned, but still stored on claims processed by earlier versions:
#: an empty extraction and a lost key item used to reject the instance outright.
LEGACY_REJECT_NOTHING_EXTRACTED = "nothing_extracted"
LEGACY_REJECT_KEY_EVIDENCE_LOST = "key_evidence_lost"
REJECT_KEY_EVIDENCE_LOST = LEGACY_REJECT_KEY_EVIDENCE_LOST  # kept for the legacy alias map

#: Not a rejection: the extractor failed on every article (an error or an unusable
#: response). The claim stays `pending` and is extracted again on the next run.
PENDING_EXTRACTION_FAILED = "extraction_failed"


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
    #: Key evidence items (those the verdict likely breaks without), and how many
    #: of them lost every source in Stage 2. Reported only; neither rejects.
    n_key: int = 0
    n_key_lost: int = 0
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
        re_retrieve: bool = False,
) -> ClaimOutcome:
    """Runs stages 1-3 for a single claim. Resumable: work already stored in the
    DB is reused unless `re_extract`/`re_filter` force it to be redone.

    `re_filter` judges every citation again against the stored sources. Sources are
    global and write-once, so it does not fetch them anew; `re_retrieve` does, and
    implies re-judging this claim's citations. Other claims citing a re-retrieved
    source notice that their citations became stale the next time they are
    processed."""
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
    if any(not item.citations for item in evidence):
        # Extraction never stores an item without a citation, so such an item comes
        # from before sources and citations had tables of their own: its sources
        # sit in the legacy `evidence_sources` table, which is no longer read.
        # Filtering it would reject the claim for "losing" sources it still has.
        logger.info(f"Claim {claim.id} has evidence stored in the legacy format; "
                    f"re-extracting it.")
        re_extract = True
    if re_extract or not (evidence or rationales):
        extraction = await extract_evidence(claim, replace=re_extract)
        if extraction.n_articles == 0:
            return await _finish(claim, outcome, STATUS_REJECTED, REJECT_NO_ARTICLE)
        if extraction.failed:
            # An error is not a finding: analyzing the empty result would judge the
            # claim as if its fact-check cited nothing. Try again next run.
            logger.warning(f"Claim {claim.id}: evidence extraction failed for all "
                           f"{extraction.n_articles} article(s); left pending.")
            return await _finish(claim, outcome, STATUS_PENDING, PENDING_EXTRACTION_FAILED)
        evidence, rationales = extraction.evidence, extraction.rationales
    outcome.n_candidates = len(evidence)
    outcome.n_rationales = len(rationales)

    # Neither an empty evidence set nor a missing rationale ends the analysis: the
    # ensemble then judges from whatever there is, down to the claim alone.
    await db.set_gold_evidence_status(claim.id, STATUS_EXTRACTED)

    # --- Stage 2 -----------------------------------------------------------
    # Items waiting out a rate limit are left alone until their window expires. An
    # item counts as unfiltered also when a citation went stale, i.e. its source
    # was re-retrieved (possibly for another claim) after the citation was judged.
    redo = re_filter or re_retrieve
    pending = [e for e in evidence
               if (redo or not e.filtered) and not e.deferred]
    for item in pending:
        if redo or any(citation.stale for citation in item.citations):
            # Redoing Stage 2 includes the item-level judgement: the dates it was
            # gated on are about to be re-established.
            item.later_event = None
    if pending:
        await filter_evidence(claim, pending, re_retrieve=re_retrieve, re_judge=re_filter)

    outcome.n_deferred = sum(1 for e in evidence if e.deferred)
    if outcome.n_deferred:
        # The evidence set is incomplete, so neither admissibility nor sufficiency
        # can be decided yet. Rejecting here would blame the claim for a throttled
        # source, and that rejection would be recorded permanently.
        logger.debug(f"Claim {claim.id} deferred: {outcome.n_deferred} evidence "
                     f"item(s) cite a rate-limited source.")
        return await _finish(claim, outcome, STATUS_DEFERRED, None)

    outcome.n_admissible = sum(1 for item in evidence if item.admissible)

    # Lost key items are counted, not acted on: the extractor's "key" is a
    # prediction of what the verdict needs, and Stage 3 tests that directly.
    outcome.n_key = len(key_evidence(evidence))
    outcome.n_key_lost = len(lost_key(evidence))
    if outcome.n_key_lost:
        logger.debug(f"Claim {claim.id}: {outcome.n_key_lost} key item(s) lost every "
                     f"source; the ensemble decides whether the rest suffices.")
    await db.set_gold_evidence_status(claim.id, STATUS_FILTERED)

    # --- Stage 3 -----------------------------------------------------------
    for condition in CONDITIONS:
        subset = restrict_to_condition(evidence, condition)
        result = await validate_sufficiency(
            claim, subset, gold, condition=condition, mode=mode, threshold=threshold,
            rationales=rationales,
            n_key_missing=len(missing_key(evidence, condition)),
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


async def retry_deferred_archive_today_sources(**kwargs) -> list[ClaimOutcome]:
    """Clears the deferral of every source still waiting behind Archive.today's
    access check and re-runs reconstruction for the claims citing them, instead of
    waiting out `defer_hours`. Sources are global, so each is cleared once however
    many claims cite it.

    Meant to be run once a human has passed the check (e.g. via scrapeMM's
    `configure_archive_today_session()`). `kwargs` are forwarded to
    `reconstruct_claims` (e.g. `mode`, `threshold`). See
    `veritas.pipeline.stage_4.retry_deferred_archive_today_appearances` for the
    main-pipeline counterpart, and `scripts/retry_deferred_archive_today.py`
    for the entry point that runs both."""
    sources = await db.get_deferred_archive_today_sources()
    if not sources:
        return []

    for source in sources:
        source.deferred_until = None
        await db.update_source(source)

    claim_ids = await db.get_claim_ids_citing_sources([source.id for source in sources])
    if not claim_ids:
        return []
    claims = await db.get_claims_by_ids(claim_ids)
    return await reconstruct_claims(claims, **kwargs)


async def retry_deferred_claims(**kwargs) -> list[ClaimOutcome]:
    """Reconstructs every claim on status `deferred` again, now.

    Called at the start of every reconstruction run, independent of the run's
    date range and limit, so that no claim stays parked just because later runs
    were configured for other claims (or for none at all). The deferral windows of
    the sources those claims wait for are cleared first: otherwise the retry would
    find every one of them still waiting and change nothing. A source that is still
    rate-limited or gated simply defers again, and its claims with it.

    `kwargs` are forwarded to `reconstruct_claims` (e.g. `mode`, `threshold`)."""
    claim_ids = await db.get_claim_ids_with_gold_evidence_status(STATUS_DEFERRED)
    if not claim_ids:
        return []

    sources = await db.get_deferred_sources_cited_by(claim_ids)
    for source in sources:
        source.deferred_until = None
        await db.update_source(source)

    logger.info(f"Retrying {len(claim_ids)} deferred claim(s), clearing the deferral "
                f"of {len(sources)} source(s).")
    claims = await db.get_claims_by_ids(claim_ids)
    return await reconstruct_claims(claims, **kwargs)


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
        "n_key": sum(o.n_key for o in outcomes),
        "n_key_lost": sum(o.n_key_lost for o in outcomes),
        "recoverable": {
            condition: {str(k): v for k, v in counter.items()}
            for condition, counter in recoverable.items()
        },
        "conditions": {"claim": CONDITION_CLAIM, "fact_check": CONDITION_FACT_CHECK},
    }
