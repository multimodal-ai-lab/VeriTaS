"""Stage 2 - Evidence filtering (Spec §3).

Each *source* is filtered independently - these criteria are about the source,
not about the proposition it reports:
  §3.1 accessibility  - retrieve the source, determine `available_since` (t_e)
  §3.2 faithfulness   - does the *current* source still support the proposition?
  §3.3 cutoffs        - is t_e before t_c, before t_f? Computed, not predicted.

One criterion is about the *item*: whether the proposition rests on a change of the
world that happened only after t_c (§3.3 (3)). It is asked once per item, after its
sources have been dated, and only when the item became available after t_c at all.

An evidence item survives as long as one of its sources does.

The faithfulness validator receives only the proposition and the source content.
The temporal validator receives the claim but never the gold verdict or the
fact-checker's reasoning, so neither check can be circular.

Quota and rate-limit errors are run-level conditions and are never recorded as a
judgement on the item that happened to hit them: they abort the run instead. A
source that only rate-limited us defers the item by `defer_hours`.
"""

from __future__ import annotations

import logging
from datetime import datetime

import json_repair

from veritas.common import Claim, Prompt, Review
from veritas.db import db
from veritas.gold_evidence import (
    CONDITION_FACT_CHECK,
    defer_hours,
    evidence_concurrency,
    filtering_model,
    max_source_content_length,
    reasoning_effort_faithfulness,
    reasoning_effort_temporal,
    undated_policy,
)
from veritas.gold_evidence.admissibility import (
    UNRETRIEVABLE_KINDS,
    apply_admissibility,
    compute_temporal_bounds,
    select,
)
from veritas.gold_evidence.llm import FATAL_ERRORS, resolve_model
from veritas.gold_evidence.models import (
    Evidence,
    EvidenceSource,
    Faithfulness,
    LaterEventCheck,
    SourceKind,
    TemporalValidation,
    to_naive,
)
from veritas.gold_evidence.retrieval import retrieve_source
from veritas.pipeline.stage_6 import CERTAINTY_OPTIONS, LABEL_REGEX
from veritas.util import run_with_semaphore
from veritas.util.parsing import extract_last, extract_last_code_block

logger = logging.getLogger("VeriTaS")

FAITHFULNESS_PROMPT_PATH = "veritas/gold_evidence/prompts/assess_faithfulness.md.j2"
LATER_EVENT_PROMPT_PATH = "veritas/gold_evidence/prompts/validate_temporally.md.j2"

#: Faithfulness categories. Mirrors the codebase's category+certainty convention,
#: yielding scores in {-1, -2/3, -1/3, 0, 1/3, 2/3, 1}.
POSITIVE_CATEGORY = "entails"
NEGATIVE_CATEGORY = "contradicts"
FAITHFULNESS_CATEGORIES = (POSITIVE_CATEGORY, NEGATIVE_CATEGORY, "unknown")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def filter_evidence(claim: Claim, evidence: list[Evidence]) -> list[Evidence]:
    """Filters every source of the provided evidence items independently and persists the
    outcome. Returns the admissible items (i.e. `E_f`)."""
    if not evidence:
        return []

    t_c, t_f = await get_reference_times(claim)

    tasks = [filter_source(source, evidence=item, claim=claim, t_c=t_c, t_f=t_f)
             for item in evidence for source in item.sources if not source.deferred]
    try:
        await run_with_semaphore(tasks, limit=evidence_concurrency)

        # The later-event check needs the dates the retrievals just established, so
        # it runs once the sources are done - once per item, not once per source.
        await run_with_semaphore(
            [check_later_event(item, claim=claim) for item in evidence
             if needs_later_event_check(item, t_c)],
            limit=evidence_concurrency)
    finally:
        # Persist whatever was decided before the failure; otherwise a single
        # failing source would discard the whole claim's filtering work.
        for item in evidence:
            await item.save_to_db()

    return select(evidence, CONDITION_FACT_CHECK)


async def filter_source(source: EvidenceSource, *,
                        evidence: Evidence,
                        claim: Claim,
                        t_c: datetime | None,
                        t_f: datetime | None) -> EvidenceSource:
    """Runs §3.1-§3.3 on a single source and sets its admissibility."""
    def decide() -> EvidenceSource:
        return apply_admissibility(source, undated_policy=undated_policy,
                                   extraction_confidence=evidence.extraction_confidence)

    try:
        # --- 0 Kinds that cannot be retrieved -----------------------------------
        # A tool has no publication to re-read and offline evidence has no locator,
        # so §3.1 and §3.2 do not apply. §3.3 does as soon as a `t_e` is known: a
        # dated source can be placed on the timeline like any other, and there is
        # no reason to exempt it from the cutoff and the later-event check.
        if source.kind in UNRETRIEVABLE_KINDS:
            if source.available_since is not None:
                source.temporal_validation = _temporal_validation(source, t_c=t_c, t_f=t_f)
            return decide()

        # --- 0 Sources the article never located ---------------------------------
        if not source.locator:
            source.accessible = False
            source.dismissed_reason = "The article cites this source without locating it."
            return decide()

        # --- 0 Blacklist comparison ---------------------------------------------
        if await _is_fact_checking_org(source.locator):
            # Recorded on the source itself, so the leak is visible in the exports
            # and `determine_inadmissibility` can decide it as a pure function.
            source.kind = SourceKind.FACT_CHECK
            source.dismissed_reason = "Source is an accredited fact-checking organization."
            return decide()

        # --- §3.1 Accessibility -------------------------------------------------
        retrieval = await retrieve_source(source.locator)

        if retrieval.rate_limited:
            # A throttled source says nothing about the item. Leave it entirely
            # unjudged and retry it after the cooldown.
            source.defer(hours=defer_hours)
            logger.debug(f"Deferring source {source.locator} of evidence {evidence.id} "
                         f"for {defer_hours}h: {retrieval.error}")
            return source

        source.accessed_at = datetime.now()
        source.accessible = retrieval.accessible
        if retrieval.content is not None:
            source.raw_content = str(retrieval.content)
        source.available_since = to_naive(retrieval.available_since)

        if not retrieval.accessible:
            source.dismissed_reason = retrieval.error
            return decide()

        source_str = str(retrieval.content)[:max_source_content_length]

        # --- §3.2 Faithfulness --------------------------------------------------
        source.faithfulness = await assess_faithfulness(evidence.proposition, source_str)

        # --- §3.3 Cutoffs --------------------------------------------------------
        source.temporal_validation = _temporal_validation(source, t_c=t_c, t_f=t_f)

    except FATAL_ERRORS:
        # Nothing is recorded about the source, so a later run evaluates it properly.
        raise
    except Exception as e:
        logger.warning(f"Filtering source {source.locator} of evidence {evidence.id} "
                       f"failed: {type(e).__name__}: {e}")
        source.dismissed_reason = f"{type(e).__name__}: {e}"
        if source.accessible is None:
            source.accessible = False

    return decide()


async def get_reference_times(claim: Claim) -> tuple[datetime | None, datetime | None]:
    """Returns `(t_c, t_f)` as naive datetimes.

    `t_c` is the claim's release time, `claims.date`.

    `t_f` is the **latest** `reviews.published` among the claim's non-dismissed
    reviews - the publication time the fact-checking organization itself states,
    as stored in the DB. The fact-checking period of a claim ends when the last
    professional fact-check of it appeared, so that is the upper bound of the
    interval under study.

    `reviews.modified` is deliberately *not* used as a fallback: it records when
    the publisher last edited the article, which can be years after publication
    and would push the cutoff arbitrarily far into the future. A review without a
    `published` time simply does not contribute to `t_f`; if no review has one,
    `t_f` is None and the claim cannot be placed on the timeline at all (the
    pipeline rejects such instances rather than analyzing them without a cutoff).
    """
    t_c = to_naive(claim.date)
    reviews: list[Review] = [r for r in await claim.reviews if r and not r.dismissed]
    times = [to_naive(r.published) for r in reviews]
    times = [t for t in times if t is not None]
    t_f = max(times) if times else None
    if t_f is not None and t_c is not None and t_f < t_c:
        # Defensive: a fact-check cannot precede the claim it checks. Clamping to
        # the claim time makes the studied interval empty rather than negative,
        # which biases towards "no gain from the fact-checking period" - the
        # conservative direction for the analysis.
        logger.debug(f"Claim {claim.id}: t_f ({t_f}) precedes t_c ({t_c}); clamping to t_c.")
        t_f = t_c
    return t_c, t_f


# ---------------------------------------------------------------------------
# §3.2 Faithfulness
# ---------------------------------------------------------------------------

async def assess_faithfulness(proposition: str, source_str: str) -> Faithfulness | None:
    """Determines whether the retrieved source still entails the proposition.

    Receives proposition + source content only - no claim, no gold verdict, no
    fact-checker reasoning - to rule out circular validation."""
    model = _resolve_filtering_model()
    prompt = Prompt(
        FAITHFULNESS_PROMPT_PATH,
        proposition=proposition,
        source=source_str,
    )
    try:
        response, reasoning = await model.generate(
            prompt, reasoning_effort=reasoning_effort_faithfulness, return_reasoning=True)
    except FATAL_ERRORS:
        raise
    except Exception as e:
        logger.debug(f"Faithfulness assessment failed: {e}")
        return None
    if response is None:
        return None

    score, justification = parse_faithfulness_response(str(response))
    if score is None:
        return None
    return Faithfulness(assessment=score, reasoning=reasoning,
                        justification=justification, rater=model.specifier)


def parse_faithfulness_response(output: str) -> tuple[float | None, str | None]:
    """Parses category + certainty + explanation into a score in [-1, 1].

    Reuses the codebase's rating convention: the category is backticked, the
    certainty underscored, and the explanation is a fenced code block."""
    category = extract_last(output, "`", allowed_symbols=LABEL_REGEX)
    certainty = extract_last(output, "_", allowed_symbols=LABEL_REGEX)
    explanation = extract_last_code_block(output)

    if not category:
        return None, explanation
    category = category.strip().lower()
    if category not in FAITHFULNESS_CATEGORIES:
        return None, explanation

    if category == "unknown":
        return 0.0, explanation

    if not certainty or certainty.strip().lower() not in CERTAINTY_OPTIONS:
        return None, explanation

    magnitude = CERTAINTY_OPTIONS[certainty.strip().lower()]
    sign = 1 if category == POSITIVE_CATEGORY else -1
    return sign * magnitude, explanation


# ---------------------------------------------------------------------------
# §3.3 Cutoffs and later events
# ---------------------------------------------------------------------------

def _temporal_validation(source: EvidenceSource, *,
                         t_c: datetime | None,
                         t_f: datetime | None) -> TemporalValidation:
    """Where the source sits relative to the two cutoffs. Pure computation."""
    before_claim, before_fact_check = compute_temporal_bounds(source.available_since, t_c, t_f)
    return TemporalValidation(before_fact_check=before_fact_check, before_claim=before_claim)


def needs_later_event_check(evidence: Evidence, t_c: datetime | None) -> bool:
    """Whether §3.3 (3) has to be asked for this item.

    Only for evidence that became available *after* the claim: a proposition that
    could already be read at t_c cannot rest on anything that happened afterwards.
    Items that lost every source are skipped as well - they are out regardless, and
    the call would be spent on a decision that no longer matters."""
    if evidence.later_event is not None:
        return False  # already judged
    if evidence.admissible is False:
        return False
    if evidence.deferred:
        # A source is still waiting out a rate limit, so `t_e` is not final yet.
        # The next run asks once the whole set is dated.
        return False
    t_e = to_naive(evidence.available_since)
    return t_e is not None and t_c is not None and t_e > to_naive(t_c)


async def check_later_event(evidence: Evidence, *,
                            claim: Claim) -> LaterEventCheck:
    """Asks whether the item rests on a world state that came about only after t_c.

    This is the one judgement that is about the *proposition* rather than about a
    single source: several sources reporting the same fact either all describe a
    pre-existing state or all describe a later change, so asking per source would
    only multiply the cost and the chances of an inconsistent answer.

    The sources are shown with their dates and an excerpt each, sharing the same
    content budget a single source used to get."""
    model = _resolve_filtering_model()
    prompt = Prompt(
        LATER_EVENT_PROMPT_PATH,
        claim=claim.data if claim else None,
        claim_date=claim.date_str if claim and claim.date_str else None,
        proposition=evidence.proposition,
        sources=_source_briefs(evidence),
    )

    check = LaterEventCheck()
    try:
        response, reasoning = await model.generate(
            prompt, reasoning_effort=reasoning_effort_temporal, return_reasoning=True)
    except FATAL_ERRORS:
        raise
    except Exception as e:
        logger.debug(f"Later-event check failed for evidence {evidence.id}: {e}")
        check.justification = f"Check failed: {e}"
        evidence.later_event = check
        return check

    if response is not None:
        check.reasoning = reasoning
        parsed = parse_later_event_response(str(response))
        if parsed is not None:
            check.change_detected = parsed["change_detected"]
            check.justification = parsed["justification"]
            check.rater = model.specifier

    evidence.later_event = check
    return check


def _source_briefs(evidence: Evidence) -> list[dict]:
    """The item's sources as the prompt renders them: who reported the proposition,
    when it became available, and an excerpt of what was retrieved.

    The excerpts share `max_source_content_length` between them, so an item with
    five sources costs the same context as one with a single source."""
    sources = evidence.sources
    budget = max_source_content_length // max(len(sources), 1)
    briefs = []
    for source in sources:
        content = (source.raw_content or "").strip()
        briefs.append({
            "name": source.name,
            "kind": source.kind.value,
            "locator": source.locator,
            "available_since": (source.available_since.strftime("%B %d, %Y")
                                if source.available_since else None),
            "excerpt": content[:budget],
            "truncated": len(content) > budget,
        })
    return briefs


def parse_later_event_response(output: str) -> dict | None:
    """Parses the later-event verdict. None if the model stated no usable judgement."""
    payload = extract_last_code_block(output) or output
    try:
        parsed = json_repair.loads(payload)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None

    change_detected = _parse_bool(parsed.get("change_detected"))
    if change_detected is None:
        return None

    return {
        "change_detected": change_detected,
        "justification": str(parsed.get("justification") or "").strip() or None,
    }


def _parse_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "yes", "y"):
            return True
        if normalized in ("false", "no", "n"):
            return False
    return None


async def _is_fact_checking_org(locator: str | None) -> bool | None:
    """Returns True iff the locator is a URL to the site of an accredited
    fact-checking organization, i.e., an IFCN or an EFCSN signatory."""
    try:
        publisher = await db.get_publisher_by_url(locator)
    except Exception:
        return None
    if not publisher:
        return False
    if publisher.ifcn_status and publisher.ifcn_status.value != "not_a_signatory":
        return True
    if publisher.efcsn_status and publisher.efcsn_status.value != "not_a_signatory":
        return True
    return False


def _resolve_filtering_model():
    """The model used for the faithfulness and temporal assessments.

    Resolved through `llm.resolve_model`, so the provider is constructed once and
    then reused - this runs twice per evidence item."""
    from veritas.models import gpt_strong

    return resolve_model(filtering_model, gpt_strong)
