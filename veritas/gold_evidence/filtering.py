"""Stage 2 - Evidence filtering (Spec §3).

Each evidence item is filtered independently:
  §3.1 accessibility  - retrieve the source, determine `available_since` (t_e)
  §3.2 faithfulness   - does the *current* source still support the proposition?
  §3.3 temporal       - cutoff violation, verdict leakage, later-event contamination

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

import aiohttp
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
    Faithfulness,
    SourceKind,
    TemporalValidation,
    to_naive,
)
from veritas.gold_evidence.retrieval import new_session, retrieve_source
from veritas.pipeline.stage_6 import CERTAINTY_OPTIONS, LABEL_REGEX
from veritas.util import run_with_semaphore
from veritas.util.parsing import extract_last, extract_last_code_block

logger = logging.getLogger("VeriTaS")

FAITHFULNESS_PROMPT_PATH = "veritas/gold_evidence/prompts/assess_faithfulness.md.j2"
TEMPORAL_PROMPT_PATH = "veritas/gold_evidence/prompts/validate_temporally.md.j2"

#: Faithfulness categories. Mirrors the codebase's category+certainty convention,
#: yielding scores in {-1, -2/3, -1/3, 0, 1/3, 2/3, 1}.
POSITIVE_CATEGORY = "entails"
NEGATIVE_CATEGORY = "contradicts"
FAITHFULNESS_CATEGORIES = (POSITIVE_CATEGORY, NEGATIVE_CATEGORY, "unknown")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def filter_evidence(claim: Claim, evidence: list[Evidence]) -> list[Evidence]:
    """Filters every candidate independently and persists the outcome.
    Returns the admissible items (i.e. `E_factcheck`)."""
    if not evidence:
        return []

    t_c, t_f = await get_reference_times(claim)

    async with new_session() as session:
        tasks = [filter_single(item, claim=claim, t_c=t_c, t_f=t_f, session=session)
                 for item in evidence]
        try:
            await run_with_semaphore(tasks, limit=evidence_concurrency)
        finally:
            # Persist whatever was decided before the failure; otherwise a single
            # failing item would discard the whole claim's filtering work.
            for item in evidence:
                await item.save_to_db()

    return select(evidence, CONDITION_FACT_CHECK)


async def filter_single(evidence: Evidence, *,
                        claim: Claim,
                        t_c: datetime | None,
                        t_f: datetime | None,
                        session: aiohttp.ClientSession | None = None) -> Evidence:
    """Runs §3.1-§3.3 on a single item and sets its admissibility."""
    try:
        # --- 0 Kinds that cannot be retrieved -----------------------------------
        # A tool has no publication to re-read and offline evidence has no locator,
        # so §3.1 and §3.2 do not apply. §3.3 does as soon as a `t_e` is known: a
        # dated item can be placed on the timeline like any other evidence, and
        # there is no reason to exempt it from the cutoff and the later-event check.
        if evidence.source.kind in UNRETRIEVABLE_KINDS:
            if evidence.available_since is not None:
                evidence.temporal_validation = await validate_temporally(
                    evidence, claim=claim, t_c=t_c, t_f=t_f)
            return apply_admissibility(evidence, undated_policy=undated_policy)

        # --- 0 Blacklist comparison ---------------------------------------------
        if await _is_fact_checking_org(evidence.source.locator):
            # Recorded on the source itself, so the leak is visible in the exports
            # and `determine_inadmissibility` can decide it as a pure function.
            evidence.source.kind = SourceKind.FACT_CHECK
            evidence.dismissed_reason = "Source is an accredited fact-checking organization."
            return apply_admissibility(evidence, undated_policy=undated_policy)

        # --- §3.1 Accessibility -------------------------------------------------
        retrieval = await retrieve_source(evidence.source.locator, session=session)

        if retrieval.rate_limited:
            # A throttled source says nothing about the item. Leave it entirely
            # unjudged and retry it after the cooldown.
            await evidence.defer(hours=defer_hours)
            logger.debug(f"Deferring evidence {evidence.id} for {defer_hours}h: "
                         f"{retrieval.error}")
            return evidence

        evidence.accessed_at = datetime.now()
        evidence.accessible = retrieval.accessible
        if retrieval.content is not None:
            evidence.source.raw_content = str(retrieval.content)
        evidence.available_since = to_naive(retrieval.available_since)

        if not retrieval.accessible:
            evidence.dismissed_reason = retrieval.error
            return apply_admissibility(evidence, undated_policy=undated_policy)

        source_str = str(retrieval.content)[:max_source_content_length]

        # --- §3.2 Faithfulness --------------------------------------------------
        evidence.faithfulness = await assess_faithfulness(evidence, source_str)

        # --- §3.3 Temporal validation -------------------------------------------
        evidence.temporal_validation = await validate_temporally(
            evidence, claim=claim, source_str=source_str, t_c=t_c, t_f=t_f
        )

    except FATAL_ERRORS:
        # Nothing is recorded about the item, so a later run evaluates it properly.
        raise
    except Exception as e:
        logger.warning(f"Filtering evidence {evidence.id} of claim {claim.id if claim else None} "
                       f"failed: {type(e).__name__}: {e}")
        evidence.dismissed_reason = f"{type(e).__name__}: {e}"
        if evidence.accessible is None:
            evidence.accessible = False

    return apply_admissibility(evidence, undated_policy=undated_policy)


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

async def assess_faithfulness(evidence: Evidence, source_str: str) -> Faithfulness | None:
    """Determines whether the retrieved source still entails the proposition.

    Receives proposition + source content only - no claim, no gold verdict, no
    fact-checker reasoning - to rule out circular validation."""
    model = _resolve_filtering_model()
    prompt = Prompt(
        FAITHFULNESS_PROMPT_PATH,
        proposition=evidence.proposition,
        source=source_str,
    )
    try:
        response, reasoning = await model.generate(
            prompt, reasoning_effort=reasoning_effort_faithfulness, return_reasoning=True)
    except FATAL_ERRORS:
        raise
    except Exception as e:
        logger.debug(f"Faithfulness assessment failed for evidence {evidence.id}: {e}")
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
# §3.3 Temporal validation
# ---------------------------------------------------------------------------

async def validate_temporally(evidence: Evidence, *,
                              claim: Claim,
                              t_c: datetime | None,
                              t_f: datetime | None,
                              source_str: str = "") -> TemporalValidation:
    """Combines the cutoff comparison with the later-event judgement (§3.3).

    The two cutoff comparisons are computed, not predicted. The single LLM call is
    made only for items inside the studied interval `t_c < t_e <= t_f`, where a
    later event is the one contamination the dates alone cannot rule out.

    `source_str` is empty for sources that cannot be retrieved (a tool, an offline
    interview): the judgement then rests on the proposition and the source metadata,
    which is all that exists for such an item."""
    before_claim, before_fact_check = compute_temporal_bounds(evidence.available_since, t_c, t_f)

    validation = TemporalValidation(
        before_fact_check=before_fact_check,
        before_claim=before_claim,
    )

    # Evidence that already existed when the claim was made cannot report on
    # anything that happened afterwards.
    if before_claim:
        return validation

    # Also skip tool evidence here
    if evidence.source.kind == SourceKind.TOOL:
        return validation

    if not before_fact_check:  # t_e > t_f
        validation.justification = "Skipped: evidence postdates the fact-check."
        return validation

    model = _resolve_filtering_model()
    prompt = Prompt(
        TEMPORAL_PROMPT_PATH,
        claim=claim.data if claim else None,
        claim_date=claim.date_str if claim and claim.date_str else None,
        proposition=evidence.proposition,
        source=source_str,
        source_name=evidence.source.name,
        source_url=evidence.source.locator,
        available_since=(evidence.available_since.strftime("%B %d, %Y")
                         if evidence.available_since else None),
    )
    try:
        response, reasoning = await model.generate(
            prompt, reasoning_effort=reasoning_effort_temporal, return_reasoning=True)
    except FATAL_ERRORS:
        raise
    except Exception as e:
        logger.debug(f"Temporal validation failed for evidence {evidence.id}: {e}")
        validation.justification = f"Validation failed: {e}"
        return validation

    if response is None:
        return validation

    validation.reasoning = reasoning
    parsed = parse_temporal_response(str(response))
    if parsed is None:
        return validation

    validation.later_event = parsed["later_event"]
    validation.justification = parsed["justification"]
    validation.rater = model.specifier
    return validation


def parse_temporal_response(output: str) -> dict | None:
    """Parses the temporal verdict. None if the model stated no usable judgement."""
    payload = extract_last_code_block(output) or output
    try:
        parsed = json_repair.loads(payload)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None

    later_event = _parse_bool(parsed.get("later_event"))
    if later_event is None:
        return None

    return {
        "later_event": later_event,
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
