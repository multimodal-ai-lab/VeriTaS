"""Stage 2 - Evidence filtering (Spec §3).

Stage 2 decides three kinds of things, each at the level it is about:

- per **source**, i.e. once per URL, however many items and claims cite it:
  §3.1 accessibility - retrieve the source, determine `available_since` (t_e) -
  and whether the publisher registry lists it as a professional fact-check;
- per **citation**, i.e. once per (evidence item, source):
  §3.2 faithfulness - does the source still support *this* proposition? - and
  §3.3 cutoffs - is t_e before the citing claim's t_c and t_f? Computed, not predicted;
- per **item**: whether the proposition rests on a change of the world that
  happened only after t_c (§3.3 (3)), asked once its citations are dated, and only
  when the item became available after t_c at all.

An evidence item survives as long as one of its citations does.

Sources are global and write-once (see `models.Source`). Before a source is
retrieved, its stored state is re-read under a per-URL lock, so a source that
another claim - possibly running concurrently - already retrieved is adopted rather
than fetched again. Only `re_retrieve` fetches a source anew; the citations judged
against its previous state then count as stale and are judged again whenever the
claims owning them are processed next.

The faithfulness validator receives only the proposition and the source content.
The temporal validator receives the claim but never the gold verdict or the
fact-checker's reasoning, so neither check can be circular.

Quota and rate-limit errors are run-level conditions and are never recorded as a
judgement on the source that happened to hit them: they abort the run instead. A
source that only rate-limited us, or is still behind Archive.today's access check,
is deferred by `defer_hours`, and so is every item citing it (see
`retrieval.SourceRetrieval`).
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime

import json_repair

from veritas.common import Claim, Prompt, Review
from veritas.db import db
from veritas.gold_evidence import (
    CONDITION_FACT_CHECK,
    defer_hours,
    evidence_concurrency,
    filtering_model,
    max_media_per_citation,
    max_source_content_length,
    reasoning_effort_faithfulness,
    reasoning_effort_temporal,
    undated_policy,
)
from veritas.gold_evidence.admissibility import (
    apply_admissibility,
    compute_temporal_bounds,
    select,
)
from veritas.gold_evidence.cleaning import clean_source_content
from veritas.gold_evidence.llm import FATAL_ERRORS, resolve_model
from veritas.gold_evidence.models import (
    Citation,
    Evidence,
    Faithfulness,
    LaterEventCheck,
    Source,
    TemporalValidation,
    media_references,
    prepend_media,
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

#: One lock per source key, shared by every claim this process reconstructs, so
#: that a URL cited by several claims running concurrently is retrieved once.
_source_locks: dict[str, asyncio.Lock] = {}
#: Keys of the sources `re_retrieve` has already fetched anew in this process: a
#: second claim citing the same URL in the same run reuses that retrieval.
_retrieved_anew: set[str] = set()


def _lock(key: str) -> asyncio.Lock:
    lock = _source_locks.get(key)
    if lock is None:
        lock = _source_locks[key] = asyncio.Lock()
    return lock


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def filter_evidence(claim: Claim, evidence: list[Evidence], *,
                          re_retrieve: bool = False,
                          re_judge: bool = False) -> list[Evidence]:
    """Filters the given evidence items and persists the outcome. Returns the
    admissible items (i.e. `E_f`).

    First every distinct source is settled - once, however many of the items cite
    it - then every citation that is not yet decided against its source's current
    state is judged, and finally the item-level later-event check is asked.

    `re_retrieve` fetches the sources anew instead of reusing what is stored;
    `re_judge` judges every citation again, even those already decided."""
    if not evidence:
        return []

    t_c, t_f = await get_reference_times(claim)

    try:
        await run_with_semaphore(
            [settle_source(source, re_retrieve=re_retrieve)
             for source in sources_to_retrieve(evidence)],
            limit=evidence_concurrency)

        await run_with_semaphore(
            [judge_citation(citation, evidence=item, t_c=t_c, t_f=t_f)
             for item in evidence for citation in item.citations
             if (re_judge or re_retrieve or not citation.filtered)
             and not citation.deferred],
            limit=evidence_concurrency)

        # The later-event check needs the dates the retrievals just established, so
        # it runs once the citations are done - once per item, not once per source.
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


def sources_to_retrieve(evidence: list[Evidence]) -> list[Source]:
    """The distinct sources the items cite as publications. A source cited only as
    a tool or as offline evidence is not retrieved: nothing about it is used."""
    seen, sources = set(), []
    for item in evidence:
        for citation in item.citations:
            source = citation.source
            if source is None or citation.exempt or id(source) in seen:
                continue
            seen.add(id(source))
            sources.append(source)
    return sources


async def settle_source(source: Source, *, re_retrieve: bool = False) -> Source:
    """Establishes what is decided once per URL: whether the source is a
    professional fact-check, whether it can be retrieved, what it says and when it
    became available. Persists the outcome.

    Under the source's lock, the stored state is adopted first: another claim may
    have settled the source since this claim loaded it. A settled source is then
    left alone - sources are write-once - unless `re_retrieve` asks for a fresh
    retrieval, which happens at most once per process and URL."""
    async with _lock(source.key):
        stored = await db.get_source_by_locator(source.locator)
        if stored is not None:
            source.adopt(stored)

        if re_retrieve and source.key not in _retrieved_anew:
            source.reset_retrieval()
        elif source.decided or source.deferred:
            # Retrieved before sources were cleaned: clean the stored content now.
            # That needs no refetch, so it does not break write-once.
            if needs_cleaning(source):
                await clean_source(source)
                await db.update_source(source)
            return source

        persist = True
        try:
            # --- Blacklist comparison ---------------------------------------------
            if source.is_fact_check is None:
                # None again if the lookup failed, so a later run asks once more.
                source.is_fact_check = await _is_fact_checking_org(source.locator)
            if source.is_fact_check:
                # Recorded on the source itself, so the leak is visible in the
                # exports and `determine_inadmissibility` can decide it as a pure
                # function. Nothing else needs to be known about it.
                return source

            # --- §3.1 Accessibility -----------------------------------------------
            retrieval = await retrieve_source(source.locator)

            if retrieval.rate_limited or retrieval.gated:
                # A throttled or Archive.today-gated source says nothing about any
                # item. Leave it unjudged and retry it after the cooldown (or on
                # demand via `scripts/retry_deferred_archive_today.py`).
                source.defer(hours=defer_hours)
                source.retrieval_error = retrieval.error
                reason = "gated by Archive.today" if retrieval.gated else "rate limited"
                logger.debug(f"Deferring source {source.locator} for {defer_hours}h "
                             f"({reason}): {retrieval.error}")
                return source

            source.deferred_until = None
            source.accessed_at = datetime.now()
            source.accessible = retrieval.accessible
            source.raw_content = (str(retrieval.content)
                                  if retrieval.content is not None else None)
            source.available_since = to_naive(retrieval.available_since)
            source.dating_method = retrieval.dating_method
            source.retrieval_method = retrieval.method
            source.retrieval_error = None if retrieval.accessible else retrieval.error
            if re_retrieve:
                _retrieved_anew.add(source.key)

        except FATAL_ERRORS:
            # Nothing is recorded about the source - in particular a stored
            # retrieval that `re_retrieve` was about to replace is kept - so a
            # later run evaluates it.
            persist = False
            raise
        except Exception as e:
            logger.warning(f"Retrieving source {source.locator} failed: "
                           f"{type(e).__name__}: {e}")
            source.accessed_at = datetime.now()
            source.accessible = False
            source.retrieval_error = f"{type(e).__name__}: {e}"
        finally:
            if persist:
                await db.update_source(source)

        # After the retrieval is stored: a quota error while cleaning must not
        # discard a retrieval that already succeeded.
        if needs_cleaning(source):
            await clean_source(source)
            await db.update_source(source)

    return source


def needs_cleaning(source: Source) -> bool:
    """Whether the source has retrieved content that was never put through
    cleaning. Attempted once: a page cleaning cannot handle keeps its raw content."""
    return bool(source.accessible and source.raw_content and source.cleaned_at is None)


async def clean_source(source: Source) -> Source:
    """Trims the source's content to its main content (`cleaning`), in place. The
    judges then read the cleaned content, see `Source.main_content`; a failed or
    refused cleaning leaves `cleaned_content` empty, i.e. the raw content applies."""
    try:
        cleaned = await clean_source_content(source.raw_content)
        # A page short enough to need no cleaning comes back as it is; storing it a
        # second time would only duplicate the raw content.
        source.cleaned_content = cleaned if cleaned != source.raw_content else None
    except FATAL_ERRORS:
        raise
    except Exception as e:
        logger.debug(f"Cleaning source {source.locator} failed: {type(e).__name__}: {e}")
        source.cleaned_content = None
    source.cleaned_at = datetime.now()
    return source


async def judge_citation(citation: Citation, *,
                         evidence: Evidence,
                         t_c: datetime | None,
                         t_f: datetime | None) -> Citation:
    """Runs §3.2 and §3.3 on one citation, against the settled state of the source
    it cites, and sets its admissibility."""
    def decide() -> Citation:
        apply_admissibility(citation, undated_policy=undated_policy,
                            extraction_confidence=evidence.extraction_confidence)
        # The media the judge found go in front of the proposition - only from a
        # citation that stays: an unfaithful or inaccessible source's media do not
        # show what the proposition states.
        if citation.admissible and citation.media:
            evidence.proposition = prepend_media(evidence.proposition, citation.media)
        return citation

    citation.error = None
    citation.faithfulness = None
    citation.temporal_validation = None
    try:
        # --- Kinds that cannot be retrieved --------------------------------------
        # A tool has no publication to re-read and offline evidence has no locator,
        # so §3.1 and §3.2 do not apply. §3.3 does as soon as a `t_e` is known.
        if citation.exempt:
            if citation.available_since is not None:
                citation.temporal_validation = _temporal_validation(
                    citation.available_since, t_c=t_c, t_f=t_f)
            return decide()

        # A source that was never located, could not be retrieved, or leaks the
        # verdict is decided by `determine_inadmissibility` on its state alone.
        source = citation.source
        if source is None or source.is_fact_check or not source.accessible:
            return decide()

        # --- §3.2 Faithfulness ------------------------------------------------------
        # The cleaned content: without navigation, ads and the like, the budget
        # goes to the page itself, and so do the media the judge may attach.
        source_str = (source.main_content or "")[:max_source_content_length]
        citation.faithfulness = await assess_faithfulness(evidence.proposition, source_str)

        # --- §3.3 Cutoffs -----------------------------------------------------------
        citation.temporal_validation = _temporal_validation(
            citation.available_since, t_c=t_c, t_f=t_f)

    except FATAL_ERRORS:
        # Nothing is recorded about the citation, so a later run evaluates it.
        raise
    except Exception as e:
        logger.warning(f"Judging citation {citation.locator or citation.name} of evidence "
                       f"{evidence.id} failed: {type(e).__name__}: {e}")
        citation.error = f"{type(e).__name__}: {e}"

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
    fact-checker reasoning - to rule out circular validation.

    The same call names the source's media that show what the proposition states
    (`Faithfulness.media`): it is the one judgement that sees a proposition next to
    a source, media included. Only references that occur in `source_str` - the
    content the judge was shown - are kept, at most `max_media_per_citation`."""
    model = _resolve_filtering_model()
    available_media = media_references(source_str)
    prompt = Prompt(
        FAITHFULNESS_PROMPT_PATH,
        proposition=proposition,
        source=source_str,
        has_media=bool(available_media),
        max_media=max_media_per_citation,
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
    media = parse_cited_media(str(response), available=available_media,
                              limit=max_media_per_citation)
    return Faithfulness(assessment=score, reasoning=reasoning,
                        justification=justification, rater=model.specifier,
                        media=media)


#: One line of the judge's media answer: `Medium: <image:12>`. Plain lines rather
#: than a fenced or backticked block, so that it cannot be mistaken for the category
#: (backticked) or the explanation (the last fenced block). Anything after the
#: reference on the line is ignored.
_MEDIUM_LINE = re.compile(
    r"^[ \t*\-]*Medium:[ \t]*(<(?:image|video|audio):\d{1,9}>)",
    flags=re.IGNORECASE | re.MULTILINE)


def parse_cited_media(output: str, *, available: list[str],
                      limit: int | None = None) -> list[str]:
    """The media references the judge named, in its order: only those in
    `available` (the ones that occur in the content it was shown - anything else
    would be invented), each once, at most `limit`."""
    allowed = set(available)
    media: list[str] = []
    for reference in _MEDIUM_LINE.findall(output or ""):
        if reference in allowed and reference not in media:
            media.append(reference)
            if limit is not None and len(media) >= limit:
                break
    return media


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

def _temporal_validation(available_since: datetime | None, *,
                         t_c: datetime | None,
                         t_f: datetime | None) -> TemporalValidation:
    """Where a cited source sits relative to the citing claim's two cutoffs. Pure
    computation."""
    before_claim, before_fact_check = compute_temporal_bounds(available_since, t_c, t_f)
    return TemporalValidation(before_fact_check=before_fact_check, before_claim=before_claim)


def needs_later_event_check(evidence: Evidence, t_c: datetime | None) -> bool:
    """Whether §3.3 (3) has to be asked for this item.

    Only for evidence that became available *after* the claim: a proposition that
    could already be read at t_c cannot rest on anything that happened afterwards.
    Items that lost every citation are skipped as well - they are out regardless, and
    the call would be spent on a decision that no longer matters."""
    if evidence.later_event is not None:
        return False  # already judged
    if evidence.admissible is False:
        return False
    if evidence.deferred:
        # A cited source is still waiting out a rate limit, so `t_e` is not final yet.
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

    The cited sources are shown with their dates and an excerpt each, sharing the
    same content budget a single source used to get."""
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
    """The item's citations as the prompt renders them: who reported the
    proposition, when it became available, and an excerpt of what was retrieved.

    The excerpts share `max_source_content_length` between them, so an item with
    five citations costs the same context as one with a single citation."""
    citations = evidence.citations
    budget = max_source_content_length // max(len(citations), 1)
    briefs = []
    for citation in citations:
        source = citation.source
        content = ((source.main_content if source and not citation.exempt else None)
                   or "").strip()
        available_since = citation.available_since
        briefs.append({
            "name": citation.name,
            "kind": citation.kind.value,
            "locator": citation.locator,
            "available_since": (available_since.strftime("%B %d, %Y")
                                if available_since else None),
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
