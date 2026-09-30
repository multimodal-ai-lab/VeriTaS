"""Stage 1 - Candidate evidence extraction (Spec §2).

Reads the original (multimodal) fact-checking article from the DB and has an MLLM
identify every distinct evidence item the fact-checker used to establish the
verdict. No web access happens here; the article was already scraped by the main
pipeline's stage 3.

The same call also returns the *verdict rationale*: the reasoning that bridges the
claim and the evidence to the verdict. The extractor first assigns each item a role
(`key` means "removing it likely breaks the verdict") and only then writes the
rationale, which carries reasoning and commonsense knowledge only - no evidence,
no reference to specific evidence items, no externally available information.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse

import json_repair

from veritas.common import Article, Claim, Prompt, Review
from veritas.db import db
from veritas.gold_evidence import (
    extraction_model,
    max_article_length,
    max_evidence_per_claim,
    max_reviews_per_claim,
    reasoning_effort_extraction,
)
from veritas.gold_evidence.llm import FATAL_ERRORS, resolve_model
from veritas.gold_evidence.models import (
    Citation,
    Evidence,
    EvidenceRole,
    ProximityLevel,
    Source,
    SourceKind,
    VerdictRationale,
    to_naive,
)
from veritas.util import get_domain
from veritas.util.parsing import detect_hallucinated_media_refs, extract_last_code_block
from veritas.util.url import unshorten

logger = logging.getLogger("VeriTaS")

PROMPT_PATH = "veritas/gold_evidence/prompts/extract_evidence.md.j2"

@dataclass
class Extraction:
    """What Stage 1 recovered from a claim's fact-checking articles."""

    evidence: list[Evidence] = field(default_factory=list)
    #: One rationale per article that produced one.
    rationales: list[VerdictRationale] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """True if the articles yielded neither evidence nor a rationale, i.e. the
        instance cannot be analyzed at all. An empty *evidence set* alone is a
        legitimate outcome: some verdicts rest on the rationale only."""
        return not self.evidence and not self.rationales


async def extract_evidence(claim: Claim, replace: bool = False) -> Extraction:
    """Extracts and persists the candidate evidence and the verdict rationale of a
    claim, using up to `max_reviews_per_claim` of its fact-checking articles.

    `replace` drops the claim's previously stored evidence and rationales first.
    Without it, a re-extraction that phrases a proposition differently would leave
    the earlier item behind as an orphan that still enters the analysis."""
    if replace:
        deleted = await db.delete_evidence_for_claim(claim.id)
        deleted += await db.delete_verdict_rationales_for_claim(claim.id)
        if deleted:
            logger.debug(f"Dropped {deleted} previously stored Stage-1 record(s) "
                         f"of claim {claim.id} before re-extraction.")

    reviews = await _select_reviews(claim)
    if not reviews:
        logger.debug(f"Claim {claim.id} has no usable review for evidence extraction.")
        return Extraction()

    extraction = Extraction()
    for review in reviews:
        article = await review.article
        if not article or article.dismissed or not str(article.content).strip():
            continue
        evidence, rationale = await extract_from_article(claim, review, article)
        extraction.evidence.extend(evidence)
        if rationale:
            extraction.rationales.append(rationale)

    # TODO: May be removed
    extraction.evidence = deduplicate(extraction.evidence)[:max_evidence_per_claim]
    share_sources(extraction.evidence)

    for candidate in extraction.evidence:
        await candidate.save_to_db()
    for rationale in extraction.rationales:
        await rationale.save_to_db()
    return extraction


async def _select_reviews(claim: Claim) -> list[Review]:
    """The non-dismissed reviews of a claim, earliest first, capped by config."""
    reviews = [r for r in await claim.reviews if r and not r.dismissed]
    reviews.sort(key=lambda r: (to_naive(r.published) or datetime.max))
    return reviews[:max_reviews_per_claim]


async def extract_from_article(claim: Claim, review: Review,
                               article: Article) -> tuple[list[Evidence], VerdictRationale | None]:
    """Runs the extractor MLLM on a single fact-checking article.

    A *rectified* claim is not the claim the article checked: it is a corrected
    version of it, produced to balance the dataset, and the fact-checker never
    ruled on it. Only part of their investigation bears on it, and their argument
    does not - it runs against the original claim, whose verdict is the opposite
    of the rectified claim's. So the extractor is shown both claims and asked to
    assemble the case for the rectified one instead of reconstructing the
    fact-checker's own. For an original claim nothing about the prompt changes.
    """
    publisher = await review.publisher
    publisher_name = publisher.name if publisher else (review.raw_publisher_name or "the fact-checker")
    article_content = article.content
    article_str = str(article_content)[:max_article_length]
    published = to_naive(review.published)

    original_claim = await claim.variant if claim.is_rectified else None
    if claim.is_rectified and original_claim is None:
        logger.warning(f"Rectified claim {claim.id} has no original variant on record; "
                       f"extracting without showing the claim the article checked.")

    prompt = Prompt(
        PROMPT_PATH,
        article=article_str,
        claim=claim.data,
        claim_date=claim.date_str or None,
        is_rectified=claim.is_rectified,
        original_claim=original_claim.data if original_claim else None,
        fact_check_date=published.strftime("%B %d, %Y") if published else None,
        publisher_name=publisher_name,
        source_kinds=[kind.value for kind in SourceKind],
    )

    model = _resolve_model(prompt)
    try:
        response, reasoning = await model.generate(
            prompt, reasoning_effort=reasoning_effort_extraction, return_reasoning=True)
    except FATAL_ERRORS:
        raise
    except Exception as e:
        logger.warning(f"Evidence extraction failed for claim {claim.id}, review {review.id}: {e}")
        return [], None

    if response is None:
        return [], None

    rationale_text, records = parse_extraction_response(str(response))
    if not records:
        logger.debug(f"Extractor returned no evidence records for claim {claim.id}, "
                     f"review {review.id}. Response:\n{str(response)[:3000]}")
    excluded_domains = _excluded_domains(review, publisher)
    resolved_locators = await resolve_locators(records)

    evidence: list[Evidence] = []
    for record in records:
        item = build_evidence(
            record,
            claim=claim,
            review=review,
            article=article,
            article_str=article_str,
            excluded_domains=excluded_domains,
            resolved_locators=resolved_locators,
        )
        if item:
            evidence.append(item)

    rationale = build_rationale(rationale_text, claim=claim, review=review,
                                article=article, reasoning=reasoning)
    # Every record dropped by the guards is a strong hint of a systematic problem
    # (e.g. locators no longer matching the stored article text), so it is surfaced
    # at INFO; the per-record reasons are logged at DEBUG by `build_citation`.
    log = logger.info if records and not evidence else logger.debug
    log(f"Extracted {len(evidence)}/{len(records)} evidence candidates "
        f"{'and a rationale ' if rationale else ''}"
        f"for claim {claim.id} from review {review.id}.")
    return evidence, rationale


def _resolve_model(prompt: Prompt):
    """Videos need a model that can natively read them (as in stage 5).

    Resolved through `llm.resolve_model`, so a configured model is constructed
    once and reused across all claims."""
    from veritas.models import gemini_strong, gpt_strong

    default = gemini_strong if prompt.has_videos() else gpt_strong
    return resolve_model(extraction_model, default)


def _excluded_domains(review: Review, publisher) -> set[str]:
    """Domains that may not serve as evidence locators: the fact-check itself and
    everything else published by the same organization. This enforces the rule
    that evidence must point to the original source, not to the fact-check."""
    domains = set(_host_keys(str(review.url)))
    if publisher and publisher.domains:
        domains.update(d.lower() for d in publisher.domains if d)
    return {d for d in domains if d}


def _host_keys(url: str) -> set[str]:
    """Identifiers of a URL's host: the registered domain and the bare hostname.

    Both are kept because `get_domain` relies on the public suffix list and
    returns nothing for hosts it cannot classify, in which case the hostname is
    the only handle we have."""
    keys = set()
    if domain := get_domain(url):
        keys.add(domain.lower())
    try:
        hostname = urlparse(url).hostname
    except ValueError:
        hostname = None
    if hostname:
        keys.add(hostname.lower().removeprefix("www."))
    return keys


def parse_extraction_response(response: str) -> tuple[str | None, list[dict]]:
    """Splits the model response into `(verdict rationale, evidence records)`,
    tolerating minor syntax errors and a missing code fence.

    The documented shape is an object with `verdict_rationale` and `evidence`. A
    bare list is accepted as well and then carries no rationale, so a model that
    answers in the older format still produces usable evidence."""
    payload = extract_last_code_block(response) or response
    try:
        parsed = json_repair.loads(payload)
    except Exception:
        logger.debug("Could not parse evidence extraction response.")
        return None, []

    rationale = None
    if isinstance(parsed, dict):
        rationale = str(parsed.get("verdict_rationale") or "").strip() or None
        records = parsed.get("evidence")
        if not isinstance(records, list):
            # Some models wrap the list under a different key, or return a single
            # record without a list around it.
            records = next((value for value in parsed.values() if isinstance(value, list)),
                           None)
            if records is None:
                records = [] if rationale else [parsed]
        parsed = records

    if not isinstance(parsed, list):
        return rationale, []
    return rationale, [record for record in parsed if isinstance(record, dict)]


def build_rationale(text: str | None, *, claim: Claim, review: Review,
                    article: Article, reasoning: str | None = None) -> VerdictRationale | None:
    """Turns the extracted rationale into a `VerdictRationale`, or None if the model
    stated none. References to media that are no longer in the store are dropped
    along with the rationale: an unresolvable reference would break every prompt the
    rationale is later rendered into."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        detect_hallucinated_media_refs(text)
    except (ValueError, AssertionError) as e:
        logger.debug(f"Dropping verdict rationale with invalid media reference: {e}")
        return None
    return VerdictRationale(
        claim_id=claim.id,
        review_id=review.id,
        article_id=article.id,
        rationale=text,
        extraction_reasoning=reasoning,
    )


async def resolve_locators(records: list[dict]) -> dict[str, str]:
    """Maps every extracted locator to its unshortened form.

    Resolving the whole article's locators in one gather overlaps the requests
    instead of serializing them; `unshorten` returns non-shortened URLs without
    touching the network at all."""
    locators = list({str(source.get("locator") or "").strip()
                     for record in records for source in _source_records(record)})
    locators = [locator for locator in locators if locator]
    if not locators:
        return {}
    resolved = await asyncio.gather(*(unshorten(locator) for locator in locators))
    return dict(zip(locators, resolved))


def build_evidence(
        record: dict,
        *,
        claim: Claim,
        review: Review,
        article: Article,
        article_str: str,
        excluded_domains: set[str],
        resolved_locators: dict[str, str] | None = None,
) -> Evidence | None:
    """Validates one extracted record and turns it into an `Evidence` object.

    A record carries one proposition and every source that reports it. Each source
    becomes a citation and is validated individually: those that fail a guard are
    dropped, and the item survives as long as one of them is left. Returns None if
    the proposition itself is unusable or no citation survived."""
    proposition = str(record.get("proposition") or "").strip()
    if not proposition:
        return None

    # Guard 1: no hallucinated media references
    try:
        detect_hallucinated_media_refs(proposition)
    except (ValueError, AssertionError) as e:
        logger.debug(f"Dropping evidence with invalid media reference: {e}")
        return None

    citations = []
    for source_record in _source_records(record):
        citation = build_citation(source_record, review=review, article_str=article_str,
                                  excluded_domains=excluded_domains,
                                  resolved_locators=resolved_locators)
        if citation and not any(citation.key == other.key for other in citations):
            citations.append(citation)

    if not citations:
        logger.debug(f"Dropping evidence without a usable source: {proposition[:80]!r}")
        return None

    return Evidence(
        claim_id=claim.id,
        review_id=review.id,
        article_id=article.id,
        proposition=proposition,
        citations=citations,
        role=_parse_enum(record.get("role"), EvidenceRole, EvidenceRole.AUXILIARY),
        extraction_reasoning=str(record.get("reasoning") or "").strip() or None,
        extraction_confidence=_parse_confidence(record.get("confidence")),
    )


def build_citation(record: dict, *,
                   review: Review,
                   article_str: str,
                   excluded_domains: set[str],
                   resolved_locators: dict[str, str] | None = None) -> Citation | None:
    """Validates one source record and turns it into a citation. Returns None if it
    violates an extraction rule.

    A located source becomes a `Source` the citation refers to; `share_sources`
    later makes all citations of the same URL within the claim refer to one
    instance, and the DB to one row across claims. A source without a locator is
    kept as a citation without a `Source` rather than dropped: fact-checks do cite
    sources they never link, and recording those is how the analysis can report
    how often that happens. Only tools and offline evidence are *expected* to have
    no locator; for any other kind, `admissibility` settles the citation as
    inaccessible without Stage 2 ever attempting a retrieval."""
    locator = str(record.get("locator") or "").strip()
    kind = _parse_enum(record.get("kind"), SourceKind, SourceKind.OTHER)

    domain = None
    if locator:
        # Guard 2: the locator must literally occur in the article (same trick as
        # stage 4's appearance extraction) - this rules out invented URLs. Checked
        # on the locator as written, before it is resolved to its long form.
        if locator not in article_str:
            logger.debug(f"Dropping source with locator not found in article: {locator}")
            return None

        # Guard 3: the locator must not point back at the fact-checker, unless it is a tool
        locator = (resolved_locators or {}).get(locator) or locator
        host_keys = _host_keys(locator)
        if kind != SourceKind.TOOL and host_keys & excluded_domains:
            logger.debug(f"Dropping source pointing back at the fact-checker: {locator}")
            return None
        domain = get_domain(locator) or next(iter(host_keys), None)
        if locator.rstrip("/") == str(review.url).rstrip("/"):
            return None

    return Citation(
        source=Source(locator=locator) if locator else None,
        name=str(record.get("name") or domain or "unnamed source").strip(),
        kind=kind,
        proximity=_parse_enum(record.get("proximity"), ProximityLevel,
                              ProximityLevel.SECONDARY),
    )


def _source_records(record: dict) -> list[dict]:
    """The source records of an extracted item.

    Accepts the documented `sources` list and, as a fallback, the flat
    `source_name`/`source_kind`/... spelling of a single source, so that a model
    answering in the older format still produces usable evidence."""
    sources = record.get("sources")
    if isinstance(sources, list):
        return [source for source in sources if isinstance(source, dict)]
    if isinstance(sources, dict):
        return [sources]
    flat = {key.removeprefix("source_"): record.get(key)
            for key in ("source_name", "source_kind", "source_locator", "source_proximity")}
    return [flat] if any(value for value in flat.values()) else []


def share_sources(evidence: list[Evidence]) -> list[Evidence]:
    """Makes every citation of the same URL within the claim refer to one `Source`
    instance, so that Stage 2 retrieves and dates it once and every citing item sees
    the same outcome. Returns the items for convenience."""
    shared: dict[str, Source] = {}
    for item in evidence:
        for citation in item.citations:
            if citation.source is not None:
                citation.source = shared.setdefault(citation.source.key, citation.source)
    return evidence


def deduplicate(candidates: list[Evidence]) -> list[Evidence]:
    """Merges items that assert the same proposition, keeping the union of their
    citations and the higher extraction confidence. Sorted by role then confidence.

    This is where redundancy across the two fact-checking articles is collected:
    if both cite the same proposition to different sources, the result is one
    evidence item with two citations rather than two items."""
    best: dict[str, Evidence] = {}
    for candidate in candidates:
        key = " ".join(candidate.proposition.lower().split())
        existing = best.get(key)
        if existing is None:
            best[key] = candidate
            continue
        for citation in candidate.citations:
            if not any(citation.key == other.key for other in existing.citations):
                existing.citations.append(citation)
        if candidate.extraction_confidence > existing.extraction_confidence:
            existing.extraction_confidence = candidate.extraction_confidence
        # The stricter role wins: if one article's verdict breaks without the
        # proposition, losing it breaks that article's case.
        if candidate.role == EvidenceRole.KEY:
            existing.role = EvidenceRole.KEY

    role_rank = {EvidenceRole.KEY: 0, EvidenceRole.AUXILIARY: 1, EvidenceRole.BACKGROUND: 2}
    return sorted(best.values(),
                  key=lambda e: (role_rank.get(e.role, 3), -e.extraction_confidence))


def _parse_enum(value, enum_cls, default):
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
        for member in enum_cls:
            if member.value == normalized or member.name.lower() == normalized:
                return member
        try:
            # Lets an enum accept aliases of its own, e.g. legacy role names.
            return enum_cls(normalized)
        except ValueError:
            pass
    return default


def _parse_confidence(value) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.5
    if confidence > 1.0:  # Some models answer in percent
        confidence /= 100.0
    return min(max(confidence, 0.0), 1.0)
