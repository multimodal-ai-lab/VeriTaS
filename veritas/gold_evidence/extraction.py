"""Stage 1 - Candidate evidence extraction (Spec §2).

Reads the original (multimodal) fact-checking article from the DB and has an MLLM
identify every distinct evidence item the fact-checker used to establish the
verdict. No web access happens here; the article was already scraped by the main
pipeline's stage 3.
"""

from __future__ import annotations

import logging
from datetime import datetime
from urllib.parse import urlparse

import json_repair

from veritas.common import Article, Claim, Prompt, Review
from veritas.gold_evidence import (
    extraction_model,
    max_article_length,
    max_evidence_per_claim,
    max_reviews_per_claim,
    reasoning_effort_extraction,
)
from veritas.gold_evidence.llm import resolve_model
from veritas.gold_evidence.models import (
    Evidence,
    EvidenceRole,
    EvidenceSource,
    ProximityLevel,
    SourceKind,
)
from veritas.util import get_domain
from veritas.util.parsing import detect_hallucinated_media_refs, extract_last_code_block
from veritas.util.url import unshorten

logger = logging.getLogger("VeriTaS")

PROMPT_PATH = "veritas/gold_evidence/prompts/extract_evidence.md.j2"


async def extract_evidence(claim: Claim) -> list[Evidence]:
    """Extracts and persists the candidate evidence for a claim, using up to
    `max_reviews_per_claim` of its fact-checking articles."""
    reviews = await _select_reviews(claim)
    if not reviews:
        logger.debug(f"Claim {claim.id} has no usable review for evidence extraction.")
        return []

    candidates: list[Evidence] = []
    for review in reviews:
        article = await review.article
        if not article or article.dismissed or not str(article.content).strip():
            continue
        candidates.extend(await extract_from_article(claim, review, article))

    candidates = deduplicate(candidates)[:max_evidence_per_claim]

    for candidate in candidates:
        await candidate.save_to_db()
    return candidates


async def _select_reviews(claim: Claim) -> list[Review]:
    """The non-dismissed reviews of a claim, earliest first, capped by config."""
    reviews = [r for r in await claim.reviews if r and not r.dismissed]
    reviews.sort(key=lambda r: (r.published is None, r.published or datetime.max))
    return reviews[:max_reviews_per_claim]


async def extract_from_article(claim: Claim, review: Review, article: Article) -> list[Evidence]:
    """Runs the extractor MLLM on a single fact-checking article."""
    publisher = await review.publisher
    publisher_name = publisher.name if publisher else (review.raw_publisher_name or "the fact-checker")
    article_content = article.content
    article_str = str(article_content)[:max_article_length]

    prompt = Prompt(
        PROMPT_PATH,
        article=article_str,
        claim=claim.data,
        claim_date=claim.date_str or None,
        fact_check_date=review.published.strftime("%B %d, %Y") if review.published else None,
        publisher_name=publisher_name,
        source_kinds=[kind.value for kind in SourceKind],
    )

    model = _resolve_model(prompt)
    try:
        response = await model.generate(prompt, reasoning_effort=reasoning_effort_extraction)
    except Exception as e:
        logger.warning(f"Evidence extraction failed for claim {claim.id}, review {review.id}: {e}")
        return []

    if response is None:
        return []

    records = parse_extraction_response(str(response))
    excluded_domains = _excluded_domains(review, publisher)

    evidence: list[Evidence] = []
    for record in records:
        item = build_evidence(
            record,
            claim=claim,
            review=review,
            article=article,
            article_str=article_str,
            excluded_domains=excluded_domains,
        )
        if item:
            evidence.append(item)
    logger.debug(f"Extracted {len(evidence)}/{len(records)} evidence candidates "
                 f"for claim {claim.id} from review {review.id}.")
    return evidence


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


def parse_extraction_response(response: str) -> list[dict]:
    """Extracts the JSON list from the model response, tolerating minor syntax
    errors and a missing code fence."""
    payload = extract_last_code_block(response) or response
    try:
        parsed = json_repair.loads(payload)
    except Exception:
        logger.debug("Could not parse evidence extraction response.")
        return []
    if isinstance(parsed, dict):
        # Some models wrap the list, e.g. {"evidence": [...]}
        for value in parsed.values():
            if isinstance(value, list):
                parsed = value
                break
        else:
            parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    return [record for record in parsed if isinstance(record, dict)]


def build_evidence(
        record: dict,
        *,
        claim: Claim,
        review: Review,
        article: Article,
        article_str: str,
        excluded_domains: set[str],
) -> Evidence | None:
    """Validates one extracted record and turns it into an `Evidence` object.
    Returns None if the record violates any of the extraction rules."""
    proposition = str(record.get("proposition") or "").strip()
    locator = str(record.get("source_locator") or "").strip()
    if not proposition or not locator:
        return None

    # Guard 1: no hallucinated media references
    try:
        detect_hallucinated_media_refs(proposition)
    except (ValueError, AssertionError) as e:
        logger.debug(f"Dropping evidence with invalid media reference: {e}")
        return None

    # Guard 2: the locator must literally occur in the article (same trick as
    # stage 4's appearance extraction) - this rules out invented URLs.
    if locator not in article_str:
        logger.debug(f"Dropping evidence with locator not found in article: {locator}")
        return None

    # Guard 3: the locator must not point back at the fact-checker, unless it is a tool
    kind = _parse_enum(record.get("source_kind"), SourceKind, SourceKind.OTHER)
    try:
        locator = unshorten(locator)
    except Exception:
        pass
    host_keys = _host_keys(locator)
    if kind != SourceKind.TOOL and host_keys & excluded_domains:
        logger.debug(f"Dropping evidence pointing back at the fact-checker: {locator}")
        return None
    domain = get_domain(locator) or next(iter(host_keys), None)
    if locator.rstrip("/") == str(review.url).rstrip("/"):
        return None

    source = EvidenceSource(
        name=str(record.get("source_name") or domain or "unknown").strip(),
        kind=kind,
        locator=locator,
        proximity=_parse_enum(record.get("source_proximity"), ProximityLevel,
                              ProximityLevel.SECONDARY),
    )

    return Evidence(
        claim_id=claim.id,
        review_id=review.id,
        article_id=article.id,
        proposition=proposition,
        source=source,
        role=_parse_enum(record.get("role"), EvidenceRole, EvidenceRole.AUXILIARY),
        extraction_reasoning=str(record.get("reasoning") or "").strip() or None,
        extraction_confidence=_parse_confidence(record.get("confidence")),
    )


def deduplicate(candidates: list[Evidence]) -> list[Evidence]:
    """Removes duplicates by (normalized locator, normalized proposition), keeping
    the item with the higher extraction confidence. Sorted by role then confidence."""
    best: dict[tuple[str, str], Evidence] = {}
    for candidate in candidates:
        key = (candidate.source.locator.rstrip("/").lower(),
               " ".join(candidate.proposition.lower().split()))
        existing = best.get(key)
        if existing is None or candidate.extraction_confidence > existing.extraction_confidence:
            best[key] = candidate
    role_rank = {EvidenceRole.ESSENTIAL: 0, EvidenceRole.AUXILIARY: 1, EvidenceRole.BACKGROUND: 2}
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
    return default


def _parse_confidence(value) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.5
    if confidence > 1.0:  # Some models answer in percent
        confidence /= 100.0
    return min(max(confidence, 0.0), 1.0)
