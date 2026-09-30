"""Fixtures for the Gold Evidence tests.

These tests are pure: no database, no network, no LLM calls. The `db` fixture
below deliberately overrides the autouse fixture in `tests/conftest.py` so that
no connection pool is opened for them.
"""

from datetime import datetime

import pytest_asyncio

from veritas.common import Verdict
from veritas.common.annotation import Rating
from veritas.common.annotation.rating import RatingAggregated
from veritas.common.verdict import MediumVerdict
from veritas.gold_evidence.admissibility import apply_admissibility_to_item
from veritas.gold_evidence.models import (
    Citation,
    Evidence,
    EvidenceRole,
    Source,
    Faithfulness,
    LaterEventCheck,
    ProximityLevel,
    SourceKind,
    TemporalValidation,
    VerdictRationale,
)


@pytest_asyncio.fixture(scope="function", autouse=True)
async def db():
    """Overrides the DB fixture from the parent conftest: these tests need no DB."""
    yield None


def make_rating(score: float, rater: str = "test") -> RatingAggregated:
    """An aggregated rating with a single member, for verdict fixtures."""
    return RatingAggregated(
        individual_ratings=[Rating(score=score, rater=rater, explanation="because")],
        rater=rater,
    )


def make_gold_verdict(*, veracity: float | None = None,
                      context_coverage: float | None = None,
                      media: dict[str, tuple[float, float]] | None = None,
                      claim_id: int = 1) -> Verdict:
    """A gold verdict with the given property scores.
    `media` maps a medium reference to `(authenticity, contextualization)`."""
    media_verdicts = [
        MediumVerdict(reference=reference,
                      authenticity=make_rating(authenticity),
                      contextualization=make_rating(contextualization))
        for reference, (authenticity, contextualization) in (media or {}).items()
    ]
    return Verdict(
        claim_id=claim_id,
        review_ids={1},
        media_verdicts=media_verdicts,
        veracity=make_rating(veracity) if veracity is not None else None,
        context_coverage=make_rating(context_coverage) if context_coverage is not None else None,
    )


def make_source(
        *,
        locator: str = "https://example.org/record/1",
        accessible: bool | None = True,
        content: str | None = "The record.",
        available_since=datetime(2024, 4, 15),
        is_fact_check: bool | None = False,
        retrieved: bool = True,
) -> Source:
    """A (global) source, by default retrieved, accessible and dated."""
    source = Source(locator=locator, available_since=available_since)
    if retrieved:
        source.accessible = accessible
        source.raw_content = content if accessible else None
        source.accessed_at = datetime(2024, 6, 1)
        source.is_fact_check = is_fact_check
    return source


def make_citation(
        *,
        locator: str | None = "https://example.org/record/1",
        name: str = "Example",
        kind: SourceKind = SourceKind.NEWS_ARTICLE,
        proximity: ProximityLevel = ProximityLevel.SECONDARY,
        accessible: bool | None = True,
        faithfulness: float | None = 1.0,
        available_since=datetime(2024, 4, 15),
        before_claim: bool = True,
        before_fact_check: bool = True,
        filtered: bool = True,
        source: Source | None = None,
) -> Citation:
    """A citation, by default of a retrievable source that is faithful to the
    proposition and available before the claim. `filtered=False` leaves both the
    source and the citation as Stage 1 produced them. Pass `source` to cite an
    existing (shared) source instead of building one from `locator`."""
    if source is None and locator:
        source = make_source(locator=locator, accessible=accessible,
                             available_since=available_since, retrieved=filtered)
    citation = Citation(source=source, name=name, kind=kind, proximity=proximity)
    if citation.exempt:
        # Not a publication: a known date is the citation's own, not a page's.
        citation.date_as_cited = available_since
    if filtered:
        if faithfulness is not None:
            citation.faithfulness = Faithfulness(assessment=faithfulness, reasoning="r")
        citation.temporal_validation = TemporalValidation(
            before_fact_check=before_fact_check,
            before_claim=before_claim,
        )
    return citation


def make_evidence(
        *,
        claim_id: int = 1,
        review_id: int | None = 7,
        proposition: str = "The mayor signed the decree on 3 May.",
        role: EvidenceRole = EvidenceRole.KEY,
        confidence: float = 0.9,
        citations: list[Citation] | None = None,
        later_event: bool | None = None,
        decided: bool = True,
        **citation_kwargs,
) -> Evidence:
    """An evidence item, by default admissible and pre-claim.

    Without `citations` it gets a single citation built from the remaining keyword
    arguments, which keeps the common one-source case short. `decided=False`
    leaves the citations unjudged, i.e. as Stage 1 produced them."""
    if citations is None:
        citations = [make_citation(filtered=decided, **citation_kwargs)]
    evidence = Evidence(
        claim_id=claim_id,
        review_id=review_id,
        proposition=proposition,
        citations=citations,
        role=role,
        extraction_confidence=confidence,
        later_event=(None if later_event is None
                     else LaterEventCheck(change_detected=later_event)),
    )
    if decided:
        # The configured policy, so that an undated fixture behaves as in production.
        apply_admissibility_to_item(evidence)
    return evidence


def make_rationale(rationale: str = "3 May is before 5 May, so the order of the two "
                                    "events in the claim is reversed.",
                   *, claim_id: int = 1, review_id: int | None = 7) -> VerdictRationale:
    """A verdict rationale: reasoning only, no external facts and no verdict."""
    return VerdictRationale(claim_id=claim_id, review_id=review_id, article_id=3,
                            rationale=rationale)
