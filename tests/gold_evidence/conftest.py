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
    Evidence,
    EvidenceRole,
    EvidenceSource,
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
) -> EvidenceSource:
    """A source, by default retrievable, faithful and available before the claim."""
    source = EvidenceSource(name=name, kind=kind, locator=locator, proximity=proximity,
                            available_since=available_since)
    if filtered:
        source.accessible = accessible
        if faithfulness is not None:
            source.faithfulness = Faithfulness(assessment=faithfulness, reasoning="r")
        source.temporal_validation = TemporalValidation(
            before_fact_check=before_fact_check,
            before_claim=before_claim,
        )
    return source


def make_evidence(
        *,
        claim_id: int = 1,
        review_id: int | None = 7,
        proposition: str = "The mayor signed the decree on 3 May.",
        role: EvidenceRole = EvidenceRole.ESSENTIAL,
        confidence: float = 0.9,
        sources: list[EvidenceSource] | None = None,
        later_event: bool | None = None,
        decided: bool = True,
        **source_kwargs,
) -> Evidence:
    """An evidence item, by default admissible and pre-claim.

    Without `sources` it gets a single source built from the remaining keyword
    arguments, which keeps the common one-source case short. `decided=False`
    leaves the sources unfiltered, i.e. as Stage 1 produced them."""
    if sources is None:
        sources = [make_source(filtered=decided, **source_kwargs)]
    evidence = Evidence(
        claim_id=claim_id,
        review_id=review_id,
        proposition=proposition,
        sources=sources,
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
