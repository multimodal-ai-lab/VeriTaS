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
from veritas.gold_evidence.models import (
    Evidence,
    EvidenceRole,
    EvidenceSource,
    Faithfulness,
    ProximityLevel,
    SourceKind,
    TemporalValidation,
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


def make_evidence(
        *,
        claim_id: int = 1,
        proposition: str = "The mayor signed the decree on 3 May.",
        locator: str | None = "https://example.org/record/1",
        kind: SourceKind = SourceKind.NEWS_ARTICLE,
        proximity: ProximityLevel = ProximityLevel.SECONDARY,
        role: EvidenceRole = EvidenceRole.ESSENTIAL,
        confidence: float = 0.9,
        accessible: bool | None = True,
        faithfulness: float | None = 1.0,
        available_since=datetime(2024, 4, 15),
        before_claim: bool = True,
        before_fact_check: bool = True,
        later_event: bool = False,
        filtered: bool = True,
) -> Evidence:
    """A fully specified evidence item, by default admissible and pre-claim."""
    evidence = Evidence(
        claim_id=claim_id,
        proposition=proposition,
        source=EvidenceSource(name="Example", kind=kind, locator=locator, proximity=proximity),
        role=role,
        extraction_confidence=confidence,
        available_since=available_since,
    )
    if filtered:
        evidence.accessible = accessible
        if faithfulness is not None:
            evidence.faithfulness = Faithfulness(assessment=faithfulness, reasoning="r")
        evidence.temporal_validation = TemporalValidation(
            before_fact_check=before_fact_check,
            before_claim=before_claim,
            later_event=later_event,
        )
    return evidence
