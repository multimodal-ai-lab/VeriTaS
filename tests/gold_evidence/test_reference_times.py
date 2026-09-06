"""`t_c` and `t_f`, the two reference times the whole temporal analysis rests on.

`t_c` is `claims.date`; `t_f` is the latest `reviews.published` among the claim's
non-dismissed reviews. `reviews.modified` must never be used.
"""

from datetime import date, datetime, timezone

import pytest

from veritas.common import Claim, Review
from veritas.gold_evidence.filtering import get_reference_times


def make_review(review_id: int, published=None, modified=None,
                dismissed: bool = False) -> Review:
    return Review(
        id=review_id,
        url=f"https://factchecker.example/{review_id}",
        raw_claim="A claim",
        published=published,
        modified=modified,
        dismissed=dismissed,
    )


def make_claim(reviews: list[Review], claim_date=datetime(2024, 5, 1)) -> Claim:
    claim = Claim(id=1, data="A claim", date=claim_date,
                  appearance_ids=set(), review_ids={r.id for r in reviews})
    return claim


@pytest.fixture
def with_reviews(monkeypatch):
    """Serves the given reviews as the claim's reviews, without touching the DB."""

    def _install(reviews: list[Review]):
        async def fake_reviews(self):
            return reviews

        monkeypatch.setattr(Claim, "reviews", property(fake_reviews))
        return reviews

    return _install


@pytest.mark.asyncio
async def test_t_c_is_the_claim_date(with_reviews):
    with_reviews([make_review(1, published=datetime(2024, 5, 10))])
    t_c, _ = await get_reference_times(make_claim([make_review(1)]))
    assert t_c == datetime(2024, 5, 1)


@pytest.mark.asyncio
async def test_t_f_is_the_latest_published_review(with_reviews):
    reviews = with_reviews([
        make_review(1, published=datetime(2024, 5, 8)),
        make_review(2, published=datetime(2024, 5, 19)),
        make_review(3, published=datetime(2024, 5, 12)),
    ])
    _, t_f = await get_reference_times(make_claim(reviews))
    assert t_f == datetime(2024, 5, 19)


@pytest.mark.asyncio
async def test_modified_is_never_used_as_a_fallback(with_reviews):
    """`modified` is the last edit time and can be years after publication;
    using it would push the evidence cutoff arbitrarily far into the future."""
    reviews = with_reviews([
        make_review(1, published=datetime(2024, 5, 8), modified=datetime(2026, 1, 1)),
        make_review(2, published=None, modified=datetime(2027, 1, 1)),
    ])
    _, t_f = await get_reference_times(make_claim(reviews))
    assert t_f == datetime(2024, 5, 8)


@pytest.mark.asyncio
async def test_reviews_without_a_published_time_do_not_contribute(with_reviews):
    reviews = with_reviews([
        make_review(1, published=None),
        make_review(2, published=datetime(2024, 5, 9)),
    ])
    _, t_f = await get_reference_times(make_claim(reviews))
    assert t_f == datetime(2024, 5, 9)


@pytest.mark.asyncio
async def test_t_f_is_none_when_no_review_has_a_published_time(with_reviews):
    reviews = with_reviews([make_review(1, published=None, modified=datetime(2026, 1, 1))])
    _, t_f = await get_reference_times(make_claim(reviews))
    assert t_f is None


@pytest.mark.asyncio
async def test_dismissed_reviews_are_ignored(with_reviews):
    reviews = with_reviews([
        make_review(1, published=datetime(2024, 5, 8)),
        make_review(2, published=datetime(2025, 1, 1), dismissed=True),
    ])
    _, t_f = await get_reference_times(make_claim(reviews))
    assert t_f == datetime(2024, 5, 8)


@pytest.mark.asyncio
async def test_times_are_normalized_to_naive_utc(with_reviews):
    reviews = with_reviews([
        make_review(1, published=datetime(2024, 5, 8, 12, 0, tzinfo=timezone.utc)),
    ])
    claim = make_claim(reviews, claim_date=date(2024, 5, 1))
    t_c, t_f = await get_reference_times(claim)
    assert t_c == datetime(2024, 5, 1, 0, 0)
    assert t_f == datetime(2024, 5, 8, 12, 0)
    assert t_f.tzinfo is None


@pytest.mark.asyncio
async def test_t_f_preceding_t_c_is_clamped(with_reviews):
    """Bad metadata must yield an empty interval, never a negative one."""
    reviews = with_reviews([make_review(1, published=datetime(2024, 4, 1))])
    t_c, t_f = await get_reference_times(make_claim(reviews))
    assert t_c == t_f == datetime(2024, 5, 1)


@pytest.mark.asyncio
async def test_t_c_may_be_missing(with_reviews):
    """`get_reference_times` reports it; the pipeline rejects such instances."""
    reviews = with_reviews([make_review(1, published=datetime(2024, 5, 8))])
    t_c, t_f = await get_reference_times(make_claim(reviews, claim_date=None))
    assert t_c is None
    assert t_f == datetime(2024, 5, 8)
