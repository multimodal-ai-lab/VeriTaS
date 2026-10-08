"""Pure tests for the re-extraction of review data in Stage 1: how fresh data
is merged into an already processed review without losing anything."""

from datetime import date, datetime

import pytest

from veritas.common import Review
from veritas.db.veritas_db import REVIEW_EXTRACTED_COLUMNS
from veritas.pipeline import stage_1
from veritas.pipeline.stage_1 import merge_extracted_review_data, to_date


def make_review(**columns) -> Review:
    return Review(url="https://factchecker.example/article", raw_claim="Some claim", **columns)


def extracted(**values) -> dict:
    """Extracted data with every column empty unless given."""
    return {column: values.get(column) for column in REVIEW_EXTRACTED_COLUMNS}


def test_new_values_fill_and_replace():
    review = make_review(raw_claimant_name="Old", raw_claim_date=None)
    changed = merge_extracted_review_data(
        review, extracted(raw_claimant_name="New", raw_claim_date=datetime(2026, 8, 10)))
    assert set(changed) == {"raw_claimant_name", "raw_claim_date"}
    assert review.raw_claimant_name == "New"
    assert review.raw_claim_date == datetime(2026, 8, 10)


def test_missing_values_do_not_erase_existing_ones():
    review = make_review(raw_claimant_name="Known", raw_rating="False", published=datetime(2026, 8, 1))
    assert merge_extracted_review_data(review, extracted()) == []
    assert (review.raw_claimant_name, review.raw_rating, review.published) == ("Known", "False",
                                                                               datetime(2026, 8, 1))


def test_appearances_are_united():
    review = make_review(appearance_ids={1, 2})
    assert merge_extracted_review_data(review, extracted(appearance_ids={2, 3})) == ["appearance_ids"]
    assert review.appearance_ids == {1, 2, 3}
    assert merge_extracted_review_data(review, extracted(appearance_ids={1})) == []
    assert merge_extracted_review_data(make_review(appearance_ids=None), extracted(appearance_ids=set())) == []


def test_known_publisher_is_kept():
    review = make_review(publisher_id=7)
    assert merge_extracted_review_data(review, extracted(publisher_id=8)) == []
    assert review.publisher_id == 7

    review = make_review(publisher_id=None)
    assert merge_extracted_review_data(review, extracted(publisher_id=8)) == ["publisher_id"]
    assert review.publisher_id == 8


def test_url_and_its_string_count_as_same():
    review = make_review(raw_publisher_url="https://voxukraine.org")  # Parsed into "https://voxukraine.org/"
    assert merge_extracted_review_data(review, extracted(raw_publisher_url="https://voxukraine.org")) == []
    assert merge_extracted_review_data(review, extracted(raw_publisher_url="https://voxukraine.org/")) == []
    assert merge_extracted_review_data(review, extracted(raw_publisher_url="https://other.org")) == [
        "raw_publisher_url"]


def test_other_columns_are_never_touched():
    review = make_review(stage=5, dismissed=False, claim_id=9, language="uk")
    merge_extracted_review_data(review, extracted(raw_claimant_name="New"))
    assert (review.stage, review.dismissed, review.claim_id, review.language) == (5, False, 9, "uk")


@pytest.mark.asyncio
async def test_extraction_yields_exactly_the_extracted_columns(monkeypatch):
    async def no_publisher(domain):
        return None

    monkeypatch.setattr(stage_1.db, "get_publisher_by_url", no_publisher)
    data = await stage_1._extract_review_data(make_review())
    assert set(data) == set(REVIEW_EXTRACTED_COLUMNS)


@pytest.mark.parametrize("value, expected", [
    (date(2026, 7, 1), date(2026, 7, 1)),
    (datetime(2026, 7, 1, 12, 30), date(2026, 7, 1)),
    ("2026-07-01", date(2026, 7, 1)),
    (" 2026-07-01 ", date(2026, 7, 1)),
])
def test_to_date(value, expected):
    assert to_date(value) == expected


def test_to_date_rejects_garbage():
    with pytest.raises(ValueError):
        to_date("July")


# --- Claim date validation for provereno.media -----------------------------------

@pytest.mark.parametrize("publisher_url, claim_date, kept", [
    ("https://provereno.media", "2026-08-05", True),  # 10 days before the review
    ("https://provereno.media", "2026-07-16", True),  # Exactly 30 days before
    ("https://provereno.media", "2026-07-01", False),  # 45 days before (formerly kept: 15 - 1 < 30)
    ("https://provereno.media", "2026-08-20", False),  # After the review
    ("https://other.example", "2026-07-01", True),  # Only applies to provereno.media
])
@pytest.mark.asyncio
async def test_provereno_claim_date_validation(monkeypatch, publisher_url, claim_date, kept):
    async def no_publisher(domain):
        return None

    monkeypatch.setattr(stage_1.db, "get_publisher_by_url", no_publisher)
    review = make_review(datacommons_claim_review={
        "datePublished": "2026-08-15",
        "author": {"@type": "Organization", "name": "Publisher", "url": publisher_url},
        "itemReviewed": {"@type": "Claim", "datePublished": claim_date},
    })
    data = await stage_1._extract_review_data(review)
    assert (data["raw_claim_date"] is not None) == kept
