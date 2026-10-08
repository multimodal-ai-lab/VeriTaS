"""DB integration tests for saving ClaimReviews (matching spelling variants of known
reviews), backfilling the match keys, the one-off backup of the legacy Google
ClaimReviews, and the date-ranged re-extraction of review data. They run on a
temporary database (see `tests/db/conftest.py`)."""

from datetime import date, datetime
import asyncio
from collections import namedtuple

import pytest

from veritas.pipeline import stage_1
from veritas.util.review_key import review_match_key
from veritas.util.util import hash_int32


async def insert_review(db, url: str, claim: str, **columns) -> int:
    """Inserts a review row directly (bypassing the pipeline), without match keys."""
    names = ["url", "raw_claim", "raw_claim_hash", *columns]
    placeholders = ", ".join(f"${i}" for i in range(1, len(names) + 1))
    query = f"INSERT INTO reviews ({', '.join(names)}) VALUES ({placeholders}) RETURNING id"
    return await db._fetchval(query, url, claim, hash_int32(claim), *columns.values())


async def get_row(db, review_id: int):
    return await db._fetchrow("SELECT * FROM reviews WHERE id = $1", review_id)


async def count_reviews(db) -> int:
    return await db._fetchval("SELECT COUNT(*) FROM reviews")


# --- Saving ClaimReviews ------------------------------------------------------------

@pytest.mark.asyncio
async def test_spelling_variants_update_instead_of_insert(db):
    url, claim = "https://www.politifact.com/factchecks/2018/a/", '"Tina (Smith) profited."'
    assert await db.save_claim_review({(url, claim): {"v": 1}}, source="google") == (1, 0)

    variant = ("http://politifact.com/factchecks/2018/a", "“Tina (Smith) profited.”")
    assert await db.save_claim_review({variant: {"v": 1}}, source="google") == (0, 0)  # Unchanged
    assert await db.save_claim_review({variant: {"v": 2}}, source="google") == (0, 1)  # Changed
    assert await count_reviews(db) == 1

    row = await db._fetchrow("SELECT * FROM reviews")
    assert row["google_claim_review"] == {"v": 2}
    assert (row["url"], row["raw_claim"]) == (url, claim)  # The original spelling is kept
    assert (row["norm_url"], row["norm_claim_hash"]) == review_match_key(url, claim)


@pytest.mark.asyncio
async def test_other_claim_of_same_article_is_inserted(db):
    url = "https://www.politifact.com/article/2024/jan/11/debate-fact-check/"
    await db.save_claim_review({(url, "Claim A"): {"v": 1}}, source="google")
    assert await db.save_claim_review({(url, "Claim B"): {"v": 1}}, source="google") == (1, 0)
    assert await count_reviews(db) == 2


@pytest.mark.asyncio
async def test_verdict_label_is_part_of_the_claim(db):
    url = "https://voxukraine.org/fejk-abc"
    await db.save_claim_review({(url, "ФЕЙК: Claim"): {"v": 1}}, source="google")
    assert await db.save_claim_review({(url, "Claim"): {"v": 1}}, source="google") == (1, 0)


@pytest.mark.asyncio
async def test_variants_within_one_batch_are_collapsed(db):
    crs = {
        ("https://www.example.com/a/", "Claim"): {"v": 1},
        ("http://example.com/a", "claim"): {"v": 2},
    }
    assert await db.save_claim_review(crs, source="datacommons") == (1, 0)


@pytest.mark.asyncio
async def test_other_source_fills_known_review(db):
    await db.save_claim_review({("https://example.com/a", "Claim"): {"dc": 1}}, source="datacommons")
    assert await db.save_claim_review({("https://www.example.com/a/", "Claim"): {"g": 1}}, source="google") == (0, 1)
    row = await db._fetchrow("SELECT * FROM reviews")
    assert (row["datacommons_claim_review"], row["google_claim_review"]) == ({"dc": 1}, {"g": 1})


@pytest.mark.asyncio
async def test_saving_leaves_extracted_data_untouched(db):
    review_id = await insert_review(
        db, "https://example.com/a", "Claim", google_claim_review={"v": 1}, stage=3,
        raw_claimant_name="Old claimant", raw_claim_date=datetime(2024, 1, 1), appearance_ids=[42],
    )
    await db.save_claim_review({("https://example.com/a", "Claim"): {"v": 2}}, source="google")

    row = await get_row(db, review_id)
    assert row["google_claim_review"] == {"v": 2}
    assert row["stage"] == 3
    assert row["raw_claimant_name"] == "Old claimant"
    assert row["raw_claim_date"] == datetime(2024, 1, 1)
    assert row["appearance_ids"] == [42]


@pytest.mark.asyncio
async def test_oldest_of_existing_duplicates_is_updated(db):
    first = await insert_review(db, "https://example.com/a", '"Claim"')
    second = await insert_review(db, "https://example.com/a", "“Claim”")
    await db.save_claim_review({("https://example.com/a", "Claim"): {"v": 1}}, source="google")
    assert (await get_row(db, first))["google_claim_review"] == {"v": 1}
    assert (await get_row(db, second))["google_claim_review"] is None


# --- Match key backfill -------------------------------------------------------------

@pytest.mark.asyncio
async def test_sync_review_keys_backfills_only_missing_keys(db):
    ids = [await insert_review(db, f"https://www.example.com/{i}/", f"Claim {i}") for i in range(5)]
    updated_at_before = await db._fetchval("SELECT MAX(updated_at) FROM reviews")

    assert await db.sync_review_keys(batch_size=2) == 5
    assert await db.sync_review_keys() == 0

    for i, review_id in enumerate(ids):
        row = await get_row(db, review_id)
        assert (row["norm_url"], row["norm_claim_hash"]) == review_match_key(row["url"], row["raw_claim"])
        assert row["norm_url"] == f"example.com/{i}"
    # Filling the keys is no content change
    assert await db._fetchval("SELECT MAX(updated_at) FROM reviews") == updated_at_before


# --- Legacy backup of the Google ClaimReviews ---------------------------------------

@pytest.mark.asyncio
async def test_legacy_google_claim_reviews_are_backed_up_once(db):
    # Simulate a DB from before the backup column existed
    await db._execute("ALTER TABLE reviews DROP COLUMN google_claim_review_legacy")
    with_google = await insert_review(db, "https://example.com/a", "A", google_claim_review={"old": True})
    without_google = await insert_review(db, "https://example.com/b", "B", datacommons_claim_review={"dc": 1})
    updated_at_before = await db._fetchval("SELECT MAX(updated_at) FROM reviews")

    await db._create_tables()  # Migrates

    assert (await get_row(db, with_google))["google_claim_review_legacy"] == {"old": True}
    assert (await get_row(db, without_google))["google_claim_review_legacy"] is None
    assert await db._fetchval("SELECT MAX(updated_at) FROM reviews") == updated_at_before

    # New Google ClaimReviews only go into the regular column; the backup stays frozen
    await db.save_claim_review({("https://example.com/a", "A"): {"new": True}}, source="google")
    await db._create_tables()
    row = await get_row(db, with_google)
    assert row["google_claim_review"] == {"new": True}
    assert row["google_claim_review_legacy"] == {"old": True}


# --- Re-extraction ------------------------------------------------------------------

def google_cr(published: str, claimant: str | None = None, claim_date: str | None = None,
              appearances: list[str] | None = None) -> dict:
    item_reviewed = {"@type": "Claim"}
    if claimant:
        item_reviewed["author"] = {"@type": "Person", "name": claimant}
    if claim_date:
        item_reviewed["datePublished"] = claim_date
    if appearances:
        item_reviewed["appearance"] = [{"url": url} for url in appearances]
    return {
        "@type": "ClaimReview",
        "datePublished": published,
        "author": {"@type": "Organization", "name": "VoxUkraine", "url": "voxukraine.org"},
        "reviewRating": {"@type": "Rating", "alternateName": "Фейк"},
        "itemReviewed": item_reviewed,
    }


@pytest.fixture
def fake_appearances(monkeypatch):
    """Avoids network access: each appearance URL maps to a fixed fake appearance."""
    ids = {"https://t.me/channel/1": 101, "https://t.me/channel/2": 102}
    fake_appearance = namedtuple("FakeAppearance", "id")  # Hashable, like `Appearance`

    async def fake_appearance_from_url(url):
        return fake_appearance(ids[url]) if url in ids else None

    monkeypatch.setattr(stage_1, "appearance_from_url", fake_appearance_from_url)


@pytest.mark.asyncio
async def test_reextraction_updates_only_reviews_in_range(db, fake_appearances):
    new_cr = google_cr("2026-08-15T00:00:00+00:00", claimant="Telegram channel",
                       claim_date="2026-08-10T00:00:00+00:00", appearances=["https://t.me/channel/1"])
    common = dict(google_claim_review=new_cr, published=datetime(2026, 8, 15), raw_rating="Фейк",
                  raw_publisher_name="VoxUkraine", raw_publisher_url="https://voxukraine.org")

    in_range = await insert_review(db, "https://voxukraine.org/1", "In range", stage=5, publisher_id=7,
                                   appearance_ids=[42], language="uk", claim_id=9, **common)
    out_of_range = await insert_review(db, "https://voxukraine.org/2", "Out of range", stage=5,
                                       **{**common, "published": datetime(2026, 6, 1)})
    dismissed = await insert_review(db, "https://voxukraine.org/3", "Dismissed", stage=5, dismissed=True,
                                    dismissed_reason="Some reason", **common)
    unprocessed = await insert_review(db, "https://voxukraine.org/4", "Unprocessed", stage=0, **common)
    before = {i: await get_row(db, i) for i in (out_of_range, dismissed, unprocessed)}

    n_updated = await stage_1.reextract_reviews(date(2026, 7, 1), date(2026, 9, 30), batch_size=1)
    assert n_updated == 1

    row = await get_row(db, in_range)
    assert row["raw_claimant_name"] == "Telegram channel"
    assert row["raw_claim_date"] == datetime(2026, 8, 10)
    assert set(row["appearance_ids"]) == {42, 101}  # United, nothing lost
    assert row["publisher_id"] == 7  # Kept, as identified by Stage 2
    # Not extracted from ClaimReviews, hence untouched
    assert (row["stage"], row["dismissed"], row["claim_id"], row["language"]) == (5, False, 9, "uk")

    for review_id, row_before in before.items():
        assert dict(await get_row(db, review_id)) == dict(row_before)


@pytest.mark.asyncio
async def test_reextraction_keeps_values_missing_in_new_claim_review(db, fake_appearances):
    review_id = await insert_review(
        db, "https://voxukraine.org/1", "Claim", stage=2, published=datetime(2026, 8, 15),
        google_claim_review=google_cr("2026-08-15T00:00:00+00:00"),  # No claimant, no claim date
        raw_claimant_name="Known claimant", raw_claim_date=datetime(2026, 8, 1),
    )
    await stage_1.reextract_reviews(date(2026, 8, 1), date(2026, 8, 31))

    row = await get_row(db, review_id)
    assert row["raw_claimant_name"] == "Known claimant"
    assert row["raw_claim_date"] == datetime(2026, 8, 1)


@pytest.mark.asyncio
async def test_reextraction_without_changes_writes_nothing(db, fake_appearances):
    cr = google_cr("2026-08-15T00:00:00+00:00", claimant="Someone")
    review_id = await insert_review(
        db, "https://voxukraine.org/1", "Claim", stage=2, published=datetime(2026, 8, 15),
        google_claim_review=cr, raw_claimant_name="Someone", raw_rating="Фейк",
        raw_publisher_name="VoxUkraine", raw_publisher_url="https://voxukraine.org",
    )
    before = await get_row(db, review_id)
    assert await stage_1.reextract_reviews(date(2026, 8, 1), date(2026, 8, 31)) == 0
    assert dict(await get_row(db, review_id)) == dict(before)


@pytest.mark.asyncio
async def test_concurrent_initialization_backs_up_once(db):
    """All stage processes initialize the DB at startup, concurrently."""
    await db._execute("ALTER TABLE reviews DROP COLUMN google_claim_review_legacy")
    review_id = await insert_review(db, "https://example.com/a", "A", google_claim_review={"old": True})

    await asyncio.gather(*(db._create_tables() for _ in range(4)))

    assert (await get_row(db, review_id))["google_claim_review_legacy"] == {"old": True}
