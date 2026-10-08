"""Pure tests for parsing raw Google Fact Check Explorer (GFCE) results. A
result is a claim with *all* its reviews; each review must become a
ClaimReview of its own (formerly, only the first one was kept). The claim's
data (claimant, claim date, appearances) goes into a schema.org `Claim`."""

import copy
import logging
from datetime import UTC, datetime

import pytest

from veritas.pipeline.util.google_factcheck_explorer import (
    GoogleFactCheckExplorerRetriever,
    _group_identity,
    _offset_step,
    _page_dates,
    _page_is_outside_cutoff,
    claim_reviews_from_raw,
)

TS_2024 = 1704067200  # 2024-01-01T00:00:00Z
TS_2025 = 1735689600  # 2025-01-01T00:00:00Z


def make_review(url: str, publisher: str = "PolitiFact", ts: int | None = TS_2024,
                rating: str = "False", rating_values: list | None = None) -> list:
    review = [[publisher, f"{publisher.lower()}.com"], url, ts, rating]
    if rating_values is not None:
        review += [None] * 5 + [rating_values]
    return review


def make_raw(reviews: list, claim: str = "Some claim", claimant: str | None = "Someone",
             claim_ts: int | None = None, appearances: list[str] | None = None,
             appearance_urls: list[str] | None = None, claimant_appearances: list[str] | None = None,
             free_text: str | None = None) -> list:
    """Builds a raw GFCE result with the layout observed in live data (see
    `_item_reviewed_from_raw_claim`): [1] claimant, [2] claim date, [3] reviews,
    [4] appearances, [10] free text, [12] appearance URLs."""
    claimant_data = [] if claimant is None else [claimant, "/g/11abc"]
    if claimant_appearances is not None:
        claimant_data = [claimant, None, claimant_appearances]
    claim_data = [
        claim,
        claimant_data,
        claim_ts,
        reviews,
        [[[None, "x.com"], url, 1] for url in appearances] if appearances else None,
    ]
    if free_text or appearance_urls:
        claim_data += [None] * 5 + [free_text, None, appearance_urls]
    return [claim_data]


def test_all_reviews_of_a_claim_are_kept():
    raw = make_raw([
        make_review("https://www.politifact.com/a/"),
        make_review("https://www.misbar.com/b", publisher="Misbar", rating="Misleading"),
        make_review("https://factcheck.afp.com/c", publisher="AFP"),
    ])
    claim_reviews, n_failed = claim_reviews_from_raw(raw)

    assert n_failed == 0
    assert [cr["url"] for cr in claim_reviews] == [
        "https://www.politifact.com/a/", "https://www.misbar.com/b", "https://factcheck.afp.com/c"
    ]
    assert all(cr["claimReviewed"] == "Some claim" for cr in claim_reviews)
    assert [cr["author"]["name"] for cr in claim_reviews] == ["PolitiFact", "Misbar", "AFP"]
    assert claim_reviews[1]["reviewRating"]["alternateName"] == "Misleading"


def test_review_fields_are_mapped():
    raw = make_raw(
        [make_review("https://www.politifact.com/a/", ts=TS_2025, rating_values=[1, 1, 5])],
        claim_ts=TS_2024,
        appearances=["https://x.com/post/1", "https://x.com/post/2"],
        appearance_urls=["https://x.com/post/2", "https://x.com/post/3"],
        free_text="In a Facebook post",
    )
    (cr,), _ = claim_reviews_from_raw(raw)

    assert cr["@type"] == "ClaimReview"
    assert cr["datePublished"] == "2025-01-01T00:00:00+00:00"
    assert cr["author"] == {"@type": "Organization", "name": "PolitiFact", "url": "politifact.com"}
    assert cr["reviewRating"] == {
        "@type": "Rating", "ratingValue": 1, "worstRating": 1, "bestRating": 5, "alternateName": "False",
    }
    assert cr["itemReviewed"] == {
        "@type": "Claim",
        "author": {"@type": "Person", "name": "Someone"},
        "datePublished": "2024-01-01T00:00:00+00:00",
        # Union of [4] and [12], deduplicated in order
        "appearance": [{"url": "https://x.com/post/1"}, {"url": "https://x.com/post/2"},
                       {"url": "https://x.com/post/3"}],
    }  # The free text at [10] is no claim URL and must not show up


def test_claimant_appearances_are_included():
    raw = make_raw([make_review("https://a.example/1")], claimant_appearances=["https://t.me/c/1"],
                   appearances=["https://t.me/c/1", "https://t.me/c/2"])
    (cr,), _ = claim_reviews_from_raw(raw)
    assert cr["itemReviewed"]["appearance"] == [{"url": "https://t.me/c/1"}, {"url": "https://t.me/c/2"}]


def test_missing_claim_data():
    (cr,), _ = claim_reviews_from_raw(make_raw([make_review("https://a.example/1", ts=None)], claimant=None))
    assert cr["datePublished"] is None
    assert cr["reviewRating"]["ratingValue"] == -1
    assert cr["itemReviewed"] == {"@type": "Claim"}


@pytest.mark.parametrize("claim_ts", [None, "2024-01-01", True, 10 ** 20])
def test_invalid_claim_date_is_left_out(claim_ts):
    (cr,), _ = claim_reviews_from_raw(make_raw([make_review("https://a.example/1")], claim_ts=claim_ts))
    assert "datePublished" not in cr["itemReviewed"]


def test_item_reviewed_is_not_shared_between_reviews():
    raw = make_raw([make_review("https://a.example/1"), make_review("https://a.example/2")],
                   appearances=["https://x.com/post/1"])
    (cr_1, cr_2), _ = claim_reviews_from_raw(raw)
    cr_1["itemReviewed"]["appearance"].append({"url": "https://x.com/other"})
    assert cr_2["itemReviewed"]["appearance"] == [{"url": "https://x.com/post/1"}]


def test_malformed_review_is_skipped_without_losing_the_others():
    raw = make_raw([
        make_review("https://a.example/1"),
        ["only-a-publisher"],  # Malformed
        None,  # Malformed
        make_review("https://a.example/2"),
    ])
    claim_reviews, n_failed = claim_reviews_from_raw(raw)
    assert [cr["url"] for cr in claim_reviews] == ["https://a.example/1", "https://a.example/2"]
    assert n_failed == 2


def test_claim_without_reviews():
    assert claim_reviews_from_raw(make_raw([])) == ([], 0)
    assert claim_reviews_from_raw(make_raw(None)) == ([], 0)


def test_process_raw_flattens_groups_and_filters_by_date():
    retriever = GoogleFactCheckExplorerRetriever()
    raws = [
        make_raw([make_review("https://a.example/1", ts=TS_2024),
                  make_review("https://a.example/2", ts=TS_2025)], claim="Claim A"),
        [["Malformed claim"]],  # Lacks the reviews entirely
        make_raw([make_review("https://b.example/1", ts=TS_2025)], claim="Claim B"),
    ]
    results = retriever._process_raw(raws, after=datetime(2024, 6, 1, tzinfo=UTC))
    assert [(cr["url"], cr["claimReviewed"]) for cr in results] == [
        ("https://a.example/2", "Claim A"),
        ("https://b.example/1", "Claim B"),
    ]


# --- Paging of the live `list:recent` listing -------------------------------

CUTOFF = datetime(2024, 6, 1, tzinfo=UTC)  # TS_2024 is older, TS_2025 newer


def group(i, ts: int | None = TS_2025) -> list:
    return make_raw([make_review(f"https://r.example/{i}", ts=ts)], claim=f"Claim {i}")


class FakeListing:
    """In-memory stand-in for `_get_recent`. `on_call(listing, call_index)` runs
    before each request and may mutate `items` to simulate a live listing."""

    def __init__(self, items: list, on_call=None):
        self.items = list(items)
        self.on_call = on_call
        self.offsets: list[int] = []
        self.num_results: list[int] = []

    def __call__(self, offset: int = 0, num_results: int = 1000, **kwargs) -> list:
        if self.on_call:
            self.on_call(self, len(self.offsets))
        self.offsets.append(offset)
        self.num_results.append(num_results)
        return copy.deepcopy(self.items[offset:offset + num_results])


def retrieve_with(listing: FakeListing, **kwargs) -> list[dict]:
    retriever = GoogleFactCheckExplorerRetriever()
    retriever._get_recent = listing
    return retriever._retrieve(**kwargs)


def urls(claim_reviews: list[dict]) -> list[str]:
    return [cr["url"] for cr in claim_reviews]


def skip_warnings(caplog) -> list:
    return [rec for rec in caplog.records
            if rec.name == "VeriTaS" and rec.levelno == logging.WARNING and "skipped" in rec.getMessage()]


def test_offset_step():
    assert _offset_step(1000, 1000, 100) == 900  # Full page: overlap
    assert _offset_step(1000, 1000, 0) == 1000
    assert _offset_step(5, 5, 4) == 1
    assert _offset_step(1, 1, 0) == 1
    assert _offset_step(50, 1000, 100) == 50  # Short last page: no overlap, no crawling
    assert _offset_step(0, 1000, 100) == 1  # Always progresses


def test_offsets_advance_with_overlap_and_stop_on_empty_page():
    listing = FakeListing([group(i) for i in range(5)])
    results = retrieve_with(listing, page_size=3, overlap=1)

    assert listing.offsets == [0, 2, 4, 5]  # Pages [0-2], [2-4], [4] (short), [] (empty)
    assert set(listing.num_results) == {3}
    assert urls(results) == [f"https://r.example/{i}" for i in range(5)]  # Deduplicated, in order


def test_after_none_pages_until_empty_even_if_everything_is_old():
    listing = FakeListing([group(i, ts=TS_2024) for i in range(4)])
    results = retrieve_with(listing, after=None, page_size=2, overlap=0)

    assert listing.offsets == [0, 2, 4]
    assert len(results) == 4


def test_default_page_size_and_overlap():
    listing = FakeListing([group(i) for i in range(1500)])
    results = retrieve_with(listing)

    assert listing.offsets == [0, 900, 1500]  # [0-999], [900-1499] (short), [] (empty)
    assert set(listing.num_results) == {1000}
    assert len(results) == 1500


def test_invalid_paging_parameters():
    with pytest.raises(ValueError):
        retrieve_with(FakeListing([]), page_size=10, overlap=10)
    with pytest.raises(ValueError):
        retrieve_with(FakeListing([]), page_size=10, overlap=-1)
    with pytest.raises(ValueError):
        retrieve_with(FakeListing([]), page_size=0, overlap=0)


def test_groups_repeated_by_prepended_items_are_deduplicated():
    def prepend(listing, call_index):
        if call_index == 1:  # Two new items get published after the first page
            listing.items[:0] = [group("new-1"), group("new-2")]

    listing = FakeListing([group(i) for i in range(10)], on_call=prepend)
    results = retrieve_with(listing, page_size=4, overlap=1)

    assert len(urls(results)) == len(set(urls(results)))  # No duplicates
    assert {f"https://r.example/{i}" for i in range(10)} <= set(urls(results))


def _drop_first_after_first_page(listing, call_index):
    if call_index == 1:  # An item vanishes (or moves) while paging: all later items shift up
        del listing.items[0]


def test_fixed_offsets_skip_an_item_on_a_shifted_listing():
    """Reference: without overlap, the shift makes item 4 slip past the page boundary."""
    listing = FakeListing([group(i) for i in range(10)], on_call=_drop_first_after_first_page)
    results = retrieve_with(listing, page_size=4, overlap=0)
    assert "https://r.example/4" not in urls(results)


def test_overlap_catches_item_shifted_across_page_boundary(caplog):
    caplog.set_level(logging.WARNING, logger="VeriTaS")
    listing = FakeListing([group(i) for i in range(10)], on_call=_drop_first_after_first_page)
    results = retrieve_with(listing, page_size=4, overlap=2)

    assert urls(results) == [f"https://r.example/{i}" for i in range(10)]
    assert not skip_warnings(caplog)


def test_warns_if_listing_shifted_by_more_than_overlap(caplog):
    def drop_three(listing, call_index):
        if call_index == 1:
            del listing.items[:3]

    caplog.set_level(logging.WARNING, logger="VeriTaS")
    listing = FakeListing([group(i) for i in range(10)], on_call=drop_three)
    retrieve_with(listing, page_size=4, overlap=2)

    assert len(skip_warnings(caplog)) == 1


def test_no_skip_warning_after_short_page(caplog):
    def prepend_many(listing, call_index):
        if call_index == 1:  # Next page (offset 5) then holds only new items
            listing.items[:0] = [group(f"new-{i}") for i in range(20)]

    caplog.set_level(logging.WARNING, logger="VeriTaS")
    listing = FakeListing([group(i) for i in range(5)], on_call=prepend_many)
    results = retrieve_with(listing, page_size=8, overlap=2)  # First page is short

    assert listing.offsets[:2] == [0, 5]
    assert not skip_warnings(caplog)  # A short page isn't expected to overlap with the next one
    assert len(urls(results)) == len(set(urls(results))) == 20  # Items 0-4 and new-5 to new-19


def test_stops_when_all_items_of_a_page_are_older_than_cutoff():
    items = [group(0), group(1), group(2, ts=TS_2024), group(3, ts=TS_2024),
             group(4, ts=TS_2024), group(5), group(6)]
    listing = FakeListing(items)
    results = retrieve_with(listing, after=CUTOFF, page_size=2, overlap=0)

    assert listing.offsets == [0, 2]  # Page [2, 3] is entirely old
    assert urls(results) == ["https://r.example/0", "https://r.example/1"]


def test_does_not_stop_when_only_the_last_item_is_old():
    # The listing is only roughly sorted by date: an old item may precede recent ones
    items = [group(0), group(1), group(2, ts=TS_2024), group(3), group(4, ts=TS_2024),
             group(5, ts=TS_2024), group(6, ts=TS_2024), group(7, ts=TS_2024)]
    listing = FakeListing(items)
    results = retrieve_with(listing, after=CUTOFF, page_size=3, overlap=1)

    assert listing.offsets == [0, 2, 4]  # [0-2] and [2-4] contain recent items, [4-6] doesn't
    assert urls(results) == ["https://r.example/0", "https://r.example/1", "https://r.example/3"]


def test_pages_without_dates_do_not_stop_paging():
    items = [group(i, ts=None) for i in range(4)] + [group(4, ts=TS_2024), group(5, ts=TS_2024)]
    listing = FakeListing(items)
    retrieve_with(listing, after=CUTOFF, page_size=2, overlap=0)

    assert listing.offsets == [0, 2, 4]  # Undated pages are passed, the old page stops


def test_undated_items_do_not_count_as_inside_cutoff():
    page = [group(0, ts=None), group(1, ts=TS_2024)]
    assert _page_is_outside_cutoff(page, CUTOFF)


def test_cutoff_considers_all_reviews_of_all_groups():
    mixed = make_raw([make_review("https://a.example/old", ts=TS_2024),
                      make_review("https://a.example/new", ts=TS_2025)])
    assert not _page_is_outside_cutoff([group(0, ts=TS_2024), mixed], CUTOFF)
    assert _page_is_outside_cutoff([group(0, ts=TS_2024), group(1, ts=TS_2024)], CUTOFF)
    assert not _page_is_outside_cutoff([group(0, ts=None)], CUTOFF)
    assert not _page_is_outside_cutoff([], CUTOFF)


def test_page_dates_skips_missing_and_malformed():
    page = [
        make_raw([make_review("https://a.example/1", ts=TS_2024), make_review("https://a.example/2", ts=None),
                  ["only-a-publisher"], None]),
        [["Malformed claim"]],
        None,
        make_raw(None),
        group(1, ts=TS_2025),
    ]
    assert _page_dates(page) == [datetime.fromtimestamp(TS_2024, tz=UTC),
                                 datetime.fromtimestamp(TS_2025, tz=UTC)]


def test_group_identity():
    assert _group_identity(group(1)) == _group_identity(copy.deepcopy(group(1)))
    assert _group_identity(group(1)) != _group_identity(group(2))
    # Same claim text, different reviews: different groups
    assert _group_identity(group(1)) != _group_identity(
        make_raw([make_review("https://other.example/1")], claim="Claim 1"))
    # Non-identifying data like the rating doesn't matter
    assert _group_identity(make_raw([make_review("https://a.example/1", rating="False")])) == \
        _group_identity(make_raw([make_review("https://a.example/1", rating="Misleading")]))
    # Malformed groups fall back to their full serialization
    assert _group_identity([["Malformed claim"]]) == _group_identity([["Malformed claim"]])
    assert _group_identity([["Malformed claim"]]) != _group_identity([["Other malformed claim"]])
    assert isinstance(_group_identity(None), str)


# --- Further review details -------------------------------------------------

def test_review_title_and_language_are_kept():
    review = [["Вокс Україна", "voxukraine.org"], "https://voxukraine.org/fejk", TS_2025, "Фейк",
              None, None, "uk", ["ua"], "ФЕЙК: У продажі в Steam скоро з’явиться гра про ТЦК", [1, 1, 5]]
    (cr,), _ = claim_reviews_from_raw(make_raw([review]))
    assert cr["name"] == "ФЕЙК: У продажі в Steam скоро з’явиться гра про ТЦК"
    assert cr["inLanguage"] == "uk"


def test_missing_title_and_language_are_left_out():
    (cr,), _ = claim_reviews_from_raw(make_raw([make_review("https://a.example/1")]))
    assert "name" not in cr
    assert "inLanguage" not in cr
