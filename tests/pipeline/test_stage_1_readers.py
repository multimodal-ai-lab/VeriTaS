"""Pure tests for how Stage 1 reads the reviewed claim (`itemReviewed`) of a
ClaimReview. Publishers type and shape it in many ways, and Google's converted
ClaimReviews used to be ignored entirely, so the readers must be tolerant."""

from datetime import datetime

import pytest

from veritas.pipeline.stage_1 import (
    _absolute_url,
    _get_appearance_urls_for_item,
    _get_author_name_url,
    _get_claim_date,
    _get_claimant,
    _get_item_reviewed,
    _get_publisher_name_url,
)
from veritas.pipeline.util.google_factcheck_explorer import claim_reviews_from_raw


# --- itemReviewed --------------------------------------------------------------------

@pytest.mark.parametrize("item_type", ["Claim", "CreativeWork", "Thing", ["Claim"], None])
def test_item_reviewed_of_any_type_is_accepted(item_type):
    item = {"author": {"name": "Someone"}}
    if item_type is not None:
        item["@type"] = item_type
    assert _get_item_reviewed({"itemReviewed": item}) is item


def test_item_reviewed_list_prefers_claim():
    claim = {"@type": "Claim", "author": {"name": "Someone"}}
    cr = {"itemReviewed": ["garbage", {"@type": "CreativeWork"}, claim]}
    assert _get_item_reviewed(cr) is claim


def test_item_reviewed_list_falls_back_to_first_dict():
    first = {"@type": "CreativeWork"}
    assert _get_item_reviewed({"itemReviewed": ["garbage", first, {"@type": "Thing"}]}) is first


@pytest.mark.parametrize("cr", [None, {}, {"itemReviewed": "A claim"}, {"itemReviewed": []}, {"itemReviewed": None}])
def test_missing_or_malformed_item_reviewed(cr):
    assert _get_item_reviewed(cr) is None
    assert _get_claimant(cr) is None
    assert _get_claim_date(cr) is None


# --- Claimant ------------------------------------------------------------------------

@pytest.mark.parametrize("author, expected", [
    ({"@type": "Person", "name": "Someone", "url": "https://someone.example"}, ("Someone", "https://someone.example")),
    ({"name": "Someone", "sameAs": ["https://a.example", "https://b.example"]}, ("Someone", "https://a.example")),
    ({"name": "Someone", "sameAs": "https://a.example"}, ("Someone", "https://a.example")),
    ({"name": "Someone"}, ("Someone", None)),
    ({"url": "https://someone.example"}, (None, "https://someone.example")),
    ("Someone", ("Someone", None)),
    ([{"name": "First"}, {"name": "Second"}], ("First", None)),
    ({"name": "Facebook Post", "url": "/"}, ("Facebook Post", None)),  # boombd.com states a relative URL
    ({"name": "Someone", "url": "someone.example/profile"}, ("Someone", "https://someone.example/profile")),
])
def test_claimant(author, expected):
    assert _get_claimant({"itemReviewed": {"@type": "Claim", "author": author}}) == expected


@pytest.mark.parametrize("author", [{}, {"@type": "Person"}, {"name": "  ", "sameAs": []}, "", [], None, 42])
def test_unknown_claimant_is_none(author):
    # Must be None (not a tuple of Nones) so that the next ClaimReview is considered
    assert _get_claimant({"itemReviewed": {"@type": "Claim", "author": author}}) is None


# --- Appearances ---------------------------------------------------------------------

def test_appearances_from_all_fields_are_collected():
    item = {
        "appearance": [{"url": "https://x.com/1"}, "https://x.com/2", {"url": "https://x.com/1"}, {"no": "url"}],
        "firstAppearance": {"@type": "CreativeWork", "url": "https://x.com/0"},
        "url": "https://claim.example",
    }
    assert _get_appearance_urls_for_item(item) == ["https://x.com/1", "https://x.com/2", "https://x.com/0"]


@pytest.mark.parametrize("item, expected", [
    ({"appearance": {"url": "https://x.com/1"}}, ["https://x.com/1"]),
    ({"appearance": "https://x.com/1"}, ["https://x.com/1"]),
    ({"firstAppearance": "https://x.com/0"}, ["https://x.com/0"]),
    ({"url": "https://claim.example"}, ["https://claim.example"]),  # Fallback
    ({"appearance": [], "url": "https://claim.example"}, ["https://claim.example"]),
    ({"appearance": [{"url": "https://x.com/" + "a" * 2100}]}, []),  # Too long
    ({}, []),
    (None, []),
])
def test_appearance_shapes(item, expected):
    assert _get_appearance_urls_for_item(item) == expected


# --- Claim date ----------------------------------------------------------------------

def test_claim_date():
    cr = {"itemReviewed": {"@type": "CreativeWork", "datePublished": "2024-01-01T00:00:00+00:00"}}
    assert _get_claim_date(cr) == datetime(2024, 1, 1)


def test_non_string_claim_date_is_ignored():
    assert _get_claim_date({"itemReviewed": {"datePublished": 1704067200}}) is None


# --- End-to-end with Google's converter ----------------------------------------------

def test_google_claim_data_reaches_stage_1():
    raw = [[
        "У продажі в Steam скоро з’явиться гра про ТЦК",
        ["Телеграм-канал «VOBLA • новости»"],
        1790121600,  # 2026-09-23
        [[["Вокс Україна", "voxukraine.org", None, "ua", "Ukraine"],
          "https://voxukraine.org/fejk-u-prodazhi-v-steam-skoro-z-yavytsya-gra-pro-ttsk",
          1790726400, "Фейк", None, None, "uk", ["ua"], "ФЕЙК: ...", [1, 1, 5], None, 1790751600]],
        [[[None, "t.me"], "https://t.me/voina_belgorod_novosti/110700", 1]],
        None, None, None, None, None, None, None,
        ["https://t.me/voina_belgorod_novosti/110700"],
    ]]  # Shape taken from a live GFCE response
    (cr,), _ = claim_reviews_from_raw(raw)

    assert _get_claimant(cr) == ("Телеграм-канал «VOBLA • новости»", None)
    assert _get_claim_date(cr) == datetime(2026, 9, 23)
    assert _get_appearance_urls_for_item(_get_item_reviewed(cr)) == ["https://t.me/voina_belgorod_novosti/110700"]


# --- URLs ----------------------------------------------------------------------------

@pytest.mark.parametrize("url, expected", [
    ("https://a.example/x", "https://a.example/x"),
    (" http://a.example ", "http://a.example"),
    ("a.example/x", "https://a.example/x"),
    ("/", None),
    ("/author/jane", None),
    ("https://", None),
    ("", None),
    (None, None),
    (42, None),
])
def test_absolute_url(url, expected):
    assert _absolute_url(url) == expected


def test_relative_publisher_and_author_urls_are_dropped():
    cr = {"author": [{"@type": "Organization", "name": "Org", "url": "/"}]}
    assert _get_publisher_name_url(cr) == ("Org", None)
    cr = {"author": [{"@type": "Person", "name": "Jane", "url": "/author/jane"}]}
    assert _get_author_name_url(cr) == ("Jane", None)
