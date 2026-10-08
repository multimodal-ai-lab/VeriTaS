"""Pure tests for the normalized review match keys and for how Stage 1 decides
between inserting and updating reviews based on them. Several cases stem from
spelling variants that created duplicate reviews in production."""

import pytest

from veritas.db.veritas_db import dedupe_by_match_key, plan_claim_review_writes
from veritas.util.review_key import hash_int64, normalize_claim, normalize_review_url, review_match_key


# --- URL normalization --------------------------------------------------------------

@pytest.mark.parametrize("variant", [
    "https://www.politifact.com/factchecks/2018/dec/10/mike-romano/poverty/",
    "http://www.politifact.com/factchecks/2018/dec/10/mike-romano/poverty/",
    "https://politifact.com/factchecks/2018/dec/10/mike-romano/poverty",
    "politifact.com/factchecks/2018/dec/10/mike-romano/poverty/",
    "https://POLITIFACT.com/factchecks/2018/dec/10/mike-romano/poverty//",
    "https://www.politifact.com/factchecks/2018/dec/10/mike-romano/poverty/#comments",
    "https://www.politifact.com/factchecks/2018/dec/10/mike-romano/poverty/?utm_source=x&fbclid=abc",
    "https://www.politifact.com:443/factchecks/2018/dec/10/mike-romano/poverty/",
    "  https://www.politifact.com/factchecks/2018/dec/10/mike-romano/poverty/  ",
])
def test_url_variants_share_normalized_url(variant):
    assert normalize_review_url(variant) == "politifact.com/factchecks/2018/dec/10/mike-romano/poverty"


def test_url_percent_encoding_is_decoded():
    encoded = "https://fatabyyano.net/%D8%A7%D9%85%D8%B1%D8%A3%D8%A9/"
    decoded = "https://fatabyyano.net/امرأة/"
    assert normalize_review_url(encoded) == normalize_review_url(decoded)


def test_url_path_case_is_kept():
    # Paths may be case-sensitive (e.g., AFP's article slugs)
    assert (normalize_review_url("https://factcheckarabic.afp.com/2018-Video-of-Macron")
            != normalize_review_url("https://factcheckarabic.afp.com/2018-video-of-macron"))


def test_url_identifying_query_is_kept_and_sorted():
    assert normalize_review_url("https://example.com/?p=1") != normalize_review_url("https://example.com/?p=2")
    assert (normalize_review_url("https://example.com/a?b=2&a=1&utm_medium=x")
            == normalize_review_url("https://example.com/a?a=1&b=2"))


def test_url_different_subdomains_stay_apart():
    assert (normalize_review_url("https://factcheck.afp.com/x")
            != normalize_review_url("https://factcheckarabic.afp.com/x"))


def test_url_non_default_port_is_kept():
    assert normalize_review_url("http://example.com:8080/a") == "example.com:8080/a"


def test_malformed_url_does_not_raise():
    assert normalize_review_url("https://example.com:notaport/a") == "https://example.com:notaport/a"


# --- Claim normalization ------------------------------------------------------------

@pytest.mark.parametrize("a, b", [
    # Straight vs. curly quotes
    ('The poverty rate in West Virginia “was 19.1 percent, the fourth-highest in the country.”',
     'The poverty rate in West Virginia "was 19.1 percent, the fourth-highest in the country."'),
    ('"Tina (Smith) profited from the opioid crisis."',
     '“Tina (Smith) profited from the opioid crisis.”'),
    ("Says it’s true", "Says it's true"),
    # HTML line breaks and trailing markup
    ("Alcohol can shrink and damage the brain as shown in the post?<br/>",
     "Alcohol can shrink and damage the brain as shown in the post?"),
    ("“Pureval’s lobbying firm made millions helping Libya reduce<br/>payments owed to families.”",
     "“Pureval’s lobbying firm made millions helping Libya reduce payments owed to families.”"),
    # Whitespace, case, entities, non-breaking spaces
    ("  Vaccines   contain\nmicrochips ", "vaccines contain microchips"),
    ("Tom &amp; Jerry", "Tom & Jerry"),
    ("Vaccines contain microchips", "Vaccines contain microchips"),
])
def test_claim_variants_share_normalized_claim(a, b):
    assert normalize_claim(a) == normalize_claim(b)


@pytest.mark.parametrize("a, b", [
    # Different claims checked within the same article must stay apart
    ('Haley brought in "Syrian refugees and she got criticized for that."',
     "(Haley) said recently that the age of Social Security is way too low."),
    ("Teachers spend $1.6 BILLION per year on school supplies.",
     "With this plan, the typical family of four will save $1,182 a year on their taxes."),
    # The wording is kept, incl. verdict labels (they get stripped at a later stage)
    ("НЕПРАВДА: В американському місті Санді-Спрінгс працює всього 7 чиновників",
     "В американському місті Санді-Спрінгс працює всього 7 чиновників"),
    ("FALSE: Vaccines contain microchips", "Vaccines contain microchips"),
])
def test_different_claims_stay_apart(a, b):
    assert normalize_claim(a) != normalize_claim(b)


def test_hash_int64_is_deterministic_and_fits_bigint():
    value = hash_int64("some claim")
    assert value == hash_int64("some claim")
    assert -2 ** 63 <= value < 2 ** 63


def test_review_match_key_combines_url_and_claim():
    key_a = review_match_key("https://www.example.com/a/", "“Claim”")
    key_b = review_match_key("http://example.com/a", '"claim"')
    key_c = review_match_key("http://example.com/a", "Another claim")
    assert key_a == key_b
    assert key_a != key_c


# --- Insert/update planning ---------------------------------------------------------

def test_dedupe_by_match_key_keeps_first_variant():
    crs = {
        ("https://www.example.com/a/", "“Claim”"): {"v": 1},
        ("http://example.com/a", '"Claim"'): {"v": 2},
        ("http://example.com/a", "Another claim"): {"v": 3},
    }
    entries = dedupe_by_match_key(crs)
    assert list(entries.values()) == [
        ("https://www.example.com/a/", "“Claim”", {"v": 1}),
        ("http://example.com/a", "Another claim", {"v": 3}),
    ]


def test_plan_claim_review_writes():
    entries = [
        ("https://example.com/new", "New claim", {"v": "new"}),
        ("https://example.com/same", "Same claim", {"v": "same"}),
        ("https://example.com/changed", "Changed claim", {"v": "changed"}),
    ]
    matches = [
        (None, None),
        (7, {"v": "same"}),
        (8, {"v": "old"}),
    ]
    to_insert, to_update = plan_claim_review_writes(entries, matches)
    assert to_insert == [("https://example.com/new", "New claim", {"v": "new"})]
    assert to_update == [(8, {"v": "changed"})]


def test_plan_claim_review_writes_fills_missing_source():
    # A review known from the other source has no ClaimReview of this source yet
    to_insert, to_update = plan_claim_review_writes(
        [("https://example.com/a", "Claim", {"v": 1})], [(3, None)]
    )
    assert to_insert == []
    assert to_update == [(3, {"v": 1})]
