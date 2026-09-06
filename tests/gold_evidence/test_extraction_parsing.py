"""Stage 1 response parsing and the guards that reject invalid candidates."""

import pytest

from veritas.gold_evidence.extraction import (
    _parse_confidence,
    _parse_enum,
    build_evidence,
    deduplicate,
    parse_extraction_response,
)
from veritas.gold_evidence.models import EvidenceRole, ProximityLevel, SourceKind

ARTICLE = (
    "The video was first posted at https://x.com/someone/status/123 on 2 May. "
    "The city register at https://register.example.gov/entry/9 lists no such permit. "
    "We geolocated the scene with https://geotool.example.com."
)


class FakeClaim:
    id = 1
    data = "A claim"


class FakeReview:
    id = 7
    url = "https://factchecker.example/article"


class FakeArticle:
    id = 3


def record(**overrides) -> dict:
    base = {
        "proposition": "The video was posted on 2 May.",
        "source_name": "Someone",
        "source_kind": "social_media_post",
        "source_locator": "https://x.com/someone/status/123",
        "source_proximity": "primary",
        "role": "essential",
        "reasoning": "The fact-check dates the video from this post.",
        "confidence": 0.9,
    }
    base.update(overrides)
    return base


def build(**overrides):
    return build_evidence(
        record(**overrides),
        claim=FakeClaim(),
        review=FakeReview(),
        article=FakeArticle(),
        article_str=ARTICLE,
        excluded_domains={"factchecker.example"},
    )


# --- Response parsing ------------------------------------------------------

def test_parses_a_fenced_json_list():
    response = 'Here you go:\n```json\n[{"proposition": "p", "source_locator": "u"}]\n```'
    assert parse_extraction_response(response) == [{"proposition": "p", "source_locator": "u"}]


def test_parses_json_without_a_fence():
    assert parse_extraction_response('[{"proposition": "p"}]') == [{"proposition": "p"}]


def test_repairs_trailing_commas_and_single_quotes():
    response = '```json\n[{"proposition": "p", "confidence": 0.5,},]\n```'
    parsed = parse_extraction_response(response)
    assert parsed and parsed[0]["proposition"] == "p"


def test_unwraps_a_dict_around_the_list():
    response = '```json\n{"evidence": [{"proposition": "p"}]}\n```'
    assert parse_extraction_response(response) == [{"proposition": "p"}]


def test_empty_list_is_valid():
    assert parse_extraction_response("```json\n[]\n```") == []


def test_unparseable_response_yields_no_records():
    assert parse_extraction_response("I could not find any evidence, sorry.") == []


# --- Guards ----------------------------------------------------------------

def test_valid_record_becomes_evidence():
    evidence = build()
    assert evidence is not None
    assert evidence.claim_id == 1
    assert evidence.review_id == 7
    assert evidence.article_id == 3
    assert evidence.source.kind is SourceKind.SOCIAL_MEDIA_POST
    assert evidence.source.proximity is ProximityLevel.PRIMARY
    assert evidence.role is EvidenceRole.ESSENTIAL
    assert evidence.admissible is None  # Stage 2 has not run yet


def test_locator_must_occur_in_the_article():
    """Guards against invented URLs, as stage 4 does for appearances."""
    assert build(source_locator="https://invented.example/never-mentioned") is None


def test_locator_pointing_back_at_the_fact_checker_is_rejected():
    article = ARTICLE + " See our earlier piece at https://factchecker.example/other."
    assert build_evidence(
        record(source_locator="https://factchecker.example/other"),
        claim=FakeClaim(), review=FakeReview(), article=FakeArticle(),
        article_str=article, excluded_domains={"factchecker.example"},
    ) is None


def test_tools_may_be_hosted_by_the_fact_checker():
    article = ARTICLE + " We used https://factchecker.example/tools/exif."
    evidence = build_evidence(
        record(source_locator="https://factchecker.example/tools/exif",
               source_kind="tool"),
        claim=FakeClaim(), review=FakeReview(), article=FakeArticle(),
        article_str=article, excluded_domains={"factchecker.example"},
    )
    assert evidence is not None
    assert evidence.source.kind is SourceKind.TOOL


def test_the_article_url_itself_is_rejected():
    article = ARTICLE + " https://factchecker.example/article"
    assert build_evidence(
        record(source_locator="https://factchecker.example/article"),
        claim=FakeClaim(), review=FakeReview(), article=FakeArticle(),
        article_str=article, excluded_domains=set(),
    ) is None


def test_hallucinated_media_reference_is_rejected():
    assert build(proposition="The video <image:999999999> shows a crowd.") is None


def test_missing_proposition_is_rejected():
    assert build(proposition="") is None


def test_a_locator_is_required_for_ordinary_sources():
    assert build(source_locator="") is None


@pytest.mark.parametrize("kind", ["offline", "tool"])
def test_tools_and_offline_evidence_need_no_locator(kind):
    """A phone call has no URL, and not every tool has a public page."""
    evidence = build(source_locator="", source_kind=kind,
                     source_name="Prof. Meier (phone interview)")
    assert evidence is not None
    assert evidence.source.locator is None
    assert evidence.source.kind is SourceKind(kind)
    assert evidence.source.name == "Prof. Meier (phone interview)"


def test_a_locator_less_item_still_needs_a_proposition():
    assert build(source_locator="", source_kind="offline", proposition="") is None


def test_locator_less_items_are_deduplicated_by_proposition():
    a = build(source_locator="", source_kind="offline", confidence=0.3)
    b = build(source_locator="", source_kind="offline", confidence=0.7)
    deduplicated = deduplicate([a, b])
    assert len(deduplicated) == 1
    assert deduplicated[0].extraction_confidence == pytest.approx(0.7)


# --- Value coercion --------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    ("primary", ProximityLevel.PRIMARY),
    ("PRIMARY", ProximityLevel.PRIMARY),
    ("Secondary ", ProximityLevel.SECONDARY),
    ("nonsense", ProximityLevel.SECONDARY),
    (None, ProximityLevel.SECONDARY),
])
def test_enum_coercion_falls_back_to_the_default(value, expected):
    assert _parse_enum(value, ProximityLevel, ProximityLevel.SECONDARY) is expected


def test_enum_coercion_normalizes_separators():
    assert _parse_enum("news-article", SourceKind, SourceKind.OTHER) is SourceKind.NEWS_ARTICLE
    assert _parse_enum("social media post", SourceKind, SourceKind.OTHER) \
           is SourceKind.SOCIAL_MEDIA_POST


@pytest.mark.parametrize("value, expected", [
    (0.9, 0.9),
    ("0.4", 0.4),
    (90, 0.9),      # answered in percent
    (1.5, 0.015),   # also percent
    (-1, 0.0),
    (None, 0.5),
    ("high", 0.5),
])
def test_confidence_coercion(value, expected):
    assert _parse_confidence(value) == pytest.approx(expected)


# --- Deduplication ---------------------------------------------------------

def test_duplicates_are_merged_keeping_the_higher_confidence():
    items = [build(confidence=0.4), build(confidence=0.8)]
    deduplicated = deduplicate(items)
    assert len(deduplicated) == 1
    assert deduplicated[0].extraction_confidence == pytest.approx(0.8)


def test_deduplication_ignores_trailing_slashes_and_whitespace():
    a = build(source_locator="https://x.com/someone/status/123")
    b = build(source_locator="https://x.com/someone/status/123",
              proposition="The  video was posted on 2 May.")
    b.source.locator += "/"
    assert len(deduplicate([a, b])) == 1


def test_deduplication_orders_essential_first():
    essential = build(role="essential", confidence=0.5)
    background = build(role="background", confidence=0.99,
                       source_locator="https://register.example.gov/entry/9")
    ordered = deduplicate([background, essential])
    assert [e.role for e in ordered] == [EvidenceRole.ESSENTIAL, EvidenceRole.BACKGROUND]
