"""Stage 1 response parsing and the guards that reject invalid candidates."""

import pytest

from veritas.gold_evidence.extraction import (
    _parse_confidence,
    _parse_enum,
    build_evidence,
    build_rationale,
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


def record(sources=None, **overrides) -> dict:
    base = {
        "proposition": "The video was posted on 2 May.",
        "sources": sources if sources is not None else [source_record()],
        "role": "essential",
        "reasoning": "The fact-check dates the video from this post.",
        "confidence": 0.9,
    }
    base.update(overrides)
    return base


def source_record(**overrides) -> dict:
    base = {
        "name": "Someone",
        "kind": "social_media_post",
        "locator": "https://x.com/someone/status/123",
        "proximity": "primary",
    }
    base.update(overrides)
    return base


def build(**overrides):
    """Builds an item; keyword arguments go to the record, `sources` included."""
    return build_evidence(
        record(**overrides),
        claim=FakeClaim(),
        review=FakeReview(),
        article=FakeArticle(),
        article_str=ARTICLE,
        excluded_domains={"factchecker.example"},
    )


# --- Response parsing ------------------------------------------------------

def records(response: str) -> list[dict]:
    return parse_extraction_response(response)[1]


def test_parses_the_documented_object():
    response = ('```json\n{"verdict_rationale": "5 is more than 3.", '
                '"evidence": [{"proposition": "p", "source_locator": "u"}]}\n```')
    rationale, parsed = parse_extraction_response(response)
    assert rationale == "5 is more than 3."
    assert parsed == [{"proposition": "p", "source_locator": "u"}]


def test_a_bare_list_still_yields_evidence():
    """A model answering in the older format must not cost us its evidence."""
    rationale, parsed = parse_extraction_response('[{"proposition": "p"}]')
    assert rationale is None
    assert parsed == [{"proposition": "p"}]


def test_repairs_trailing_commas_and_single_quotes():
    parsed = records('```json\n[{"proposition": "p", "confidence": 0.5,},]\n```')
    assert parsed and parsed[0]["proposition"] == "p"


def test_unwraps_a_differently_named_list():
    assert records('```json\n{"items": [{"proposition": "p"}]}\n```') == [{"proposition": "p"}]


def test_a_rationale_without_evidence_is_valid():
    """The case the rationale exists for: no external source, but an argument."""
    rationale, parsed = parse_extraction_response(
        '```json\n{"verdict_rationale": "The two figures contradict each other.", '
        '"evidence": []}\n```')
    assert rationale == "The two figures contradict each other."
    assert parsed == []


def test_empty_list_is_valid():
    assert records("```json\n[]\n```") == []


def test_unparseable_response_yields_nothing():
    assert parse_extraction_response("I could not find any evidence, sorry.") == (None, [])


# --- Guards ----------------------------------------------------------------

def test_valid_record_becomes_evidence():
    evidence = build()
    assert evidence is not None
    assert evidence.claim_id == 1
    assert evidence.review_id == 7
    assert evidence.article_id == 3
    assert evidence.sources[0].kind is SourceKind.SOCIAL_MEDIA_POST
    assert evidence.sources[0].proximity is ProximityLevel.PRIMARY
    assert evidence.role is EvidenceRole.ESSENTIAL
    assert evidence.admissible is None  # Stage 2 has not run yet


def test_locator_must_occur_in_the_article():
    """Guards against invented URLs, as stage 4 does for appearances."""
    assert build(sources=[source_record(locator="https://invented.example/never")]) is None


def test_locator_pointing_back_at_the_fact_checker_is_rejected():
    article = ARTICLE + " See our earlier piece at https://factchecker.example/other."
    assert build_evidence(
        record(sources=[source_record(locator="https://factchecker.example/other")]),
        claim=FakeClaim(), review=FakeReview(), article=FakeArticle(),
        article_str=article, excluded_domains={"factchecker.example"},
    ) is None


def test_tools_may_be_hosted_by_the_fact_checker():
    article = ARTICLE + " We used https://factchecker.example/tools/exif."
    evidence = build_evidence(
        record(sources=[source_record(locator="https://factchecker.example/tools/exif",
                                      kind="tool")]),
        claim=FakeClaim(), review=FakeReview(), article=FakeArticle(),
        article_str=article, excluded_domains={"factchecker.example"},
    )
    assert evidence is not None
    assert evidence.sources[0].kind is SourceKind.TOOL


def test_the_article_url_itself_is_rejected():
    article = ARTICLE + " https://factchecker.example/article"
    assert build_evidence(
        record(sources=[source_record(locator="https://factchecker.example/article")]),
        claim=FakeClaim(), review=FakeReview(), article=FakeArticle(),
        article_str=article, excluded_domains=set(),
    ) is None


def test_hallucinated_media_reference_is_rejected():
    assert build(proposition="The video <image:999999999> shows a crowd.") is None


def test_missing_proposition_is_rejected():
    assert build(proposition="") is None


def test_a_source_the_article_never_located_is_still_extracted():
    """Fact-checks do cite material they never link. Recording those is how the
    analysis can report how often it happens, instead of losing them silently."""
    evidence = build(sources=[source_record(locator="")])

    assert evidence is not None
    assert evidence.sources[0].locator is None


@pytest.mark.parametrize("kind", ["offline", "tool"])
def test_tools_and_offline_evidence_need_no_locator(kind):
    """A phone call has no URL, and not every tool has a public page."""
    evidence = build(sources=[source_record(locator="", kind=kind,
                                            name="Prof. Meier (phone interview)")])
    assert evidence is not None
    assert evidence.sources[0].locator is None
    assert evidence.sources[0].kind is SourceKind(kind)
    assert evidence.sources[0].name == "Prof. Meier (phone interview)"


def test_an_item_without_any_usable_source_is_dropped():
    assert build(sources=[]) is None
    assert build(sources=[source_record(locator="https://invented.example/never")]) is None


def test_an_unnamed_source_is_labelled_as_such():
    evidence = build(sources=[source_record(locator="", name="")])
    assert evidence.sources[0].name == "unnamed source"


def test_a_locator_less_item_still_needs_a_proposition():
    assert build(sources=[source_record(locator="", kind="offline")],
                 proposition="") is None


# --- Several sources for one proposition -----------------------------------

def test_every_source_of_a_proposition_is_kept():
    """Two outlets reporting the same fact are two sources of one item."""
    evidence = build(sources=[
        source_record(locator="https://x.com/someone/status/123"),
        source_record(locator="https://register.example.gov/entry/9", kind="government_record"),
    ])
    assert [s.locator for s in evidence.sources] == [
        "https://x.com/someone/status/123", "https://register.example.gov/entry/9"]


def test_an_invalid_source_does_not_take_the_item_down():
    evidence = build(sources=[
        source_record(locator="https://invented.example/never"),
        source_record(locator="https://register.example.gov/entry/9"),
    ])
    assert evidence is not None
    assert len(evidence.sources) == 1


def test_the_same_source_is_not_listed_twice():
    evidence = build(sources=[source_record(), source_record()])
    assert len(evidence.sources) == 1


def test_the_flat_single_source_spelling_is_still_accepted():
    """A model answering in the older format must not cost us its evidence."""
    evidence = build_evidence(
        {"proposition": "The video was posted on 2 May.",
         "source_name": "Someone", "source_kind": "social_media_post",
         "source_locator": "https://x.com/someone/status/123",
         "source_proximity": "primary", "role": "essential"},
        claim=FakeClaim(), review=FakeReview(), article=FakeArticle(),
        article_str=ARTICLE, excluded_domains=set(),
    )
    assert evidence is not None
    assert evidence.sources[0].name == "Someone"


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


def test_merging_collects_the_sources_of_both_articles():
    """Redundancy across two fact-checks of one claim ends up in one item."""
    a = build()
    b = build(sources=[source_record(locator="https://register.example.gov/entry/9")],
              proposition="The  video was posted on 2 May.")
    [merged] = deduplicate([a, b])

    assert len(merged.sources) == 2


def test_merging_keeps_the_stricter_role():
    """If one article's rationale leans on the proposition, losing it breaks that
    article's argument - whatever the other article made of it."""
    auxiliary = build(role="auxiliary")
    essential = build(role="essential",
                      sources=[source_record(locator="https://register.example.gov/entry/9")])
    [merged] = deduplicate([auxiliary, essential])

    assert merged.role is EvidenceRole.ESSENTIAL


def test_deduplication_orders_essential_first():
    essential = build(role="essential", confidence=0.5)
    background = build(role="background", confidence=0.99, proposition="Something else.")
    ordered = deduplicate([background, essential])
    assert [e.role for e in ordered] == [EvidenceRole.ESSENTIAL, EvidenceRole.BACKGROUND]


# --- Verdict rationale -----------------------------------------------------

def test_a_rationale_becomes_a_record():
    rationale = build_rationale("3 May precedes 5 May.", claim=FakeClaim(),
                                review=FakeReview(), article=FakeArticle(),
                                reasoning="trace")
    assert rationale is not None
    assert rationale.claim_id == 1
    assert rationale.review_id == 7
    assert rationale.article_id == 3
    assert rationale.extraction_reasoning == "trace"


@pytest.mark.parametrize("text", [None, "", "   "])
def test_no_rationale_is_not_a_record(text):
    assert build_rationale(text, claim=FakeClaim(), review=FakeReview(),
                           article=FakeArticle()) is None


def test_a_rationale_with_a_hallucinated_medium_is_dropped():
    """It would otherwise break every prompt it is rendered into."""
    assert build_rationale("As <image:999999999> shows, ...", claim=FakeClaim(),
                           review=FakeReview(), article=FakeArticle()) is None


# --- Corroboration groups --------------------------------------------------

