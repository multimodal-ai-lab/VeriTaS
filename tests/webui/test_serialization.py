"""Row-to-payload conversion: nothing stored may get lost on the way to the UI."""

from datetime import datetime

import pytest

from webui.serialization import (
    PREVIEW_MEDIA_LIMIT,
    build_timeline,
    claim_list_payload,
    claim_summary_payload,
    evidence_item_payload,
    evidence_source_payload,
    render_text,
)

from tests.webui.conftest import FakeRegistry

T_C = datetime(2024, 3, 1)
T_F = datetime(2024, 3, 20)


def make_row(**overrides) -> dict:
    """One source, joined with the evidence item it belongs to."""
    row = {
        "id": 1,
        "evidence_id": 4,
        "claim_id": 100,
        "review_id": 7,
        "article_id": 9,
        "proposition": "The bridge collapsed <image:5> on 3 March.",
        "evidence_admissible": True,
        "n_sources": 2,
        "source_name": "Reuters",
        "source_kind": "news_article",
        "source_locator": "https://www.reuters.com/world/story",
        "source_proximity": "secondary",
        "source_raw_content": "Report text with <video:8>.",
        "available_since": datetime(2024, 3, 5),
        "role": "essential",
        "accessed_at": datetime(2024, 6, 1),
        "extraction_reasoning": "Stated in the second paragraph.",
        "extraction_confidence": 0.82,
        "accessible": True,
        "faithfulness_assessment": 0.667,
        "faithfulness_reasoning": "provider trace",
        "faithfulness_justification": "The article states it verbatim.",
        "before_fact_check": True,
        "before_claim": False,
        "later_event": False,
        "temporal_reasoning": "temporal trace",
        "temporal_justification": "Published after the claim.",
        "admissible": True,
        "inadmissibility_reason": None,
        "dismissed": False,
        "dismissed_reason": None,
        "deferred_until": None,
        "created_at": datetime(2024, 6, 1),
        "updated_at": datetime(2024, 6, 2),
        "full_source": {
            "name": "Reuters", "kind": "news_article", "proximity": "secondary",
            "faithfulness": {"assessment": 0.667, "rater": "openai:gpt-5"},
            "temporal_validation": {"before_claim": False, "rater": "openai:gpt-5"},
            "a_field_added_later": "still visible",
        },
        "full_evidence": {
            "proposition": "The bridge collapsed <image:5> on 3 March.",
            "role": "essential",
        },
    }
    row.update(overrides)
    return row


def make_item(**overrides) -> dict:
    """One evidence item with the sources that report its proposition."""
    item = {
        "id": 4,
        "claim_id": 100,
        "review_id": 7,
        "article_id": 9,
        "proposition": "The bridge collapsed <image:5> on 3 March.",
        "role": "essential",
        "extraction_reasoning": "Stated in the second paragraph.",
        "extraction_confidence": 0.82,
        "admissible": True,
        "inadmissibility_reason": None,
        "n_sources": 2,
        "n_admissible_sources": 1,
        "available_since": datetime(2024, 3, 5),
        "dismissed": False,
        "dismissed_reason": None,
        "created_at": datetime(2024, 6, 1),
        "updated_at": datetime(2024, 6, 2),
        "full_evidence": {"proposition": "The bridge collapsed <image:5> on 3 March.",
                          "an_item_field_added_later": "still visible"},
        "sources": [make_row(), make_row(id=2, source_name="AP", admissible=False,
                                         inadmissibility_reason="inaccessible")],
    }
    item.update(overrides)
    return item


def test_render_text_keeps_references_inline_and_lists_the_media():
    rendered = render_text("A <image:5> B", FakeRegistry())
    assert rendered["is_multimodal"] is True
    assert rendered["n_media"] == 1
    assert [part["type"] for part in rendered["segments"]] == ["text", "media", "text"]
    assert rendered["media"][0]["reference"] == "<image:5>"


def test_render_text_of_none():
    rendered = render_text(None, FakeRegistry())
    assert rendered == {"text": None, "segments": [], "media": [],
                        "n_media": 0, "is_multimodal": False}


def test_render_text_marks_a_reference_whose_file_is_gone():
    rendered = render_text("A <image:5>", FakeRegistry(missing={"<image:5>"}))
    assert rendered["media"][0]["exists"] is False
    # The reference is still a segment, so the UI can show it as broken.
    assert rendered["segments"][-1]["reference"] == "<image:5>"


def test_source_payload_exposes_the_structured_view():
    payload = evidence_source_payload(make_row(), FakeRegistry())
    assert payload["id"] == 1
    assert payload["evidence_id"] == 4
    assert payload["source"]["domain"] == "reuters.com"
    assert payload["source"]["content"]["n_media"] == 1
    assert payload["proposition"]["is_multimodal"] is True
    assert payload["faithfulness"]["assessment"] == pytest.approx(0.667)
    assert payload["faithfulness"]["rater"] == "openai:gpt-5"
    assert payload["temporal_validation"]["before_claim"] is False
    assert payload["extraction"]["confidence"] == pytest.approx(0.82)


def test_source_payload_keeps_every_column_and_the_full_blob():
    row = make_row()
    payload = evidence_source_payload(row, FakeRegistry())
    for column in row:
        if column in ("full_source", "full_evidence"):
            continue
        assert column in payload["columns"], f"{column} was dropped"
    # Fields the model gained after this UI was written stay visible.
    assert payload["full_source"]["a_field_added_later"] == "still visible"


def test_source_payload_falls_back_to_the_blob_when_a_column_is_null():
    row = make_row(source_name=None, faithfulness_assessment=None)
    payload = evidence_source_payload(row, FakeRegistry())
    assert payload["source"]["name"] == "Reuters"
    assert payload["faithfulness"]["assessment"] == pytest.approx(0.667)


def test_source_payload_without_any_judgement():
    row = make_row(
        faithfulness_assessment=None, before_fact_check=None, before_claim=None,
        full_source={"name": "s"},
    )
    payload = evidence_source_payload(row, FakeRegistry())
    assert payload["faithfulness"] is None
    assert payload["temporal_validation"] is None


def test_source_payload_carries_the_item_it_belongs_to():
    """A source is only interpretable together with the proposition it reports."""
    payload = evidence_source_payload(make_row(), FakeRegistry())
    assert payload["proposition"]["is_multimodal"] is True
    assert payload["role"] == "essential"
    assert payload["evidence_admissible"] is True
    assert payload["n_sources"] == 2


def test_item_payload_nests_its_sources():
    payload = evidence_item_payload(make_item(), FakeRegistry())

    assert payload["id"] == 4
    assert payload["proposition"]["is_multimodal"] is True
    assert [source["source"]["name"] for source in payload["sources"]] == ["Reuters", "AP"]
    assert payload["n_admissible_sources"] == 1
    # The item survived although one of its sources did not.
    assert payload["admissible"] is True
    assert payload["sources"][1]["admissible"] is False


def test_item_payload_keeps_every_column_and_the_full_blob():
    item = make_item()
    payload = evidence_item_payload(item, FakeRegistry())
    for column in item:
        if column in ("full_evidence", "sources"):
            continue
        assert column in payload["columns"], f"{column} was dropped"
    assert payload["full_evidence"]["an_item_field_added_later"] == "still visible"


def test_item_payload_without_sources():
    """An item whose sources were all removed still renders."""
    payload = evidence_item_payload(make_item(sources=[], n_sources=0), FakeRegistry())
    assert payload["sources"] == []


def timeline_item(item_id: int, *sources: dict) -> dict:
    """An evidence item whose sources carry the dates the timeline plots."""
    return {"id": item_id, "admissible": True, "role": "essential", "sources": list(sources)}


def timeline_source(source_id: int, available_since, **overrides) -> dict:
    source = {"id": source_id, "available_since": available_since,
              "admissible": True, "temporal_validation": {}}
    source.update(overrides)
    return source


def test_build_timeline_classifies_sources_into_the_three_zones():
    claim = {"t_c": T_C, "t_f": T_F}
    evidence = [
        timeline_item(1, timeline_source(
            11, datetime(2024, 2, 1),
            temporal_validation={"before_claim": True, "before_fact_check": True})),
        timeline_item(2, timeline_source(
            12, datetime(2024, 3, 10),
            temporal_validation={"before_claim": False, "before_fact_check": True})),
        timeline_item(3, timeline_source(
            13, datetime(2024, 4, 1),
            temporal_validation={"before_claim": False, "before_fact_check": False})),
        timeline_item(4, timeline_source(14, None, temporal_validation=None)),
    ]
    timeline = build_timeline(claim, evidence)
    assert [point["zone"] for point in timeline["points"]] == [
        "before_claim", "in_window", "after_fact_check",
    ]
    assert timeline["n_undated"] == 1


def test_build_timeline_plots_every_source_of_an_item():
    """An item can have sources in different zones; each gets its own dot."""
    claim = {"t_c": T_C, "t_f": T_F}
    evidence = [timeline_item(1,
                              timeline_source(11, datetime(2024, 2, 1)),
                              timeline_source(12, datetime(2024, 3, 10)))]
    points = build_timeline(claim, evidence)["points"]

    assert [point["source_id"] for point in points] == [11, 12]
    assert [point["zone"] for point in points] == ["before_claim", "in_window"]
    # Every dot names the item it belongs to, which is what the UI scrolls to.
    assert {point["evidence_id"] for point in points} == {1}


def test_build_timeline_computes_the_zone_when_the_flags_are_missing():
    claim = {"t_c": T_C, "t_f": T_F}
    evidence = [timeline_item(1, timeline_source(11, datetime(2024, 3, 10)))]
    assert build_timeline(claim, evidence)["points"][0]["zone"] == "in_window"


def test_build_timeline_sorts_the_points_chronologically():
    claim = {"t_c": T_C, "t_f": T_F}
    evidence = [
        timeline_item(1, timeline_source(11, datetime(2024, 3, 15))),
        timeline_item(2, timeline_source(12, datetime(2024, 2, 15))),
    ]
    points = build_timeline(claim, evidence)["points"]
    assert [point["evidence_id"] for point in points] == [2, 1]


# ------------------------------------------------------- claim list previews

def make_claim_row(**overrides) -> dict:
    row = {
        "id": 4711,
        "data": "This photo <image:9> and this clip <video:4> prove it.",
        "t_c": T_C, "t_f": T_F, "language": "en", "status": "accepted",
        "n_evidence": 3, "n_admissible": 1,
    }
    row.update(overrides)
    return row


def test_claim_summary_payload_resolves_the_claim_media():
    payload = claim_summary_payload(make_claim_row(), FakeRegistry())
    assert payload["id"] == 4711
    assert payload["content"]["n_media"] == 2
    assert [item["reference"] for item in payload["content"]["media"]] == [
        "<image:9>", "<video:4>"]
    assert payload["content"]["n_hidden_media"] == 0
    # The original row is left untouched alongside the rendered content.
    assert payload["n_admissible"] == 1


def test_claim_summary_payload_caps_the_preview_and_reports_the_rest():
    data = " ".join(f"<image:{identifier}>" for identifier in range(1, 8))
    payload = claim_summary_payload(make_claim_row(data=data), FakeRegistry())
    assert len(payload["content"]["media"]) == PREVIEW_MEDIA_LIMIT
    assert payload["content"]["n_hidden_media"] == 7 - PREVIEW_MEDIA_LIMIT
    # The segment list still describes the whole claim, not just the preview.
    assert sum(1 for part in payload["content"]["segments"]
               if part["type"] == "media") == 7


def test_claim_summary_payload_of_a_text_only_claim():
    payload = claim_summary_payload(make_claim_row(data="no media here"), FakeRegistry())
    assert payload["content"]["media"] == []
    assert payload["content"]["is_multimodal"] is False
    assert payload["content"]["n_hidden_media"] == 0


def test_claim_list_payload_keeps_the_pagination_fields():
    page = {"total": 137, "limit": 25, "offset": 50, "claims": [make_claim_row()]}
    payload = claim_list_payload(page, FakeRegistry())
    assert payload["total"] == 137
    assert payload["limit"] == 25
    assert payload["offset"] == 50
    assert payload["claims"][0]["content"]["n_media"] == 2


def test_claim_list_payload_of_an_empty_page():
    assert claim_list_payload({"total": 0, "claims": []}, FakeRegistry())["claims"] == []
