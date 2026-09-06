"""Row-to-payload conversion: nothing stored may get lost on the way to the UI."""

from datetime import datetime

import pytest

from webui.serialization import build_timeline, evidence_payload, render_text

T_C = datetime(2024, 3, 1)
T_F = datetime(2024, 3, 20)


class FakeRegistry:
    """Stands in for the ezMM registry: `missing` references resolve to nothing."""

    def __init__(self, missing=()):
        self.missing = set(missing)

    def describe(self, references):
        return [
            {
                "reference": reference,
                "kind": reference.strip("<>").split(":")[0],
                "id": int(reference.strip("<>").split(":")[1]),
                "exists": reference not in self.missing,
                "url": f"/api/media/{reference.strip('<>').replace(':', '/')}",
            }
            for reference in references
        ]

    def get_by_reference(self, reference):
        return None


def make_row(**overrides) -> dict:
    row = {
        "id": 1,
        "claim_id": 100,
        "review_id": 7,
        "article_id": 9,
        "proposition": "The bridge collapsed <image:5> on 3 March.",
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
        "full_evidence": {
            "proposition": "The bridge collapsed <image:5> on 3 March.",
            "source": {"name": "Reuters", "kind": "news_article", "proximity": "secondary"},
            "faithfulness": {"assessment": 0.667, "rater": "openai:gpt-5"},
            "temporal_validation": {"before_claim": False, "rater": "openai:gpt-5"},
            "a_field_added_later": "still visible",
        },
    }
    row.update(overrides)
    return row


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


def test_evidence_payload_exposes_the_structured_view():
    payload = evidence_payload(make_row(), FakeRegistry())
    assert payload["id"] == 1
    assert payload["source"]["domain"] == "reuters.com"
    assert payload["source"]["content"]["n_media"] == 1
    assert payload["proposition"]["is_multimodal"] is True
    assert payload["faithfulness"]["assessment"] == pytest.approx(0.667)
    assert payload["faithfulness"]["rater"] == "openai:gpt-5"
    assert payload["temporal_validation"]["before_claim"] is False
    assert payload["extraction"]["confidence"] == pytest.approx(0.82)


def test_evidence_payload_keeps_every_column_and_the_full_blob():
    row = make_row()
    payload = evidence_payload(row, FakeRegistry())
    for column in row:
        if column == "full_evidence":
            continue
        assert column in payload["columns"], f"{column} was dropped"
    # Fields the model gained after this UI was written stay visible.
    assert payload["full_evidence"]["a_field_added_later"] == "still visible"


def test_evidence_payload_falls_back_to_the_blob_when_a_column_is_null():
    row = make_row(source_name=None, faithfulness_assessment=None)
    payload = evidence_payload(row, FakeRegistry())
    assert payload["source"]["name"] == "Reuters"
    assert payload["faithfulness"]["assessment"] == pytest.approx(0.667)


def test_evidence_payload_without_any_judgement():
    row = make_row(
        faithfulness_assessment=None, before_fact_check=None, before_claim=None,
        full_evidence={"proposition": "x", "source": {"name": "s"}},
    )
    payload = evidence_payload(row, FakeRegistry())
    assert payload["faithfulness"] is None
    assert payload["temporal_validation"] is None


def test_build_timeline_classifies_items_into_the_three_zones():
    claim = {"t_c": T_C, "t_f": T_F}
    evidence = [
        {"id": 1, "available_since": datetime(2024, 2, 1),
         "temporal_validation": {"before_claim": True, "before_fact_check": True}},
        {"id": 2, "available_since": datetime(2024, 3, 10),
         "temporal_validation": {"before_claim": False, "before_fact_check": True}},
        {"id": 3, "available_since": datetime(2024, 4, 1),
         "temporal_validation": {"before_claim": False, "before_fact_check": False}},
        {"id": 4, "available_since": None, "temporal_validation": None},
    ]
    timeline = build_timeline(claim, evidence)
    assert [point["zone"] for point in timeline["points"]] == [
        "before_claim", "in_window", "after_fact_check",
    ]
    assert timeline["n_undated"] == 1


def test_build_timeline_computes_the_zone_when_the_flags_are_missing():
    claim = {"t_c": T_C, "t_f": T_F}
    evidence = [{"id": 1, "available_since": datetime(2024, 3, 10), "temporal_validation": {}}]
    assert build_timeline(claim, evidence)["points"][0]["zone"] == "in_window"


def test_build_timeline_sorts_the_points_chronologically():
    claim = {"t_c": T_C, "t_f": T_F}
    evidence = [
        {"id": 1, "available_since": datetime(2024, 3, 15), "temporal_validation": {}},
        {"id": 2, "available_since": datetime(2024, 2, 15), "temporal_validation": {}},
    ]
    points = build_timeline(claim, evidence)["points"]
    assert [point["evidence_id"] for point in points] == [2, 1]
