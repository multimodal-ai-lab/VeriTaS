"""The evidence browser's row-to-payload conversion.

A browser row is one *source*, joined with the evidence item it belongs to.
`evidence_summary` is deliberately lighter than `evidence_source_payload` - the
long texts and the JSONB blobs stay in the database - but everything a list entry
shows must survive, including the claim context the row was joined with.
"""

from datetime import datetime

import pytest

from webui.serialization import evidence_list_payload, evidence_summary, temporal_zone

from tests.webui.conftest import FakeRegistry


def make_row(**overrides) -> dict:
    row = {
        "id": 1,
        "evidence_id": 4,
        "evidence_admissible": True,
        "n_sources": 2,
        "claim_id": 100,
        "review_id": 7,
        "article_id": 9,
        "proposition": "The bridge collapsed <image:5> on 3 March.",
        "source_name": "Reuters",
        "source_kind": "news_article",
        "source_locator": "https://www.reuters.com/world/story",
        "source_proximity": "secondary",
        "source_domain": "reuters.com",
        "available_since": datetime(2024, 3, 5),
        "role": "key",
        "accessed_at": datetime(2024, 6, 1),
        "extraction_confidence": 0.82,
        "accessible": True,
        "faithfulness_assessment": 0.667,
        "faithfulness_justification": "The article states it verbatim.",
        "before_fact_check": True,
        "before_claim": False,
        "later_event": False,
        "later_event_justification": "Reporting only.",
        "temporal_justification": "Published after the claim.",
        "admissible": True,
        "inadmissibility_reason": None,
        "dismissed": False,
        "dismissed_reason": None,
        "deferred_until": None,
        "created_at": datetime(2024, 6, 1),
        "updated_at": datetime(2024, 6, 2),
        "claim_data": "A bridge collapsed in March.",
        "claim_date": datetime(2024, 3, 3),
        "claim_language": "en",
        "claim_status": "accepted",
    }
    row.update(overrides)
    return row


def test_summary_renders_the_proposition_multimodally():
    summary = evidence_summary(make_row(), FakeRegistry())
    assert summary["proposition"]["is_multimodal"] is True
    assert summary["proposition"]["media"][0]["reference"] == "<image:5>"


def test_summary_carries_the_claim_context():
    summary = evidence_summary(make_row(), FakeRegistry())
    assert summary["claim"] == {
        "id": 100,
        "data": "A bridge collapsed in March.",
        "t_c": datetime(2024, 3, 3),
        "language": "en",
        "status": "accepted",
    }


def test_summary_leaves_the_heavy_fields_out():
    summary = evidence_summary(make_row(), FakeRegistry())
    for key in ("full_evidence", "columns"):
        assert key not in summary


def test_summary_prefers_the_domain_computed_in_sql():
    summary = evidence_summary(make_row(source_domain="bbc.co.uk"), FakeRegistry())
    assert summary["source"]["domain"] == "bbc.co.uk"


def test_summary_derives_the_domain_when_sql_did_not():
    summary = evidence_summary(make_row(source_domain=None), FakeRegistry())
    assert summary["source"]["domain"] == "reuters.com"


def test_summary_without_any_judgement():
    summary = evidence_summary(
        make_row(faithfulness_assessment=None, before_fact_check=None, before_claim=None),
        FakeRegistry(),
    )
    assert summary["faithfulness"] is None
    assert summary["temporal_validation"] is None
    assert summary["zone"] is None


def test_summary_carries_the_items_later_event_judgement():
    summary = evidence_summary(make_row(later_event=True), FakeRegistry())
    assert summary["later_event"]["change_detected"] is True


@pytest.mark.parametrize("before_claim,before_fact_check,expected", [
    (True, True, "before_claim"),
    (False, True, "in_window"),
    (False, False, "after_fact_check"),
    # An item the temporal check never reached stays unclassified rather than
    # being silently counted as post-fact-check.
    (None, None, None),
])
def test_temporal_zone(before_claim, before_fact_check, expected):
    assert temporal_zone(before_claim, before_fact_check) == expected


def test_list_payload_keeps_the_pagination_fields():
    page = {"total": 42, "limit": 25, "offset": 25, "evidence": [make_row()]}
    payload = evidence_list_payload(page, FakeRegistry())
    assert payload["total"] == 42
    assert payload["limit"] == 25
    assert payload["offset"] == 25
    assert payload["evidence"][0]["id"] == 1


def test_list_payload_of_an_empty_page():
    payload = evidence_list_payload({"total": 0, "limit": 25, "offset": 0}, FakeRegistry())
    assert payload["evidence"] == []
