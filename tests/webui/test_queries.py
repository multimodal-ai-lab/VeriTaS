"""The SQL filter builder: user input must always become a bound parameter."""

from datetime import date, datetime

import pytest

from webui.queries import (
    MEDIA_REF_SQL,
    SORTABLE,
    as_datetime,
    build_claim_filters,
    claim_summary,
    delta_summary,
    weighted_series,
)


def test_default_filter_restricts_to_processed_claims():
    where, args = build_claim_filters()
    assert where == "c.gold_evidence_status IS NOT NULL"
    assert args == []


def test_status_filter_is_parameterized():
    where, args = build_claim_filters(statuses=["accepted", "rejected"])
    assert where == "(c.gold_evidence_status = ANY ($1::text[]))"
    assert args == [["accepted", "rejected"]]


def test_unprocessed_can_be_combined_with_concrete_statuses():
    where, args = build_claim_filters(statuses=["accepted"], wants_unprocessed=True)
    assert where == ("(c.gold_evidence_status = ANY ($1::text[]) "
                     "OR c.gold_evidence_status IS NULL)")
    assert args == [["accepted"]]


def test_search_term_never_reaches_the_sql_text():
    where, args = build_claim_filters(query="'; DROP TABLE claims; --")
    assert "DROP TABLE" not in where
    assert args[-1] == "'; DROP TABLE claims; --"
    assert "ILIKE '%' || $1 || '%'" in where


def test_placeholders_are_numbered_consecutively():
    where, args = build_claim_filters(
        statuses=["accepted"],
        reason="insufficient_evidence",
        query="flood",
        date_from=date(2024, 1, 1),
        date_to=date(2024, 12, 31),
    )
    assert len(args) == 5
    for index in range(1, 6):
        assert f"${index}" in where


def test_boolean_filters_use_fixed_fragments():
    where, args = build_claim_filters(released=True, has_media=False)
    assert "(c.released_quarter OR c.released_longitudinal)" in where
    assert f"c.data !~ '{MEDIA_REF_SQL}'" in where
    assert args == []


def test_has_media_true_uses_the_matching_operator():
    where, _ = build_claim_filters(has_media=True)
    assert f"c.data ~ '{MEDIA_REF_SQL}'" in where


def test_sortable_columns_are_whitelisted():
    # Anything the API accepts as `sort` must resolve to a known column.
    assert set(SORTABLE) == {"id", "date", "updated", "status"}
    for column in SORTABLE.values():
        assert column.startswith("c.")


@pytest.mark.parametrize("value,end_of_day,expected_hour", [
    (date(2024, 5, 1), False, 0),
    (date(2024, 5, 1), True, 23),
])
def test_as_datetime_widens_dates(value, end_of_day, expected_hour):
    assert as_datetime(value, end_of_day=end_of_day).hour == expected_hour


def test_as_datetime_passes_datetimes_through():
    moment = datetime(2024, 5, 1, 13, 30)
    assert as_datetime(moment, end_of_day=True) == moment


def test_claim_summary_tolerates_rows_without_the_lateral_joins():
    row = {
        "id": 5, "data": "text", "date": None, "language": "en",
        "gold_evidence_status": "accepted", "gold_evidence_reason": None,
        "gold_evidence_updated_at": None, "released_quarter": False,
        "released_longitudinal": True, "is_rectified": False, "variant_id": None,
    }
    summary = claim_summary(row)
    assert summary["released"] is True
    assert summary["n_evidence"] == 0
    assert summary["t_f"] is None


def test_weighted_series_sorts_by_value_and_adds_shares():
    rows = [{"value": 1.0, "count": 3}, {"value": -1.0, "count": 1}]
    result = weighted_series(rows)
    assert [entry["value"] for entry in result] == [-1.0, 1.0]
    assert result[1]["share"] == pytest.approx(0.75)


def test_delta_summary_returns_both_a_summary_and_a_histogram():
    result = delta_summary([1.0, 2.0, None, 3.0])
    assert result["summary"]["n"] == 3
    assert result["histogram"]["n"] == 3
