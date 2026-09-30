"""The SQL filter builder: user input must always become a bound parameter."""

from datetime import date, datetime

import pytest

from pathlib import Path

from webui import database
from webui.config import Settings
from webui.queries import (
    IMAGE_REF_SQL,
    MEDIA_FILTERS,
    MEDIA_REF_SQL,
    VIDEO_REF_SQL,
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


def test_the_reason_filter_also_matches_the_legacy_name():
    where, args = build_claim_filters(reason="key_evidence_lost")
    assert "c.gold_evidence_reason = ANY ($1::text[])" in where
    assert args == [["key_evidence_lost", "essential_evidence_lost"]]


def test_other_reasons_are_filtered_as_is():
    where, args = build_claim_filters(reason="insufficient_evidence")
    assert "c.gold_evidence_reason = ANY ($1::text[])" in where
    assert args == [["insufficient_evidence"]]


def test_reason_expr_reads_the_legacy_reason_under_the_current_name():
    from webui.queries import CLAIM_REASON_SQL, reason_expr

    assert reason_expr("c") == ("(CASE c.gold_evidence_reason "
                                "WHEN 'essential_evidence_lost' THEN 'key_evidence_lost' "
                                "ELSE c.gold_evidence_reason END)")
    assert reason_expr("") in CLAIM_REASON_SQL


def test_boolean_filters_use_fixed_fragments():
    where, args = build_claim_filters(released=True, media="none")
    assert "(c.released_quarter OR c.released_longitudinal)" in where
    assert f"c.data !~ '{MEDIA_REF_SQL}'" in where
    assert args == []


@pytest.mark.parametrize("media,expected", [
    ("any", f"c.data ~ '{MEDIA_REF_SQL}'"),
    ("image", f"c.data ~ '{IMAGE_REF_SQL}'"),
    ("video", f"c.data ~ '{VIDEO_REF_SQL}'"),
    ("none", f"c.data !~ '{MEDIA_REF_SQL}'"),
])
def test_each_media_filter_maps_to_its_own_pattern(media, expected):
    where, args = build_claim_filters(media=media)
    assert expected in where
    assert args == []


def test_the_image_and_video_filters_are_distinct():
    images, _ = build_claim_filters(media="image")
    videos, _ = build_claim_filters(media="video")
    assert images != videos


def test_an_unknown_media_filter_is_rejected():
    with pytest.raises(ValueError, match="Unknown media filter"):
        build_claim_filters(media="gif")


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


# ---------------------------------------------- deployment misconfigurations

def test_a_loopback_host_inside_a_container_is_reported(monkeypatch, tmp_path):
    """`localhost` inside a container is the container, so a database on the
    host's loopback can never be reached. A bare "connection refused" does not
    say that, and this deployment gets it wrong easily."""
    monkeypatch.setattr(database, "in_container", lambda: True)
    monkeypatch.setattr(database, "get_settings",
                        lambda: _settings(db_host="localhost"))
    hint = database.connection_hint()
    assert hint and "network_mode: host" in hint and "host.docker.internal" in hint


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost"])
def test_every_loopback_spelling_is_recognised(monkeypatch, host):
    monkeypatch.setattr(database, "in_container", lambda: True)
    monkeypatch.setattr(database, "get_settings", lambda: _settings(db_host=host))
    assert database.connection_hint() is not None


def test_a_real_host_is_not_flagged(monkeypatch):
    monkeypatch.setattr(database, "in_container", lambda: True)
    monkeypatch.setattr(database, "get_settings",
                        lambda: _settings(db_host="db.example.org"))
    assert database.connection_hint() is None


def test_nothing_is_flagged_outside_a_container(monkeypatch):
    monkeypatch.setattr(database, "in_container", lambda: False)
    monkeypatch.setattr(database, "get_settings", lambda: _settings(db_host="localhost"))
    assert database.connection_hint() is None


def _settings(**overrides):
    values = dict(db_name="veritas_db", db_user="postgres", db_password="",
                  db_host="localhost", db_port=5432, ezmm_path=Path("/media"),
                  page_size=25, max_page_size=200, chunk_size=1024)
    values.update(overrides)
    return Settings(**values)
