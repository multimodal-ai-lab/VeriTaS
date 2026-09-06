"""Aggregation helpers behind the dashboard numbers."""

import math

import pytest

from webui.stats import (
    STATUS_ORDER,
    describe,
    histogram,
    recoverability,
    series,
    share,
)


def test_series_orders_known_labels_first_and_computes_shares():
    rows = [
        {"label": "rejected", "count": 30},
        {"label": "accepted", "count": 70},
    ]
    result = series(rows, order=STATUS_ORDER)
    assert [entry["label"] for entry in result] == ["accepted", "rejected"]
    assert result[0]["share"] == pytest.approx(0.7)


def test_series_sorts_unknown_labels_by_descending_count():
    rows = [{"label": "b", "count": 1}, {"label": "a", "count": 5}]
    assert [entry["label"] for entry in series(rows)] == ["a", "b"]


def test_series_maps_none_labels_to_unknown_and_merges_them():
    rows = [{"label": None, "count": 2}, {"label": None, "count": 3}]
    result = series(rows)
    assert result == [{"label": "unknown", "count": 5, "share": 1.0}]


def test_series_of_nothing_is_empty():
    assert series([]) == []


@pytest.mark.parametrize("numerator,denominator,expected", [
    (1, 4, 0.25),
    (0, 4, 0.0),
    (3, 0, None),
    (3, None, None),
])
def test_share(numerator, denominator, expected):
    assert share(numerator, denominator) == expected


def test_describe_five_number_summary():
    summary = describe([1, 2, 3, 4, 5])
    assert summary["n"] == 5
    assert summary["mean"] == pytest.approx(3.0)
    assert summary["median"] == pytest.approx(3.0)
    assert summary["p25"] == pytest.approx(2.0)
    assert summary["p75"] == pytest.approx(4.0)
    assert summary["min"] == 1
    assert summary["max"] == 5
    # Sample standard deviation of 1..5
    assert summary["std"] == pytest.approx(math.sqrt(2.5))


def test_describe_ignores_none_and_nan():
    assert describe([None, 2.0, float("nan"), 4.0])["n"] == 2


def test_describe_of_nothing():
    summary = describe([None, None])
    assert summary["n"] == 0
    assert summary["mean"] is None


def test_describe_of_a_single_value_has_zero_spread():
    summary = describe([7.0])
    assert summary["std"] == 0.0
    assert summary["median"] == 7.0


def test_histogram_bins_cover_the_range_and_keep_every_value():
    result = histogram([0, 1, 2, 3, 4], bins=5)
    assert result["n"] == 5
    assert sum(entry["count"] for entry in result["bins"]) == 5
    assert result["low"] == 0 and result["high"] == 4


def test_histogram_clamps_outliers_into_the_edge_bins():
    result = histogram([-100, 0, 100], bins=4, low=-1, high=1)
    assert sum(entry["count"] for entry in result["bins"]) == 3
    assert result["bins"][0]["count"] == 1
    assert result["bins"][-1]["count"] == 1


def test_histogram_of_a_constant_series_does_not_divide_by_zero():
    result = histogram([5, 5, 5], bins=4)
    assert sum(entry["count"] for entry in result["bins"]) == 3


def test_histogram_of_nothing():
    assert histogram([None, None]) == {"bins": [], "n": 0, "low": None, "high": None}


def test_recoverability_contingency():
    rows = [
        {"claim_close": True, "fact_check_close": True, "count": 10},
        {"claim_close": False, "fact_check_close": True, "count": 6},
        {"claim_close": True, "fact_check_close": False, "count": 2},
        {"claim_close": False, "fact_check_close": False, "count": 4},
    ]
    result = recoverability(rows)
    assert result["n_paired_claims"] == 22
    assert result["contingency"] == {
        "both": 10, "only_E_claim": 2, "only_E_factcheck": 6, "neither": 4,
    }
    assert result["recoverable_from_E_claim"] == 12
    assert result["recoverable_from_E_factcheck"] == 16
    assert result["gain_from_fact_check_period"] == pytest.approx(4 / 22)


def test_recoverability_skips_rows_with_a_missing_side():
    rows = [
        {"claim_close": None, "fact_check_close": True, "count": 5},
        {"claim_close": True, "fact_check_close": True, "count": 1},
    ]
    result = recoverability(rows)
    assert result["n_paired_claims"] == 1


def test_recoverability_of_nothing():
    result = recoverability([])
    assert result["n_paired_claims"] == 0
    assert result["rate_E_claim"] is None
