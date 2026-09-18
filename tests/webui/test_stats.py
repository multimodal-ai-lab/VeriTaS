"""Aggregation helpers behind the dashboard numbers."""

import math

import pytest

from webui.stats import (
    STATUS_ORDER,
    count_ticks,
    describe,
    evidence_funnel,
    from_log,
    recoverability,
    series,
    share,
    signed_log_histogram,
    to_log,
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


def test_log_transform_round_trips_and_is_signed():
    for value in (-1000.0, -3.5, -0.25, 0.0, 0.25, 3.5, 1000.0):
        assert from_log(to_log(value)) == pytest.approx(value, abs=1e-9)
    assert to_log(-5) == -to_log(5)
    assert to_log(0.0) == 0.0


def test_log_transform_compresses_the_tail():
    # One decade of days is one unit on the axis, which is the point of it.
    assert to_log(9) == pytest.approx(1.0)
    assert to_log(99) == pytest.approx(2.0)


def test_signed_log_histogram_keeps_every_value():
    values = [-100, -10, -1, -0.5, 0.5, 1, 10, 100]
    result = signed_log_histogram(values, bins_per_side=8)
    assert result["n"] == 8
    assert sum(entry["count"] for entry in result["bins"]) == 8


def test_no_bin_straddles_zero():
    result = signed_log_histogram([-50, -1, 2, 80], bins_per_side=6)
    for entry in result["bins"]:
        assert entry["log_start"] < entry["log_end"]
        # The decisive property: no bin spans the origin.
        assert not (entry["log_start"] < 0 < entry["log_end"])
        assert not (entry["start"] < 0 < entry["end"])
        assert entry["side"] == ("negative" if entry["log_end"] <= 0 else "positive")


def test_zero_is_always_a_bin_edge():
    result = signed_log_histogram([-3, 4], bins_per_side=5)
    edges = {round(entry["log_start"], 12) for entry in result["bins"]}
    edges |= {round(entry["log_end"], 12) for entry in result["bins"]}
    assert 0.0 in edges


def test_negative_and_positive_bins_are_counted_separately():
    result = signed_log_histogram([-5, -5, -5, 7, 7], bins_per_side=4)
    assert result["n_negative"] == 3
    assert result["n_positive"] == 2


def test_an_exact_zero_counts_as_before_the_reference_time():
    # t_e == t_c means "available at the claim", which admissibility treats as
    # before it, so the value belongs to the negative side.
    result = signed_log_histogram([0.0, 0.0, 5.0], bins_per_side=4)
    assert result["n_zero"] == 2
    assert result["n_negative"] == 2
    assert result["n_positive"] == 1


def test_signed_log_histogram_is_symmetric_around_zero():
    result = signed_log_histogram([-10, 10], bins_per_side=6)
    negative = [entry for entry in result["bins"] if entry["side"] == "negative"]
    positive = [entry for entry in result["bins"] if entry["side"] == "positive"]
    assert len(negative) == len(positive) == 6
    assert negative[0]["log_start"] == pytest.approx(-result["limit"])
    assert positive[-1]["log_end"] == pytest.approx(result["limit"])


def test_ticks_cover_the_decades_the_data_spans():
    result = signed_log_histogram([-120, 90], bins_per_side=8)
    values = sorted(tick["value"] for tick in result["ticks"])
    assert values == [-100, -10, -1, 0, 1, 10, 100]
    assert all(abs(tick["log"]) <= result["limit"] + 1e-9 for tick in result["ticks"])


def test_ticks_stay_inside_a_narrow_range():
    result = signed_log_histogram([-2, 3], bins_per_side=4)
    assert sorted(tick["value"] for tick in result["ticks"]) == [-1, 0, 1]


def test_signed_log_histogram_of_only_zeros_still_has_a_span():
    result = signed_log_histogram([0.0, 0.0], bins_per_side=4)
    assert result["limit"] > 0
    assert sum(entry["count"] for entry in result["bins"]) == 2


def test_signed_log_histogram_of_nothing():
    result = signed_log_histogram([None, None])
    assert result["n"] == 0 and result["bins"] == [] and result["ticks"] == []


@pytest.mark.parametrize("max_count", [1, 7, 23, 480, 5123])
def test_count_ticks_start_at_zero_and_cover_the_maximum(max_count):
    ticks = count_ticks(max_count)
    assert ticks[0] == 0
    assert ticks[-1] >= max_count
    steps = {ticks[index + 1] - ticks[index] for index in range(len(ticks) - 1)}
    assert len(steps) == 1  # evenly spaced


def test_count_ticks_of_an_empty_chart():
    assert count_ticks(0) == [0]


# --------------------------------------------------------------------- funnel

def make_funnel(**overrides):
    kwargs = dict(
        n_candidates=100,
        n_unfiltered=5,
        reasons=[{"label": "inaccessible", "count": 20},
                 {"label": "unfaithful", "count": 10},
                 {"label": "after_fact_check", "count": 5}],
        n_admissible=60,
        n_before_claim=40,
    )
    kwargs.update(overrides)
    return evidence_funnel(**kwargs)


def test_funnel_starts_at_the_candidates_and_ends_at_E_c():
    stages = make_funnel()
    assert stages[0]["label"] == "Candidates extracted"
    assert stages[0]["count"] == 100
    assert stages[-1]["count"] == 40
    assert stages[-2]["count"] == 60


def test_funnel_is_monotonically_decreasing():
    counts = [stage["count"] for stage in make_funnel()]
    assert counts == sorted(counts, reverse=True)


def test_funnel_subtracts_each_reason_at_its_own_stage():
    stages = {stage["reason"]: stage for stage in make_funnel()}
    assert stages["inaccessible"]["dropped"] == 20
    assert stages["unfaithful"]["dropped"] == 10
    # Unfiltered items leave together with the `not_filtered` reason.
    assert stages["not_filtered"]["dropped"] == 5


def test_funnel_reports_a_disagreement_with_the_stored_admissible_count():
    stages = make_funnel(n_admissible=50)
    admissible = stages[-2]
    assert admissible["count"] == 50          # the stored flag wins
    assert admissible["discrepancy"] == 10    # but the mismatch is visible


def test_funnel_without_any_evidence():
    stages = evidence_funnel(n_candidates=0, n_unfiltered=0, reasons=[],
                             n_admissible=0, n_before_claim=0)
    assert all(stage["count"] == 0 for stage in stages)
    assert stages[0]["share"] is None


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
        "both": 10, "only_E_c": 2, "only_E_f": 6, "neither": 4,
    }
    assert result["recoverable_from_E_c"] == 12
    assert result["recoverable_from_E_f"] == 16
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
    assert result["rate_E_c"] is None
