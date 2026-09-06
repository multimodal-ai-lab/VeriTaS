"""Aggregation for the temporal analysis (Spec §5)."""

from datetime import datetime

import pytest

from tests.gold_evidence.conftest import make_evidence
from veritas.gold_evidence.admissibility import apply_admissibility
from veritas.gold_evidence.analysis import (
    ClaimRecord,
    aggregate,
    build_evidence_record,
    describe,
    mcnemar_exact,
    recoverability,
)

T_C = datetime(2024, 5, 1)
T_F = datetime(2024, 5, 21)


def claim_record(**overrides) -> ClaimRecord:
    base = dict(
        claim_id=1, t_c=T_C, t_f=T_F, status="accepted", reason=None, released=True,
        is_rectified=False, language="en", n_candidates=5, n_admissible=4,
        n_in_window=1, n_undated=0, n_evidence_claim=3, n_evidence_fact_check=4,
        gold_integrity=-1.0, gold_veracity=-1.0, gold_context_coverage=None,
        has_media=False,
    )
    base.update(overrides)
    return ClaimRecord(**base)


def test_fact_check_duration_is_in_days():
    assert claim_record().fact_check_duration == pytest.approx(20.0)
    assert claim_record(t_f=None).fact_check_duration is None


def test_claim_record_serializes_derived_fields():
    data = claim_record().to_dict()
    assert data["fact_check_duration_days"] == pytest.approx(20.0)
    assert data["has_window_evidence"] is True
    assert data["t_c"] == T_C.isoformat()


def test_evidence_record_carries_both_time_differences():
    evidence = make_evidence(available_since=datetime(2024, 5, 11),
                             before_claim=False, before_fact_check=True)
    apply_admissibility(evidence)
    row = build_evidence_record(evidence, claim_id=1, t_c=T_C, t_f=T_F)

    assert row["t_e_minus_t_c_days"] == pytest.approx(10.0)
    assert row["t_e_minus_t_f_days"] == pytest.approx(-10.0)
    assert row["in_window"] is True
    assert row["admissible"] is True
    assert row["source_kind"] == "news_article"


def test_evidence_record_of_an_undated_source():
    evidence = make_evidence(available_since=None, kind=make_evidence().source.kind)
    apply_admissibility(evidence)
    row = build_evidence_record(evidence, claim_id=1, t_c=T_C, t_f=T_F)
    assert row["available_since"] is None
    assert row["t_e_minus_t_c_days"] is None
    assert row["in_window"] is False
    assert row["inadmissibility_reason"] == "undated_source"


# --- Aggregates ------------------------------------------------------------

def evidence_rows() -> list[dict]:
    rows = []
    specs = [
        # (available_since, before_claim, admissible-affecting kwargs)
        (datetime(2024, 4, 1), True, {}),
        (datetime(2024, 4, 20), True, {}),
        (datetime(2024, 5, 10), False, {}),                 # in the window
        (datetime(2024, 5, 15), False, {}),                 # in the window
        (datetime(2024, 6, 1), False, {"before_fact_check": False}),  # rejected
        (datetime(2024, 4, 5), True, {"accessible": False}),          # rejected
    ]
    for i, (t_e, before_claim, kwargs) in enumerate(specs):
        evidence = make_evidence(locator=f"https://example.org/{i}",
                                 available_since=t_e, before_claim=before_claim,
                                 **kwargs)
        apply_admissibility(evidence)
        rows.append(build_evidence_record(evidence, claim_id=1, t_c=T_C, t_f=T_F))
    return rows


def test_aggregate_reports_the_window_quantities():
    rows = evidence_rows()
    records = [claim_record(n_candidates=6, n_admissible=4, n_in_window=2)]
    result = aggregate(records, rows)

    assert result["n_evidence_candidates"] == 6
    assert result["n_evidence_admissible"] == 4
    assert result["n_evidence_in_window"] == 2
    assert result["share_evidence_in_window"] == pytest.approx(0.5)
    assert result["share_claims_with_window_evidence"] == pytest.approx(1.0)
    assert result["share_evidence_rejected"] == pytest.approx(2 / 6)


def test_aggregate_reports_rejection_reasons():
    result = aggregate([claim_record()], evidence_rows())
    assert result["rejection_reasons"] == {"after_fact_check": 1, "inaccessible": 1}


def test_aggregate_reports_categorical_distributions():
    result = aggregate([claim_record()], evidence_rows())
    assert result["source_kind_distribution"] == {"news_article": 4}
    assert result["source_proximity_distribution"] == {"secondary": 4}
    assert result["role_distribution"] == {"essential": 4}
    assert result["modality_composition"]["text_only"] == 4
    assert result["modality_composition"]["share_multimodal"] == pytest.approx(0.0)


def test_aggregate_reports_instance_rejections():
    records = [
        claim_record(claim_id=1, status="accepted", reason=None),
        claim_record(claim_id=2, status="rejected", reason="insufficient_evidence"),
        claim_record(claim_id=3, status="rejected", reason="no_admissible_evidence"),
        claim_record(claim_id=4, status="rejected", reason="insufficient_evidence"),
    ]
    result = aggregate(records, [])
    assert result["instance_statuses"] == {"accepted": 1, "rejected": 3}
    assert result["share_instances_rejected_insufficient"] == pytest.approx(0.5)
    assert result["share_instances_rejected_no_admissible"] == pytest.approx(0.25)


def test_aggregate_on_empty_input_does_not_crash():
    result = aggregate([], [])
    assert result["n_claims"] == 0
    assert result["share_evidence_in_window"] is None
    assert result["distribution_t_e_minus_t_c"] == {"n": 0}


# --- Recoverability --------------------------------------------------------

def test_contingency_counts_the_four_cells():
    records = [
        claim_record(claim_id=1, recoverable_claim=True, recoverable_fact_check=True),
        claim_record(claim_id=2, recoverable_claim=False, recoverable_fact_check=True),
        claim_record(claim_id=3, recoverable_claim=False, recoverable_fact_check=True),
        claim_record(claim_id=4, recoverable_claim=False, recoverable_fact_check=False),
        claim_record(claim_id=5, recoverable_claim=True, recoverable_fact_check=False),
    ]
    result = recoverability(records)
    assert result["contingency"] == {
        "both": 1, "only_E_claim": 1, "only_E_factcheck": 2, "neither": 1}
    assert result["recoverable_from_E_claim"] == 2
    assert result["recoverable_from_E_factcheck"] == 3
    assert result["gain_from_fact_check_period"] == pytest.approx(0.2)


def test_unpaired_claims_are_excluded():
    records = [
        claim_record(claim_id=1, recoverable_claim=True, recoverable_fact_check=True),
        claim_record(claim_id=2, recoverable_claim=None, recoverable_fact_check=True),
        claim_record(claim_id=3, recoverable_claim=True, recoverable_fact_check=None),
    ]
    assert recoverability(records)["n_paired_claims"] == 1


@pytest.mark.parametrize("b, c, expected", [
    (0, 0, None),        # no discordant pairs -> undefined
    (5, 5, 1.0),         # perfectly balanced
    (10, 0, pytest.approx(2 / 1024)),
    (0, 10, pytest.approx(2 / 1024)),  # symmetric
])
def test_mcnemar_exact(b, c, expected):
    assert mcnemar_exact(b, c) == expected


def test_mcnemar_is_capped_at_one():
    assert mcnemar_exact(1, 1) == 1.0


# --- describe() ------------------------------------------------------------

def test_describe_reports_order_statistics():
    result = describe([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result["n"] == 5
    assert result["mean"] == pytest.approx(3.0)
    assert result["median"] == pytest.approx(3.0)
    assert result["q1"] == pytest.approx(2.0)
    assert result["q3"] == pytest.approx(4.0)
    assert result["min"] == 1.0 and result["max"] == 5.0
    assert result["values"] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_describe_handles_degenerate_input():
    assert describe([]) == {"n": 0}
    assert describe([7.0])["std"] == 0.0
