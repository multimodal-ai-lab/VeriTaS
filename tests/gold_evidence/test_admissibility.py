"""Admissibility decision rule (Spec §3)."""

from datetime import datetime

import pytest

from tests.gold_evidence.conftest import make_evidence
from veritas.gold_evidence import CONDITION_CLAIM, CONDITION_FACT_CHECK
from veritas.gold_evidence.admissibility import (
    InadmissibilityReason,
    apply_admissibility,
    compute_temporal_bounds,
    determine_inadmissibility,
    in_condition,
    in_window,
    keeps_undated,
    select,
)
from veritas.gold_evidence.models import EvidenceRole, SourceKind

T_C = datetime(2024, 5, 1)
T_F = datetime(2024, 5, 20)


def test_fully_valid_evidence_is_admissible():
    assert determine_inadmissibility(make_evidence()) is None


def test_unfiltered_evidence_is_not_admissible():
    evidence = make_evidence(filtered=False)
    assert determine_inadmissibility(evidence) is InadmissibilityReason.NOT_FILTERED


def test_missing_faithfulness_counts_as_unfiltered():
    evidence = make_evidence(faithfulness=None)
    assert determine_inadmissibility(evidence) is InadmissibilityReason.NOT_FILTERED


@pytest.mark.parametrize("kwargs, expected", [
    (dict(accessible=False), InadmissibilityReason.INACCESSIBLE),
    (dict(faithfulness=0.0), InadmissibilityReason.UNFAITHFUL),
    (dict(faithfulness=-1.0), InadmissibilityReason.UNFAITHFUL),
    (dict(before_fact_check=False), InadmissibilityReason.AFTER_FACT_CHECK),
    (dict(professional_fact_check=True, concurrent_fact_check=True),
     InadmissibilityReason.VERDICT_LEAK),
    (dict(later_event=True), InadmissibilityReason.LATER_EVENT),
    (dict(confidence=0.1), None),  # No minimum confidence by default
])
def test_each_criterion_rejects(kwargs, expected):
    evidence = make_evidence(available_since=T_C, **kwargs)
    assert determine_inadmissibility(evidence) is expected


def test_professional_fact_check_on_another_claim_is_kept():
    """Only a *concurrent* fact-check leaks the verdict."""
    evidence = make_evidence(available_since=T_C, professional_fact_check=True,
                             concurrent_fact_check=False)
    assert determine_inadmissibility(evidence) is None


def test_reason_order_is_deterministic():
    """An item violating several criteria reports the first one in REASON_ORDER."""
    evidence = make_evidence(accessible=False, faithfulness=-1.0, later_event=True,
                             available_since=T_C)
    assert determine_inadmissibility(evidence) is InadmissibilityReason.INACCESSIBLE


def test_faithfulness_threshold_boundary():
    at_threshold = make_evidence(available_since=T_C, faithfulness=1 / 3)
    below = make_evidence(available_since=T_C, faithfulness=1 / 3 - 1e-9)
    assert determine_inadmissibility(at_threshold, faithfulness_threshold=1 / 3) is None
    assert determine_inadmissibility(below, faithfulness_threshold=1 / 3) \
           is InadmissibilityReason.UNFAITHFUL


def test_min_extraction_confidence_is_applied_when_configured():
    evidence = make_evidence(available_since=T_C, confidence=0.2)
    assert determine_inadmissibility(evidence, min_extraction_confidence=0.5) \
           is InadmissibilityReason.LOW_CONFIDENCE


# --- Undated sources -------------------------------------------------------

@pytest.mark.parametrize("policy, kind, expected", [
    ("tool_only", SourceKind.TOOL, True),
    ("tool_only", SourceKind.NEWS_ARTICLE, False),
    ("permissive", SourceKind.NEWS_ARTICLE, True),
    ("permissive", SourceKind.TOOL, True),
    ("strict", SourceKind.TOOL, False),
    ("strict", SourceKind.NEWS_ARTICLE, False),
])
def test_undated_policies(policy, kind, expected):
    assert keeps_undated(kind, policy) is expected


def test_undated_news_source_is_rejected_by_default():
    evidence = make_evidence(available_since=None, kind=SourceKind.NEWS_ARTICLE)
    assert determine_inadmissibility(evidence, undated_policy="tool_only") \
           is InadmissibilityReason.UNDATED


def test_undated_tool_is_kept_by_default():
    evidence = make_evidence(available_since=None, kind=SourceKind.TOOL)
    assert determine_inadmissibility(evidence, undated_policy="tool_only") is None


def test_unknown_undated_policy_raises():
    with pytest.raises(AssertionError):
        keeps_undated(SourceKind.TOOL, "whatever")


# --- Temporal bounds -------------------------------------------------------

@pytest.mark.parametrize("t_e, expected", [
    (datetime(2024, 4, 30), (True, True)),      # before the claim
    (T_C, (True, True)),                        # exactly at t_c -> inclusive
    (datetime(2024, 5, 10), (False, True)),     # inside the studied window
    (T_F, (False, True)),                       # exactly at t_f -> inclusive
    (datetime(2024, 5, 21), (False, False)),    # after the fact-check
    (None, (True, True)),                       # unknown is not a violation here
])
def test_compute_temporal_bounds(t_e, expected):
    assert compute_temporal_bounds(t_e, T_C, T_F) == expected


def test_compute_temporal_bounds_accepts_dates_and_tz_aware_datetimes():
    from datetime import date, timezone

    assert compute_temporal_bounds(date(2024, 4, 30), T_C, T_F) == (True, True)
    aware = datetime(2024, 5, 10, tzinfo=timezone.utc)
    assert compute_temporal_bounds(aware, T_C, T_F) == (False, True)


# --- Condition selection ---------------------------------------------------

def test_e_claim_is_a_subset_of_e_factcheck():
    items = [
        make_evidence(locator="https://a/1", before_claim=True),
        make_evidence(locator="https://a/2", before_claim=False),
        make_evidence(locator="https://a/3", before_claim=False, later_event=True),
    ]
    for item in items:
        apply_admissibility(item, undated_policy="permissive")

    fact_check = select(items, CONDITION_FACT_CHECK)
    claim = select(items, CONDITION_CLAIM)

    assert len(fact_check) == 2  # the later-event item is inadmissible
    assert len(claim) == 1
    assert set(id(e) for e in claim).issubset(set(id(e) for e in fact_check))


def test_inadmissible_items_are_in_no_condition():
    evidence = make_evidence(accessible=False)
    apply_admissibility(evidence)
    assert not in_condition(evidence, CONDITION_CLAIM)
    assert not in_condition(evidence, CONDITION_FACT_CHECK)


def test_unknown_condition_raises():
    evidence = make_evidence(available_since=T_C)
    apply_admissibility(evidence)
    with pytest.raises(ValueError):
        in_condition(evidence, "whenever")


def test_selection_orders_essential_evidence_first():
    items = [
        make_evidence(locator="https://a/1", role=EvidenceRole.BACKGROUND, confidence=0.9,
                      available_since=T_C),
        make_evidence(locator="https://a/2", role=EvidenceRole.ESSENTIAL, confidence=0.5,
                      available_since=T_C),
        make_evidence(locator="https://a/3", role=EvidenceRole.AUXILIARY, confidence=0.8,
                      available_since=T_C),
    ]
    for item in items:
        apply_admissibility(item)
    ordered = select(items, CONDITION_FACT_CHECK)
    assert [e.role for e in ordered] == [EvidenceRole.ESSENTIAL, EvidenceRole.AUXILIARY,
                                         EvidenceRole.BACKGROUND]


# --- The studied interval --------------------------------------------------

def test_in_window_requires_a_known_date():
    assert not in_window(make_evidence(available_since=None, before_claim=False))


def test_in_window_is_exactly_t_c_to_t_f():
    inside = make_evidence(available_since=datetime(2024, 5, 10),
                           before_claim=False, before_fact_check=True)
    before = make_evidence(available_since=datetime(2024, 4, 1),
                           before_claim=True, before_fact_check=True)
    after = make_evidence(available_since=datetime(2024, 6, 1),
                          before_claim=False, before_fact_check=False)
    assert in_window(inside)
    assert not in_window(before)
    assert not in_window(after)


def test_apply_admissibility_sets_both_fields():
    evidence = apply_admissibility(make_evidence(accessible=False))
    assert evidence.admissible is False
    assert evidence.inadmissibility_reason == InadmissibilityReason.INACCESSIBLE.value

    evidence = apply_admissibility(make_evidence(available_since=T_C))
    assert evidence.admissible is True
    assert evidence.inadmissibility_reason is None
