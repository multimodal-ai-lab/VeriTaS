"""Admissibility decision rule (Spec §3).

Every criterion is judged on a *source*: whether it can still be retrieved, when it
became available, whether it still supports the proposition. An evidence item
inherits the outcome — it survives as long as one of its sources does.
"""

from datetime import datetime

import pytest

from tests.gold_evidence.conftest import make_evidence, make_source
from veritas.gold_evidence import CONDITION_CLAIM, CONDITION_FACT_CHECK
from veritas.gold_evidence.admissibility import (
    InadmissibilityReason,
    apply_admissibility,
    apply_admissibility_to_item,
    compute_temporal_bounds,
    determine_inadmissibility,
    in_condition,
    in_window,
    keeps_undated,
    restrict_to_condition,
    select,
    source_in_condition,
)
from veritas.gold_evidence.models import EvidenceRole, SourceKind

T_C = datetime(2024, 5, 1)
T_F = datetime(2024, 5, 20)


def test_a_fully_valid_source_is_admissible():
    assert determine_inadmissibility(make_source()) is None


def test_an_unfiltered_source_is_not_admissible():
    assert determine_inadmissibility(make_source(filtered=False)) \
           is InadmissibilityReason.NOT_FILTERED


def test_missing_faithfulness_counts_as_unfiltered():
    assert determine_inadmissibility(make_source(faithfulness=None)) \
           is InadmissibilityReason.NOT_FILTERED


@pytest.mark.parametrize("kwargs, expected", [
    (dict(accessible=False), InadmissibilityReason.INACCESSIBLE),
    (dict(faithfulness=0.0), InadmissibilityReason.UNFAITHFUL),
    (dict(faithfulness=-1.0), InadmissibilityReason.UNFAITHFUL),
    (dict(before_fact_check=False), InadmissibilityReason.AFTER_FACT_CHECK),
    (dict(kind=SourceKind.FACT_CHECK), InadmissibilityReason.VERDICT_LEAK),
])
def test_each_criterion_rejects(kwargs, expected):
    source = make_source(available_since=T_C, **kwargs)
    assert determine_inadmissibility(source) is expected


def test_a_fact_check_source_leaks_the_verdict_whenever_it_appeared():
    """The registry check records its finding as `SourceKind.FACT_CHECK`, so a
    fact-check is rejected as a leak no matter where it sits on the timeline."""
    early = make_source(available_since=datetime(2024, 4, 1), kind=SourceKind.FACT_CHECK)
    assert determine_inadmissibility(early) is InadmissibilityReason.VERDICT_LEAK


def test_reason_order_is_deterministic():
    """A source violating several criteria reports the first one in REASON_ORDER."""
    source = make_source(accessible=False, faithfulness=-1.0, available_since=T_C)
    assert determine_inadmissibility(source) is InadmissibilityReason.INACCESSIBLE


def test_a_later_event_rules_the_item_out_whatever_its_sources_say():
    """§3.3 (3) is about the proposition, so it is not a source criterion: the
    sources stay admissible, and the item they belong to does not."""
    item = make_evidence(later_event=True)

    assert all(source.admissible for source in item.sources)
    assert item.admissible is False
    assert item.inadmissibility_reason == "later_event"
    assert not in_condition(item, CONDITION_FACT_CHECK)
    assert not in_condition(item, CONDITION_CLAIM)


def test_faithfulness_threshold_boundary():
    at_threshold = make_source(available_since=T_C, faithfulness=1 / 3)
    below = make_source(available_since=T_C, faithfulness=1 / 3 - 1e-9)
    assert determine_inadmissibility(at_threshold, faithfulness_threshold=1 / 3) is None
    assert determine_inadmissibility(below, faithfulness_threshold=1 / 3) \
           is InadmissibilityReason.UNFAITHFUL


def test_min_extraction_confidence_is_applied_when_configured():
    """Confidence belongs to the item; the source is rejected on the item's behalf."""
    source = make_source(available_since=T_C)
    assert determine_inadmissibility(source, extraction_confidence=0.2,
                                     min_extraction_confidence=0.5) \
           is InadmissibilityReason.LOW_CONFIDENCE


def test_a_source_without_a_locator_is_inaccessible_at_once():
    """There is nothing to retrieve, now or ever - Stage 2 never even tries."""
    source = make_source(locator=None, filtered=False)
    assert determine_inadmissibility(source) is InadmissibilityReason.INACCESSIBLE


@pytest.mark.parametrize("kind", [SourceKind.TOOL, SourceKind.OFFLINE])
def test_kinds_that_are_not_publications_may_have_no_locator(kind):
    source = make_source(locator=None, kind=kind, filtered=False)
    assert determine_inadmissibility(source, undated_policy="tool_only") is None


# --- Undated sources -------------------------------------------------------

@pytest.mark.parametrize("policy, kind, expected", [
    ("tool_only", SourceKind.TOOL, True),
    ("tool_only", SourceKind.OFFLINE, True),
    ("tool_only", SourceKind.NEWS_ARTICLE, False),
    ("permissive", SourceKind.NEWS_ARTICLE, True),
    ("permissive", SourceKind.TOOL, True),
    ("strict", SourceKind.TOOL, False),
    ("strict", SourceKind.OFFLINE, False),
    ("strict", SourceKind.NEWS_ARTICLE, False),
])
def test_undated_policies(policy, kind, expected):
    assert keeps_undated(kind, policy) is expected


def test_undated_news_source_is_rejected_by_default():
    source = make_source(available_since=None, kind=SourceKind.NEWS_ARTICLE)
    assert determine_inadmissibility(source, undated_policy="tool_only") \
           is InadmissibilityReason.UNDATED


def test_unknown_undated_policy_raises():
    with pytest.raises(AssertionError):
        keeps_undated(SourceKind.TOOL, "whatever")


# --- Sources that cannot be retrieved --------------------------------------

@pytest.mark.parametrize("kind", [SourceKind.TOOL, SourceKind.OFFLINE])
def test_unretrievable_sources_are_admissible_without_locator_or_date(kind):
    """Neither is a publication: nothing to retrieve, nothing to date, and for
    offline evidence nothing to point at either."""
    source = make_source(kind=kind, locator=None, available_since=None, filtered=False)
    assert source.locator is None
    assert determine_inadmissibility(source, undated_policy="tool_only") is None


@pytest.mark.parametrize("kind", [SourceKind.TOOL, SourceKind.OFFLINE])
def test_the_strict_policy_still_removes_them(kind):
    """`strict` is the sensitivity analysis: it drops everything undated."""
    source = make_source(kind=kind, available_since=None, filtered=False)
    assert determine_inadmissibility(source, undated_policy="strict") \
           is InadmissibilityReason.UNDATED


def test_an_unfiltered_tool_is_in_both_conditions():
    """Without a temporal validation, an undated source satisfies both cutoffs."""
    source = apply_admissibility(make_source(kind=SourceKind.TOOL, available_since=None,
                                             filtered=False))
    assert source_in_condition(source, CONDITION_FACT_CHECK)
    assert source_in_condition(source, CONDITION_CLAIM)


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

def test_e_c_is_a_subset_of_e_f():
    items = [
        make_evidence(locator="https://a/1", before_claim=True),
        make_evidence(locator="https://a/2", before_claim=False),
        make_evidence(locator="https://a/3", before_claim=False, later_event=True),
    ]
    fact_check = select(items, CONDITION_FACT_CHECK)
    claim = select(items, CONDITION_CLAIM)

    assert len(fact_check) == 2  # the later-event item is inadmissible
    assert len(claim) == 1
    assert set(id(e) for e in claim).issubset(set(id(e) for e in fact_check))


def test_inadmissible_items_are_in_no_condition():
    evidence = make_evidence(accessible=False)
    assert not in_condition(evidence, CONDITION_CLAIM)
    assert not in_condition(evidence, CONDITION_FACT_CHECK)


def test_unknown_condition_raises():
    with pytest.raises(ValueError):
        in_condition(make_evidence(available_since=T_C), "whenever")


def test_selection_orders_essential_evidence_first():
    items = [
        make_evidence(proposition="p1", role=EvidenceRole.BACKGROUND, confidence=0.9,
                      available_since=T_C),
        make_evidence(proposition="p2", role=EvidenceRole.ESSENTIAL, confidence=0.5,
                      available_since=T_C),
        make_evidence(proposition="p3", role=EvidenceRole.AUXILIARY, confidence=0.8,
                      available_since=T_C),
    ]
    ordered = select(items, CONDITION_FACT_CHECK)
    assert [e.role for e in ordered] == [EvidenceRole.ESSENTIAL, EvidenceRole.AUXILIARY,
                                         EvidenceRole.BACKGROUND]


def test_a_condition_shows_only_the_sources_it_provides():
    """An item can be in `E_c` through one source while another only became
    available later - naming that later source would hand `E_c` evidence it has not."""
    item = make_evidence(sources=[
        make_source(locator="https://a/1", available_since=datetime(2024, 4, 1),
                    before_claim=True),
        make_source(locator="https://a/2", available_since=datetime(2024, 5, 10),
                    before_claim=False),
    ])
    apply_admissibility_to_item(item)

    [restricted] = restrict_to_condition([item], CONDITION_CLAIM)
    assert [s.locator for s in restricted.sources] == ["https://a/1"]
    # ... while the loose condition keeps both.
    [full] = restrict_to_condition([item], CONDITION_FACT_CHECK)
    assert len(full.sources) == 2
    # The restriction is a view, not a mutation of the stored item.
    assert len(item.sources) == 2


# --- The studied interval --------------------------------------------------

def test_in_window_requires_a_known_date():
    assert not in_window(make_source(available_since=None, before_claim=False))


def test_in_window_is_exactly_t_c_to_t_f():
    inside = make_source(available_since=datetime(2024, 5, 10),
                         before_claim=False, before_fact_check=True)
    before = make_source(available_since=datetime(2024, 4, 1),
                         before_claim=True, before_fact_check=True)
    after = make_source(available_since=datetime(2024, 6, 1),
                        before_claim=False, before_fact_check=False)
    assert in_window(inside)
    assert not in_window(before)
    assert not in_window(after)


def test_apply_admissibility_sets_both_fields():
    source = apply_admissibility(make_source(accessible=False))
    assert source.admissible is False
    assert source.inadmissibility_reason == InadmissibilityReason.INACCESSIBLE.value

    source = apply_admissibility(make_source(available_since=T_C))
    assert source.admissible is True
    assert source.inadmissibility_reason is None
