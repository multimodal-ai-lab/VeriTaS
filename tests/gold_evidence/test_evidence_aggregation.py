"""Evidence as the aggregation layer over its sources.

Fact-checks cite redundantly: two outlets for one fact, a register plus a screenshot
of it. Those are several *sources* of one evidence item, so losing one of them costs
the reconstruction nothing. An item is lost only when it has lost every source, and
only losing an *essential* item disqualifies the instance.
"""

import pytest

from tests.gold_evidence.conftest import make_evidence, make_source
from veritas.gold_evidence import CONDITION_CLAIM, CONDITION_FACT_CHECK
from veritas.gold_evidence.admissibility import (
    essential,
    lost_essential,
    missing_essential,
)
from veritas.gold_evidence.models import Evidence, EvidenceRole


def two_sources(first: bool, second: bool, **kwargs) -> Evidence:
    """An item with two sources, each accessible or not as given."""
    return make_evidence(sources=[
        make_source(locator="https://a/1", accessible=first, **kwargs),
        make_source(locator="https://a/2", accessible=second, **kwargs),
    ])


# --- Admissibility of the item ---------------------------------------------

def test_one_surviving_source_keeps_the_item():
    item = two_sources(False, True)
    assert item.admissible is True
    assert len(item.admissible_sources) == 1


def test_an_item_is_lost_only_when_every_source_is():
    item = two_sources(False, False)
    assert item.admissible is False
    assert item.inadmissibility_reason == "inaccessible"


def test_an_item_is_undecided_while_a_source_is_unfiltered():
    """Not decided until its last source is: the pending one may yet save it."""
    item = make_evidence(sources=[make_source(accessible=False),
                                  make_source(locator="https://a/2", filtered=False)],
                         decided=False)
    from veritas.gold_evidence.admissibility import apply_admissibility

    apply_admissibility(item.sources[0], extraction_confidence=item.extraction_confidence)
    assert item.admissible is None
    assert item.filtered is False


def test_an_item_without_sources_is_not_admissible():
    assert Evidence(claim_id=1, proposition="p").admissible is False


def test_the_item_time_is_the_earliest_of_its_sources():
    """t_e of the proposition: when it could first be read anywhere."""
    from datetime import datetime

    item = make_evidence(sources=[
        make_source(locator="https://a/1", available_since=datetime(2024, 5, 10)),
        make_source(locator="https://a/2", available_since=datetime(2024, 4, 1)),
    ])
    assert item.available_since == datetime(2024, 4, 1)


# --- Conditions ------------------------------------------------------------

def test_an_item_is_in_a_condition_through_any_of_its_sources():
    """One pre-claim source puts the proposition into `E_c`, even if the other
    source only appeared during the fact-checking period."""
    from veritas.gold_evidence.admissibility import in_condition

    item = make_evidence(sources=[
        make_source(locator="https://a/1", before_claim=False),
        make_source(locator="https://a/2", before_claim=True),
    ])
    assert in_condition(item, CONDITION_CLAIM)
    assert in_condition(item, CONDITION_FACT_CHECK)


# --- Essential evidence ----------------------------------------------------

def test_only_items_the_rationale_rests_on_are_essential():
    items = [make_evidence(proposition="a", role=EvidenceRole.ESSENTIAL),
             make_evidence(proposition="b", role=EvidenceRole.AUXILIARY),
             make_evidence(proposition="c", role=EvidenceRole.BACKGROUND)]
    assert len(essential(items)) == 1


def test_redundancy_keeps_an_essential_item_alive():
    """The central case: one source is gone, the proposition is not."""
    assert lost_essential([two_sources(False, True)]) == []


def test_an_essential_item_that_lost_every_source_is_lost():
    lost = lost_essential([two_sources(False, False), make_evidence(proposition="other")])
    assert len(lost) == 1
    assert not lost[0].admissible


@pytest.mark.parametrize("role", [EvidenceRole.AUXILIARY, EvidenceRole.BACKGROUND])
def test_losing_non_essential_evidence_never_disqualifies(role):
    items = [two_sources(False, False, ), make_evidence(proposition="other")]
    items[0].role = role
    assert lost_essential(items) == []


def test_an_empty_evidence_set_loses_nothing():
    """A verdict resting on the rationale alone has no essential item to lose."""
    assert lost_essential([]) == []
    assert essential([]) == []


# --- Essential evidence a condition cannot supply --------------------------

def test_a_condition_without_an_essential_item_is_reported():
    """`E_c` cannot support a rationale whose premise only appeared later - which
    is the finding, not a defect, so the ensemble is never asked."""
    item = make_evidence(before_claim=False)
    assert missing_essential([item], CONDITION_CLAIM) == [item]
    assert missing_essential([item], CONDITION_FACT_CHECK) == []


def test_nothing_is_missing_when_every_essential_item_is_present():
    items = [make_evidence(proposition="a"), make_evidence(proposition="b")]
    assert missing_essential(items, CONDITION_CLAIM) == []


def test_auxiliary_evidence_outside_a_condition_is_not_missed():
    item = make_evidence(before_claim=False, role=EvidenceRole.AUXILIARY)
    assert missing_essential([item], CONDITION_CLAIM) == []


# --- Later events ----------------------------------------------------------

def test_a_later_event_survives_no_amount_of_redundancy():
    """Every source may be intact; the proposition still rests on something that
    only became true after the claim."""
    item = two_sources(True, True)
    from veritas.gold_evidence.models import LaterEventCheck

    item.later_event = LaterEventCheck(change_detected=True)

    assert item.relies_on_later_event is True
    assert item.admissible is False
    assert lost_essential([item]) == [item]


def test_a_cleared_later_event_check_changes_nothing():
    item = two_sources(True, True)
    from veritas.gold_evidence.models import LaterEventCheck

    item.later_event = LaterEventCheck(change_detected=False, justification="Reporting only.")
    assert item.admissible is True
