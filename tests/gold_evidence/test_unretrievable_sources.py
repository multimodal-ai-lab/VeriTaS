"""Tools and offline evidence in Stage 2.

They are not publications: there is nothing to retrieve and nothing to re-read, and
they need carry neither a locator nor a publication time. But once a `t_e` *is*
known (`Citation.date_as_cited`), they sit on the timeline like any other evidence
and are validated against the cutoffs and for later events.
"""

from datetime import datetime

import pytest

from tests.gold_evidence.conftest import make_citation, make_evidence
from veritas.common import Claim
from veritas.gold_evidence import filtering as filtering_module
from veritas.gold_evidence.admissibility import citation_in_condition
from veritas.gold_evidence.models import SourceKind
from veritas.gold_evidence import CONDITION_CLAIM, CONDITION_FACT_CHECK

T_C = datetime(2024, 5, 1)
T_F = datetime(2024, 5, 21)

CLAIM = Claim(id=1, data="A claim", date=T_C, appearance_ids=set(), review_ids={1})


class StubModel:
    """A filtering model that always answers the later-event question the same way."""

    specifier = "stub:model"

    def __init__(self, change_detected: bool = False):
        self.change_detected = change_detected
        self.prompts: list[str] = []

    async def generate(self, prompt, **kwargs):
        self.prompts.append(str(prompt))
        answer = ('```json\n'
                  f'{{"change_detected": {str(self.change_detected).lower()}, '
                  f'"justification": "Because."}}\n```')
        return answer, "provider trace"


@pytest.fixture
def stage_2(monkeypatch):
    """Records whether retrieval was attempted and what the model was asked."""
    state = {"retrievals": [], "model": StubModel()}

    async def fake_retrieve(locator, determine_time=True):
        state["retrievals"].append(locator)
        raise AssertionError("An unretrievable source must not be retrieved.")

    monkeypatch.setattr(filtering_module, "retrieve_source", fake_retrieve)
    monkeypatch.setattr(filtering_module, "_resolve_filtering_model",
                        lambda: state["model"])
    return state


async def filter_item(**kwargs):
    """Judges the item's single citation and returns it. Tools, offline evidence
    and unlocated sources have no source to settle, so this is all of Stage 2 for
    them."""
    evidence = make_evidence(decided=False, **kwargs)
    return await filtering_module.judge_citation(evidence.citations[0], evidence=evidence,
                                                 t_c=T_C, t_f=T_F)


async def check_item(**kwargs):
    """Runs the item-level later-event check and returns the item."""
    evidence = make_evidence(**kwargs)
    await filtering_module.check_later_event(evidence, claim=CLAIM)
    return evidence


# --- Without a publication time --------------------------------------------

@pytest.mark.parametrize("kind", [SourceKind.TOOL, SourceKind.OFFLINE])
@pytest.mark.asyncio
async def test_an_undated_citation_is_admitted_untouched(stage_2, kind):
    citation = await filter_item(kind=kind, locator=None, available_since=None)

    assert stage_2["retrievals"] == []
    assert stage_2["model"].prompts == []
    assert citation.temporal_validation is None
    assert citation.admissible is True
    assert citation_in_condition(citation, CONDITION_CLAIM)
    assert citation_in_condition(citation, CONDITION_FACT_CHECK)


# --- With a publication time -----------------------------------------------

@pytest.mark.parametrize("kind", [SourceKind.TOOL, SourceKind.OFFLINE])
@pytest.mark.asyncio
async def test_a_dated_source_is_validated_against_both_cutoffs(stage_2, kind):
    """Available before the claim: on the timeline, and no model call needed."""
    citation = await filter_item(kind=kind, locator=None,
                                 available_since=datetime(2024, 4, 15))

    assert citation.temporal_validation is not None
    assert citation.temporal_validation.before_claim is True
    assert citation.temporal_validation.before_fact_check is True
    assert stage_2["model"].prompts == []  # the cutoffs are computed, not predicted
    assert citation.admissible is True
    assert citation_in_condition(citation, CONDITION_CLAIM)


@pytest.mark.parametrize("kind", [SourceKind.TOOL, SourceKind.OFFLINE])
@pytest.mark.asyncio
async def test_a_source_dated_after_the_fact_check_is_inadmissible(stage_2, kind):
    citation = await filter_item(kind=kind, locator=None,
                                 available_since=datetime(2024, 6, 1))

    assert citation.admissible is False
    assert citation.inadmissibility_reason == "after_fact_check"
    assert stage_2["model"].prompts == []


@pytest.mark.asyncio
async def test_an_interview_inside_the_window_is_asked_about_later_events(stage_2):
    """An interview conducted during the fact-checking period is exactly the case
    the later-event check exists for. It is asked once for the item."""
    item = await check_item(kind=SourceKind.OFFLINE, locator=None,
                            available_since=datetime(2024, 5, 10), before_claim=False)

    assert len(stage_2["model"].prompts) == 1
    assert item.later_event is not None
    assert item.later_event.change_detected is False
    assert item.admissible is True


@pytest.mark.asyncio
async def test_a_later_event_makes_the_item_inadmissible(stage_2):
    stage_2["model"] = StubModel(change_detected=True)
    item = await check_item(kind=SourceKind.OFFLINE, locator=None,
                            available_since=datetime(2024, 5, 10), before_claim=False)

    assert item.later_event.change_detected is True
    assert item.admissible is False
    assert item.inadmissibility_reason == "later_event"


@pytest.mark.asyncio
async def test_the_check_runs_once_however_many_sources_report_the_proposition(stage_2):
    """It is the proposition that rests on a later event, not one place it is read."""
    item = await check_item(citations=[
        make_citation(locator="https://a/1", available_since=datetime(2024, 5, 10),
                      before_claim=False),
        make_citation(locator="https://a/2", available_since=datetime(2024, 5, 12),
                      before_claim=False),
    ])
    assert len(stage_2["model"].prompts) == 1
    # Both sources are named in that one prompt.
    assert "https://a/1" in stage_2["model"].prompts[0]
    assert "https://a/2" in stage_2["model"].prompts[0]


@pytest.mark.asyncio
async def test_the_prompt_states_when_a_source_could_not_be_retrieved(stage_2):
    """There is no source content to show, so the model is told what it is judging."""
    await check_item(kind=SourceKind.OFFLINE, locator=None,
                     available_since=datetime(2024, 5, 10), before_claim=False)

    prompt = stage_2["model"].prompts[0]
    assert "could not be retrieved" in prompt
    assert "The mayor signed the decree on 3 May." in prompt  # the proposition


@pytest.mark.parametrize("t_e, expected", [
    (datetime(2024, 4, 15), False),   # readable before the claim: nothing to ask
    (datetime(2024, 5, 10), True),    # appeared afterwards: worth asking
    (None, False),                    # undated: no timeline to place it on
])
def test_the_check_runs_only_for_evidence_that_appeared_after_the_claim(t_e, expected):
    item = make_evidence(available_since=t_e, before_claim=t_e is None or t_e <= T_C)
    assert filtering_module.needs_later_event_check(item, T_C) is expected


def test_an_item_that_lost_every_source_is_not_worth_the_call():
    item = make_evidence(accessible=False, available_since=datetime(2024, 5, 10),
                         before_claim=False)
    assert item.admissible is False
    assert filtering_module.needs_later_event_check(item, T_C) is False


# --- Sources the article never located --------------------------------------

@pytest.mark.asyncio
async def test_an_unlocated_source_is_discarded_without_a_retrieval(stage_2):
    """A news article the fact-check cites but never links: nothing to retrieve."""
    citation = await filter_item(kind=SourceKind.NEWS_ARTICLE, locator=None)

    assert stage_2["retrievals"] == []
    assert citation.source is None
    assert citation.admissible is False
    assert citation.inadmissibility_reason == "locator_missing"
