"""Tools and offline evidence in Stage 2.

They are not publications: there is nothing to retrieve and nothing to re-read, and
they need carry neither a locator nor a publication time. But once a `t_e` *is*
known, they sit on the timeline like any other evidence and are validated against
the cutoffs and for later events.
"""

from datetime import datetime

import pytest

from tests.gold_evidence.conftest import make_evidence
from veritas.common import Claim
from veritas.gold_evidence import filtering as filtering_module
from veritas.gold_evidence.admissibility import in_condition
from veritas.gold_evidence.models import SourceKind
from veritas.gold_evidence import CONDITION_CLAIM, CONDITION_FACT_CHECK

T_C = datetime(2024, 5, 1)
T_F = datetime(2024, 5, 21)

CLAIM = Claim(id=1, data="A claim", date=T_C, appearance_ids=set(), review_ids={1})


class StubModel:
    """A filtering model that always answers the temporal question the same way."""

    specifier = "stub:model"

    def __init__(self, later_event: bool = False):
        self.later_event = later_event
        self.prompts: list[str] = []

    async def generate(self, prompt, **kwargs):
        self.prompts.append(str(prompt))
        answer = ('```json\n'
                  f'{{"later_event": {str(self.later_event).lower()}, '
                  f'"justification": "Because."}}\n```')
        return answer, "provider trace"


@pytest.fixture
def stage_2(monkeypatch):
    """Records whether retrieval was attempted and what the model was asked."""
    state = {"retrievals": [], "model": StubModel()}

    async def fake_retrieve(locator, session=None, determine_time=True):
        state["retrievals"].append(locator)
        raise AssertionError("An unretrievable source must not be retrieved.")

    monkeypatch.setattr(filtering_module, "retrieve_source", fake_retrieve)
    monkeypatch.setattr(filtering_module, "_resolve_filtering_model",
                        lambda: state["model"])
    return state


async def filter_item(**kwargs):
    evidence = make_evidence(filtered=False, **kwargs)
    return await filtering_module.filter_single(evidence, claim=CLAIM, t_c=T_C, t_f=T_F)


# --- Without a publication time --------------------------------------------

@pytest.mark.parametrize("kind", [SourceKind.TOOL, SourceKind.OFFLINE])
@pytest.mark.asyncio
async def test_an_undated_source_is_admitted_untouched(stage_2, kind):
    evidence = await filter_item(kind=kind, locator=None, available_since=None)

    assert stage_2["retrievals"] == []
    assert stage_2["model"].prompts == []
    assert evidence.temporal_validation is None
    assert evidence.admissible is True
    assert in_condition(evidence, CONDITION_CLAIM)
    assert in_condition(evidence, CONDITION_FACT_CHECK)


# --- With a publication time -----------------------------------------------

@pytest.mark.parametrize("kind", [SourceKind.TOOL, SourceKind.OFFLINE])
@pytest.mark.asyncio
async def test_a_dated_source_is_validated_against_both_cutoffs(stage_2, kind):
    """Available before the claim: on the timeline, and no model call needed."""
    evidence = await filter_item(kind=kind, locator=None,
                                 available_since=datetime(2024, 4, 15))

    assert evidence.temporal_validation is not None
    assert evidence.temporal_validation.before_claim is True
    assert evidence.temporal_validation.before_fact_check is True
    assert stage_2["model"].prompts == []  # pre-claim evidence cannot report later events
    assert evidence.admissible is True
    assert in_condition(evidence, CONDITION_CLAIM)


@pytest.mark.parametrize("kind", [SourceKind.TOOL, SourceKind.OFFLINE])
@pytest.mark.asyncio
async def test_a_source_dated_after_the_fact_check_is_inadmissible(stage_2, kind):
    evidence = await filter_item(kind=kind, locator=None,
                                 available_since=datetime(2024, 6, 1))

    assert evidence.admissible is False
    assert evidence.inadmissibility_reason == "after_fact_check"
    assert stage_2["model"].prompts == []


@pytest.mark.asyncio
async def test_an_interview_inside_the_window_is_asked_about_later_events(stage_2):
    """An interview conducted during the fact-checking period is exactly the case
    the later-event check exists for."""
    evidence = await filter_item(kind=SourceKind.OFFLINE, locator=None,
                                 available_since=datetime(2024, 5, 10))

    assert evidence.temporal_validation.before_claim is False
    assert len(stage_2["model"].prompts) == 1
    assert evidence.admissible is True
    assert in_condition(evidence, CONDITION_FACT_CHECK)
    assert not in_condition(evidence, CONDITION_CLAIM)


@pytest.mark.asyncio
async def test_a_later_event_makes_it_inadmissible(stage_2):
    stage_2["model"] = StubModel(later_event=True)
    evidence = await filter_item(kind=SourceKind.OFFLINE, locator=None,
                                 available_since=datetime(2024, 5, 10))

    assert evidence.temporal_validation.later_event is True
    assert evidence.admissible is False
    assert evidence.inadmissibility_reason == "later_event"


@pytest.mark.asyncio
async def test_the_prompt_states_that_the_source_cannot_be_retrieved(stage_2):
    """There is no source content to show, so the model is told what it is judging."""
    await filter_item(kind=SourceKind.OFFLINE, locator=None,
                      available_since=datetime(2024, 5, 10))

    prompt = stage_2["model"].prompts[0]
    assert "cannot be retrieved" in prompt
    assert "The mayor signed the decree on 3 May." in prompt  # the proposition
