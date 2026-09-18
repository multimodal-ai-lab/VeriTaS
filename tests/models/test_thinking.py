"""How `reasoning_effort` reaches Anthropic.

From Claude 4.6 on, the model decides itself how much to think and the depth is
steered by an effort level; the explicit `budget_tokens` of the older models is
rejected with a 400 there. `Claude._generate` picks the right one per model.
"""

from types import SimpleNamespace

import pytest

from veritas.models.claude import Claude, uses_adaptive_thinking

ADAPTIVE = ["claude-opus-5", "claude-sonnet-5", "claude-opus-4-8", "claude-opus-4-7",
            "claude-opus-4-6", "claude-sonnet-4-6", "claude-fable-5-1"]
BUDGETED = ["claude-opus-4-5", "claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001",
            "claude-sonnet-4-20250514", "claude-opus-4-1-20250805",
            "claude-3-5-sonnet-20240620"]


@pytest.mark.parametrize("specifier", ADAPTIVE)
def test_current_models_think_adaptively(specifier):
    assert uses_adaptive_thinking(specifier)


@pytest.mark.parametrize("specifier", BUDGETED)
def test_older_models_take_an_explicit_budget(specifier):
    """The date suffix of e.g. 'claude-sonnet-4-20250514' is not a minor version."""
    assert not uses_adaptive_thinking(specifier)


def test_an_unknown_specifier_is_assumed_to_be_a_new_model():
    assert uses_adaptive_thinking("claude-something-new")


def make_claude(specifier: str, recorded: list) -> Claude:
    """A Claude whose client records the request instead of sending it."""
    model = object.__new__(Claude)  # no API key, no network
    model.specifier = specifier

    async def create(**kwargs):
        recorded.append(kwargs)
        return None  # not a Message, so `_generate` returns None

    model.client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return model


@pytest.mark.asyncio
async def test_effort_is_sent_as_adaptive_thinking():
    recorded = []
    await make_claude("claude-opus-5", recorded)._generate("hi", reasoning_effort="high")
    request = recorded[0]

    assert request["thinking"]["type"] == "adaptive"
    assert "budget_tokens" not in request["thinking"]
    assert request["output_config"] == dict(effort="high")


@pytest.mark.asyncio
async def test_thinking_blocks_are_requested_readable():
    """Their text is empty by default, and the reasoning traces are read off them."""
    recorded = []
    await make_claude("claude-opus-5", recorded)._generate("hi", reasoning_effort="low")
    assert recorded[0]["thinking"]["display"] == "summarized"


@pytest.mark.asyncio
async def test_thinking_tokens_fit_into_max_tokens():
    recorded = []
    await make_claude("claude-opus-5", recorded)._generate("hi", reasoning_effort="high",
                                                           max_tokens=2048)
    assert recorded[0]["max_tokens"] > 2048


@pytest.mark.asyncio
async def test_effort_none_switches_thinking_off():
    recorded = []
    await make_claude("claude-opus-5", recorded)._generate("hi", reasoning_effort="none")

    assert recorded[0]["thinking"] == dict(type="disabled")
    assert "output_config" not in recorded[0]


@pytest.mark.asyncio
async def test_older_models_still_get_a_budget():
    recorded = []
    await make_claude("claude-sonnet-4-5-20250929", recorded)._generate(
        "hi", reasoning_effort="high")
    request = recorded[0]

    assert request["thinking"]["type"] == "enabled"
    assert request["max_tokens"] > request["thinking"]["budget_tokens"]
    assert "output_config" not in request


@pytest.mark.asyncio
async def test_no_effort_leaves_the_parameters_out():
    recorded = []
    await make_claude("claude-opus-5", recorded)._generate("hi")

    assert "thinking" not in recorded[0]
    assert "output_config" not in recorded[0]
