"""Reasoning is read from each provider's dedicated reasoning field, never parsed
out of the answer text."""

from types import SimpleNamespace

import pytest

from veritas.models.base import Generation, Model, as_generation


# ---------------------------------------------------------------------------
# Model.generate contract
# ---------------------------------------------------------------------------

class StubModel(Model):
    """A model whose `_generate` returns whatever the test puts in `to_return`."""

    def __init__(self, to_return):
        self.to_return = to_return
        self.calls = []
        super().__init__("stub:model")

    async def _generate(self, prompt, response_format=None, **kwargs):
        self.calls.append(kwargs)
        return self.to_return


@pytest.mark.asyncio
async def test_generate_returns_only_the_response_by_default():
    model = StubModel(Generation(content="the answer", reasoning="the thoughts"))
    response = await model.generate("hi")

    assert str(response) == "the answer"
    assert not isinstance(response, tuple)


@pytest.mark.asyncio
async def test_generate_returns_reasoning_on_request():
    model = StubModel(Generation(content="the answer", reasoning="the thoughts"))
    response, reasoning = await model.generate("hi", return_reasoning=True)

    assert str(response) == "the answer"
    assert reasoning == "the thoughts"


@pytest.mark.asyncio
async def test_reasoning_is_none_when_the_provider_reports_none():
    model = StubModel(Generation(content="the answer"))
    response, reasoning = await model.generate("hi", return_reasoning=True)

    assert str(response) == "the answer"
    assert reasoning is None


@pytest.mark.asyncio
async def test_a_bare_return_value_still_works():
    """Providers that do not report reasoning keep working unchanged."""
    model = StubModel("plain text")
    assert str(await model.generate("hi")) == "plain text"

    response, reasoning = await model.generate("hi", return_reasoning=True)
    assert str(response) == "plain text"
    assert reasoning is None


@pytest.mark.asyncio
async def test_empty_content_is_passed_through_with_its_reasoning():
    model = StubModel(Generation(content=None, reasoning="thought, then gave up"))
    response, reasoning = await model.generate("hi", return_reasoning=True)

    assert response is None
    assert reasoning == "thought, then gave up"


@pytest.mark.asyncio
async def test_extraction_applies_to_the_content_not_the_reasoning():
    model = StubModel(Generation(content="blah `Entails` blah", reasoning="`Contradicts`"))
    response, reasoning = await model.generate("hi", extract="last_code_span",
                                               return_reasoning=True)

    assert str(response) == "Entails"
    assert reasoning == "`Contradicts`"


@pytest.mark.asyncio
async def test_response_format_bypasses_text_formatting():
    parsed = SimpleNamespace(value=42)
    model = StubModel(Generation(content=parsed, reasoning="thoughts"))
    response, reasoning = await model.generate("hi", response_format=object,
                                               return_reasoning=True)

    assert response is parsed
    assert reasoning == "thoughts"


def test_as_generation_normalizes():
    generation = Generation(content="a", reasoning="b")
    assert as_generation(generation) is generation
    assert as_generation("a") == Generation(content="a", reasoning=None)
    assert as_generation(None) == Generation(content=None, reasoning=None)


# ---------------------------------------------------------------------------
# OpenAI: reasoning items on the Responses API
# ---------------------------------------------------------------------------

def openai_response(output, **extra):
    return SimpleNamespace(output=output, output_text="the answer", **extra)


def reasoning_item(summary=(), content=()):
    return SimpleNamespace(
        type="reasoning",
        summary=[SimpleNamespace(text=t) for t in summary],
        content=[SimpleNamespace(text=t) for t in content],
    )


def message_item(text="the answer"):
    return SimpleNamespace(type="message", content=[SimpleNamespace(text=text)])


def test_openai_reads_reasoning_summaries():
    from veritas.models.gpt import GPT

    response = openai_response([reasoning_item(summary=["step one", "step two"]),
                                message_item()])
    assert GPT._extract_reasoning(response) == "step one\n\nstep two"


def test_openai_ignores_the_answer_message():
    from veritas.models.gpt import GPT

    assert GPT._extract_reasoning(openai_response([message_item("no reasoning here")])) is None


def test_openai_reads_plain_reasoning_content():
    """OpenAI-compatible servers put the trace in `content` rather than `summary`."""
    from veritas.models.gpt import GPT

    response = openai_response([reasoning_item(content=["local model thoughts"])])
    assert GPT._extract_reasoning(response) == "local model thoughts"


def test_openai_falls_back_to_reasoning_content_on_the_response():
    from veritas.models.gpt import GPT

    response = openai_response([message_item()], reasoning_content="vllm style")
    assert GPT._extract_reasoning(response) == "vllm style"


def test_openai_handles_a_response_without_output():
    from veritas.models.gpt import GPT

    assert GPT._extract_reasoning(SimpleNamespace(output=None)) is None
    assert GPT._extract_reasoning(SimpleNamespace()) is None


# ---------------------------------------------------------------------------
# Anthropic: thinking blocks
# ---------------------------------------------------------------------------

def anthropic_message(blocks):
    return SimpleNamespace(content=blocks)


def test_anthropic_reads_thinking_blocks():
    from veritas.models.claude import Claude

    message = anthropic_message([
        SimpleNamespace(type="thinking", thinking="first thought"),
        SimpleNamespace(type="thinking", thinking="second thought"),
        SimpleNamespace(type="text", text="the answer"),
    ])
    assert Claude._extract_reasoning(message) == "first thought\n\nsecond thought"


def test_anthropic_skips_redacted_thinking():
    from veritas.models.claude import Claude

    message = anthropic_message([
        SimpleNamespace(type="redacted_thinking", data="encrypted"),
        SimpleNamespace(type="text", text="the answer"),
    ])
    assert Claude._extract_reasoning(message) is None


def test_anthropic_without_thinking_reports_none():
    from veritas.models.claude import Claude

    message = anthropic_message([SimpleNamespace(type="text", text="the answer")])
    assert Claude._extract_reasoning(message) is None


def test_anthropic_thinking_budget_scales_with_effort():
    from veritas.models.claude import Claude

    budgets = Claude.THINKING_BUDGETS
    assert budgets["low"] < budgets["medium"] < budgets["high"]
    assert budgets["none"] == 0


# ---------------------------------------------------------------------------
# Gemini: thought parts
# ---------------------------------------------------------------------------

def gemini_response(parts, text="joined"):
    content = SimpleNamespace(parts=parts)
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)], text=text)


def part(text, thought=False):
    return SimpleNamespace(text=text, thought=thought)


def test_gemini_reads_thought_parts():
    from veritas.models.gemini import Gemini

    response = gemini_response([part("thinking hard", thought=True),
                                part("the answer")])
    assert Gemini._extract_reasoning(response) == "thinking hard"


def test_gemini_answer_excludes_thought_parts():
    """`response.text` concatenates everything, so thoughts must be filtered out
    or they would corrupt the downstream parsing of the answer."""
    from veritas.models.gemini import Gemini

    response = gemini_response([part("thinking hard", thought=True),
                                part("the answer")],
                               text="thinking hardthe answer")
    assert Gemini._extract_text(response) == "the answer"


def test_gemini_without_thoughts_reports_none_and_keeps_the_text():
    from veritas.models.gemini import Gemini

    response = gemini_response([part("the answer")])
    assert Gemini._extract_reasoning(response) is None
    assert Gemini._extract_text(response) == "the answer"


def test_gemini_falls_back_to_response_text_without_parts():
    from veritas.models.gemini import Gemini

    response = SimpleNamespace(candidates=None, text="the answer")
    assert Gemini._extract_text(response) == "the answer"
    assert Gemini._extract_reasoning(response) is None
