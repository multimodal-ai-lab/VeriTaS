"""Test doubles for the fact-checking baseline: a provider that replays scripted model steps,
and fakes for scrapeMM's search and retrieve."""

import pytest
from ezmm import MultimodalSequence
from ezmm.common import registry
from scrapemm.common import ScrapedContent, ScrapingResponse
from scrapemm.search import SerperOrganicResult, SerperResponse

from eval.baselines.common import tools
from eval.baselines.providers.base import BaseFactChecker, only_text


class ScriptedFactChecker(BaseFactChecker):
    """Replays scripted model steps (the last one repeats) and records what the loop
    passes to the model."""

    provider_name = "scripted"
    default_model = "scripted-model"

    def __init__(self, steps, native_video=False, **kwargs):
        self.steps = list(steps)
        self.native_video = native_video
        self.allow_tools = []  # Per model call
        self.conversation = None
        super().__init__(**kwargs)

    def _create_client(self, api_key):
        return None

    def _start_conversation(self, prompt):
        self.conversation = [prompt]
        return self.conversation

    def _step(self, conversation, allow_tools):
        self.allow_tools.append(allow_tools)
        step = self.steps.pop(0) if len(self.steps) > 1 else self.steps[0]
        conversation.append(step)
        return step

    def _add_tool_results(self, conversation, tool_calls, results):
        conversation.append([(call.name, only_text(result) or result) for call, result in zip(tool_calls, results)])

    def _add_text(self, conversation, text):
        conversation.append(text)


@pytest.fixture
def ezmm_registry(tmp_path, monkeypatch):
    """An ezMM registry of the test's own, so that its media don't end up in the repository's."""
    monkeypatch.setattr(registry, "item_registry", registry.ItemRegistry(tmp_path / "ezmm"))


def fake_search(monkeypatch, urls: list[str]):
    """Make web_search return organic results with the given URLs. Returns the queries."""
    queries = []

    async def search(query):
        queries.append(query)
        return SerperResponse(organic=[SerperOrganicResult(title=f"Result {i}", link=url)
                                       for i, url in enumerate(urls, 1)])

    monkeypatch.setattr(tools, "search", search)
    return queries


def page(*parts) -> ScrapingResponse:
    """What scrapeMM returns for a page with the multimodal content `parts` (text and ezMM items)."""
    return ScrapingResponse(url="", content=ScrapedContent(multimodal=MultimodalSequence(*parts)),
                            output_format="multimodal")


def fake_retrieve(monkeypatch, text_for):
    """Make fetch_url return a page with the text `text_for(url)`. Returns the fetched URLs."""
    fetched = []

    async def retrieve(url, **kwargs):
        fetched.append(url)
        return page(text_for(url))

    monkeypatch.setattr(tools, "retrieve", retrieve)
    return fetched
