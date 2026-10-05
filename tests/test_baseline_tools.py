"""Tests for the baseline's web_search and fetch_url tools, with scrapeMM mocked."""

from datetime import date

import pytest
from scrapemm import QuotaExceededError, RateLimitError, TargetUnavailableError
from scrapemm.common import ScrapedContent, ScrapingResponse
from scrapemm.search import SerperAnswerBox, SerperKnowledgeGraph, SerperOrganicResult, SerperResponse

from eval.baselines.common import tools
from eval.baselines.common.prompts import SYSTEM_PROMPT
from eval.baselines.common.tools import MAX_FETCH_CHARS, SearchQuotaExceededError, ToolSession
from eval.baselines.common.types import LABELS_7


def _serper_response() -> SerperResponse:
    return SerperResponse(
        organic=[
            SerperOrganicResult(title="Old article", link="https://example.com/old",
                                snippet="Before the claim.", date="Jan 3, 2024"),
            SerperOrganicResult(title="No link", link=None),
        ],
        # Not date-filtered by Google, must never reach the model
        knowledge_graph=SerperKnowledgeGraph(title="Leaky knowledge graph"),
        answer_box=SerperAnswerBox(answer="Leaky answer box", link="https://example.com/answer"),
    )


@pytest.fixture
def searches(monkeypatch):
    """Replace scrapeMM's search with a fake that records the queries."""
    queries = []

    async def fake_search(query):
        queries.append(query)
        return _serper_response()

    monkeypatch.setattr(tools, "search", fake_search)
    return queries


def _fake_retrieve(monkeypatch, response_for):
    calls = []

    async def fake_retrieve(url, **kwargs):
        calls.append((url, kwargs))
        return response_for(url)

    monkeypatch.setattr(tools, "retrieve", fake_retrieve)
    return calls


def test_web_search_filters_by_claim_day_and_formats_organic_only(searches):
    session = ToolSession(claim_date="2024-02-05T12:22:02")
    text = session.web_search("eiffel tower")

    assert searches[0].q == "eiffel tower"
    assert searches[0].before == date(2024, 2, 5)
    assert "https://example.com/old" in text
    assert "Jan 3, 2024" in text
    assert "Before the claim." in text
    assert "Leaky" not in text
    assert session.citations == ["https://example.com/old"]


def test_web_search_limit(searches):
    session = ToolSession(claim_date="2024-02-05", max_searches=1)
    session.web_search("first")
    text = session.web_search("second")

    assert len(searches) == 1
    assert "limit" in text.lower()


def test_web_search_quota_exceeded_terminates_run(monkeypatch):
    async def fake_search(query):
        raise QuotaExceededError("Serper credits used up")

    monkeypatch.setattr(tools, "search", fake_search)
    with pytest.raises(SearchQuotaExceededError):
        ToolSession(claim_date="2024-02-05").web_search("q")


def test_web_search_retries_rate_limits(monkeypatch):
    attempts = []

    async def fake_search(query):
        attempts.append(query)
        if len(attempts) < 3:
            raise RateLimitError("slow down")
        return _serper_response()

    monkeypatch.setattr(tools, "search", fake_search)
    monkeypatch.setattr(tools.time, "sleep", lambda _: None)
    text = ToolSession(claim_date="2024-02-05").web_search("q")

    assert len(attempts) == 3
    assert "https://example.com/old" in text


def test_fetch_url_truncates_content(monkeypatch):
    calls = _fake_retrieve(monkeypatch, lambda url: ScrapingResponse(
        url=url, content=ScrapedContent(markdown="Q" * (MAX_FETCH_CHARS + 5000)), output_format="markdown",
    ))
    session = ToolSession(claim_date="2024-02-05", scrape_methods=["firecrawl"])
    text = session.fetch_url("https://example.com/page")

    assert calls[0][1]["output_format"] == "markdown"
    assert calls[0][1]["methods"] == ["firecrawl"]
    assert text.count("Q") == MAX_FETCH_CHARS
    assert "[Content truncated...]" in text
    assert session.citations == ["https://example.com/page"]


def test_fetch_url_failure_returns_error_text(monkeypatch):
    _fake_retrieve(monkeypatch, lambda url: ScrapingResponse(
        url=url, content=None, output_format="markdown",
        errors={"firecrawl": TargetUnavailableError("404 Not Found")},
    ))
    session = ToolSession(claim_date="2024-02-05")
    text = session.fetch_url("https://example.com/gone")

    assert text.startswith("Could not retrieve https://example.com/gone")
    assert "TargetUnavailableError" in text
    assert session.citations == []


def test_fetch_url_limit(monkeypatch):
    calls = _fake_retrieve(monkeypatch, lambda url: ScrapingResponse(
        url=url, content=ScrapedContent(markdown="content"), output_format="markdown",
    ))
    session = ToolSession(claim_date="2024-02-05", max_fetches=1)
    session.fetch_url("https://example.com/a")
    text = session.fetch_url("https://example.com/b")

    assert len(calls) == 1
    assert "limit" in text.lower()


def test_call_dispatch_and_invalid_input(monkeypatch):
    calls = _fake_retrieve(monkeypatch, lambda url: None)
    session = ToolSession(claim_date="2024-02-05")

    assert "Unknown tool" in session.call("delete_everything", {})
    assert "Invalid arguments" in session.call("fetch_url", None)
    assert "not a valid URL" in session.call("fetch_url", {"url": "example.com"})
    assert "must not be empty" in session.call("web_search", {"query": "  "})
    assert calls == []


def test_tool_formats():
    assert [t["function"]["name"] for t in tools.openai_tools()] == ["web_search", "fetch_url"]
    assert [t["name"] for t in tools.anthropic_tools()] == ["web_search", "fetch_url"]
    assert all("input_schema" in t for t in tools.anthropic_tools())


def test_system_prompt_mentions_tools_and_all_labels():
    assert "web_search" in SYSTEM_PROMPT
    assert "fetch_url" in SYSTEM_PROMPT
    for label in LABELS_7:
        assert label.upper() in SYSTEM_PROMPT
