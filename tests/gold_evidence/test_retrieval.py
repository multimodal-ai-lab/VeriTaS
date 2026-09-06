"""Source retrieval order (Spec §3.1).

Everything goes through scrapeMM. The prescribed order is: HTML format -> meta-tag
publication time -> scrapeMM's HTML-to-MultimodalSequence conversion -> LLM dating
only if the meta tags carried nothing. There is no second attempt in scrapeMM's
default format: a source its HTML backends cannot serve is inaccessible.
"""

from datetime import datetime

import pytest
from ezmm import MultimodalSequence
from scrapemm.common import ScrapingResponse
from scrapemm.common.exceptions import RateLimitError as ScrapeRateLimitError

from veritas.gold_evidence import retrieval as retrieval_module
from veritas.gold_evidence.retrieval import HTML_METHODS, retrieve_source

PAGE = ("A sufficiently long piece of source content so that it passes the "
        "sufficiency check applied to every scraped evidence source. " * 3)


@pytest.fixture
def scrapemm(monkeypatch):
    """Replaces every scrapeMM entry point and records how it was called."""
    state = {
        "calls": [],
        "html": "<html><head></head><body>page</body></html>",
        "html_response": None,       # set to override the html-format response
        "meta_date": None,
        "llm_date": None,
        "llm_calls": [],
        "converted": MultimodalSequence(PAGE),
        "postprocessed": [],
    }

    async def fake_retrieve(url, *, show_progress=True, format="multimodal_sequence",
                            methods="auto", prioritize="completeness", **kwargs):
        state["calls"].append({"url": url, "format": format, "methods": methods,
                               "kwargs": kwargs})
        if state["html_response"] is not None:
            return state["html_response"]
        return ScrapingResponse(url=url, content=state["html"], method="firecrawl")

    async def fake_convert(html, session=None, url=None):
        state["converted_from"] = html
        return state["converted"]

    def fake_meta(html):
        return state["meta_date"]

    async def fake_llm(url, content):
        state["llm_calls"].append(url)
        return state["llm_date"]

    monkeypatch.setattr(retrieval_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(retrieval_module, "to_multimodal_sequence", fake_convert)
    monkeypatch.setattr(retrieval_module, "extract_date_meta", fake_meta)
    monkeypatch.setattr(retrieval_module, "determine_publication_time_llm", fake_llm)
    monkeypatch.setattr(retrieval_module, "postprocess_media",
                        lambda content: state["postprocessed"].append(content))
    return state


def failed_response(error: Exception, url: str = "https://example.org/a",
                    method: str = "decodo") -> ScrapingResponse:
    return ScrapingResponse(url=url, content=None, errors={method: error}, method=method)


# --- The prescribed order --------------------------------------------------

@pytest.mark.asyncio
async def test_html_format_is_requested_first(scrapemm):
    await retrieve_source("https://example.org/a")
    first = scrapemm["calls"][0]
    assert first["format"] == "html"
    assert first["methods"] == HTML_METHODS


@pytest.mark.asyncio
async def test_html_is_converted_by_scrapemm_not_refetched(scrapemm):
    """Step 3 reuses the very HTML from step 1 - there is no second download."""
    result = await retrieve_source("https://example.org/a")
    assert result.accessible
    assert scrapemm["converted_from"] == scrapemm["html"]
    assert len(scrapemm["calls"]) == 1
    assert result.format == "html"
    assert result.method == "firecrawl"


@pytest.mark.asyncio
async def test_meta_date_wins_and_skips_the_llm(scrapemm):
    scrapemm["meta_date"] = datetime(2024, 5, 10)
    scrapemm["llm_date"] = datetime(1999, 1, 1)
    result = await retrieve_source("https://example.org/a")

    assert result.available_since == datetime(2024, 5, 10)
    assert result.dating_method == "meta"
    assert scrapemm["llm_calls"] == []


@pytest.mark.asyncio
async def test_llm_runs_only_when_meta_found_nothing(scrapemm):
    scrapemm["meta_date"] = None
    scrapemm["llm_date"] = datetime(2024, 5, 12)
    result = await retrieve_source("https://example.org/a")

    assert result.available_since == datetime(2024, 5, 12)
    assert result.dating_method == "llm"
    assert scrapemm["llm_calls"] == ["https://example.org/a"]


@pytest.mark.asyncio
async def test_undated_source_stays_undated(scrapemm):
    result = await retrieve_source("https://example.org/a")
    assert result.accessible
    assert result.available_since is None
    assert result.dating_method is None


@pytest.mark.asyncio
async def test_dating_can_be_switched_off(scrapemm):
    scrapemm["llm_date"] = datetime(2024, 5, 12)
    result = await retrieve_source("https://example.org/a", determine_time=False)
    assert result.available_since is None
    assert scrapemm["llm_calls"] == []


@pytest.mark.asyncio
async def test_media_are_postprocessed_into_the_ezmm_store(scrapemm):
    await retrieve_source("https://example.org/a")
    assert scrapemm["postprocessed"] == [scrapemm["converted"]]


# --- Sources the HTML backends cannot serve --------------------------------

@pytest.mark.asyncio
async def test_no_second_attempt_in_the_default_format(scrapemm):
    """Integration-served sources (social media, archives) cannot deliver HTML;
    scrapeMM reports that as an unsuccessful response, not an exception."""
    scrapemm["html_response"] = failed_response(
        AssertionError("'html' format is only compatible with 'firecrawl' and 'decodo'"))
    result = await retrieve_source("https://x.com/a/status/1")

    assert result.accessible is False
    assert [call["format"] for call in scrapemm["calls"]] == ["html"]


@pytest.mark.asyncio
async def test_the_failure_is_reported(scrapemm):
    scrapemm["html_response"] = failed_response(Exception("html failed"))
    result = await retrieve_source("https://example.org/gone")

    assert result.accessible is False
    assert "html failed" in result.error


@pytest.mark.asyncio
async def test_insufficient_content_is_not_accessible(scrapemm):
    scrapemm["converted"] = MultimodalSequence("404")
    result = await retrieve_source("https://example.org/empty")
    assert result.accessible is False


@pytest.mark.asyncio
async def test_a_meta_date_is_kept_even_when_the_content_is_unusable(scrapemm):
    """HTML came back but converted to nothing usable; its meta date is still valid."""
    scrapemm["meta_date"] = datetime(2024, 5, 10)
    scrapemm["converted"] = MultimodalSequence("404")
    result = await retrieve_source("https://example.org/a")

    assert result.accessible is False
    assert result.available_since == datetime(2024, 5, 10)
    assert result.dating_method == "meta"
    assert scrapemm["llm_calls"] == []


@pytest.mark.asyncio
async def test_the_video_size_cap_is_forwarded(scrapemm):
    """scrapeMM applies it while downloading the page's media."""
    from veritas.gold_evidence import max_video_size

    await retrieve_source("https://example.org/a")
    kwargs = scrapemm["calls"][0]["kwargs"]
    assert "max_video_size" in kwargs
    assert kwargs["max_video_size"] == max_video_size


# --- Rate limiting ---------------------------------------------------------

@pytest.mark.asyncio
async def test_rate_limit_is_reported_and_not_retried(scrapemm):
    """A rate-limited source is deferrable, not inaccessible - retrying the other
    format would only burn the same quota."""
    scrapemm["html_response"] = failed_response(ScrapeRateLimitError("slow down"))
    result = await retrieve_source("https://example.org/a")

    assert result.accessible is False
    assert result.rate_limited is True
    assert [call["format"] for call in scrapemm["calls"]] == ["html"]


@pytest.mark.asyncio
async def test_raised_rate_limit_is_caught(scrapemm, monkeypatch):
    async def raising(*a, **k):
        raise ScrapeRateLimitError("slow down")

    monkeypatch.setattr(retrieval_module, "retrieve", raising)
    result = await retrieve_source("https://example.org/a")
    assert result.rate_limited is True


@pytest.mark.asyncio
async def test_unexpected_errors_do_not_propagate(scrapemm, monkeypatch):
    async def raising(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(retrieval_module, "retrieve", raising)
    result = await retrieve_source("https://example.org/a")
    assert result.accessible is False
    assert "RuntimeError" in result.error


@pytest.mark.asyncio
async def test_non_response_return_value_is_handled(scrapemm):
    scrapemm["html_response"] = "not a ScrapingResponse"
    result = await retrieve_source("https://example.org/a")
    assert result.accessible is False
    assert "ScrapingResponse" in result.error


@pytest.mark.asyncio
async def test_scrapemm_quota_aborts_the_run(scrapemm):
    """Unlike a rate limit, an exhausted quota is not a per-source outcome - and
    scrapeMM reports it inside the response rather than by raising."""
    from scrapemm.common.exceptions import QuotaExceededError as ScrapeQuota

    from veritas.models import QuotaExceededError

    scrapemm["html_response"] = failed_response(ScrapeQuota("no credits left"))
    with pytest.raises(QuotaExceededError):
        await retrieve_source("https://example.org/a")
