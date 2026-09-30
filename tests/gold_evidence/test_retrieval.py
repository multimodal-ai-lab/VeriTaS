"""Source retrieval order (Spec §3.1).

Everything goes through scrapeMM, in exactly one call per source: the response
carries the multimodal sequence *and* the raw HTML the method saw along the way.
The order is: one `retrieve(output_format="multimodal")` -> publication time from
the meta tags of that response's HTML -> LLM dating only if the meta tags carried
nothing.
"""

from datetime import datetime

import pytest
from ezmm import MultimodalSequence
from scrapemm.common import ScrapedContent, ScrapingResponse
from scrapemm.common.exceptions import RateLimitError as ScrapeRateLimitError

from veritas.gold_evidence import retrieval as retrieval_module
from veritas.gold_evidence.retrieval import retrieve_source

PAGE = ("A sufficiently long piece of source content so that it passes the "
        "sufficiency check applied to every scraped evidence source. " * 3)


@pytest.fixture
def scrapemm(monkeypatch):
    """Replaces every scrapeMM entry point and records how it was called."""
    state = {
        "calls": [],
        "html": "<html><head></head><body>page</body></html>",
        "response": None,            # set to override the response
        "meta_date": None,
        "meta_input": [],
        "llm_date": None,
        "llm_calls": [],
        "content": MultimodalSequence(PAGE),
    }

    async def fake_retrieve(url, *, show_progress=True, output_format="multimodal",
                            methods="auto", prioritize="completeness", **kwargs):
        state["calls"].append({"url": url, "output_format": output_format,
                               "methods": methods, "kwargs": kwargs})
        if state["response"] is not None:
            return state["response"]
        content = ScrapedContent(html=state["html"], multimodal=state["content"])
        return ScrapingResponse(url=url, content=content, method="firecrawl",
                                output_format=output_format)

    def fake_meta(html):
        state["meta_input"].append(html)
        return state["meta_date"]

    async def fake_llm(url, content):
        state["llm_calls"].append(url)
        return state["llm_date"]

    monkeypatch.setattr(retrieval_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(retrieval_module, "extract_date_meta", fake_meta)
    monkeypatch.setattr(retrieval_module, "determine_publication_time_llm", fake_llm)
    return state


def failed_response(error: Exception, url: str = "https://example.org/a",
                    method: str = "decodo") -> ScrapingResponse:
    return ScrapingResponse(url=url, content=None, errors={method: error}, method=method)


# --- The prescribed order --------------------------------------------------

@pytest.mark.asyncio
async def test_the_source_is_retrieved_exactly_once(scrapemm):
    """The multimodal content and the HTML come out of the same response, so there
    is no second call and no separate conversion."""
    result = await retrieve_source("https://example.org/a")

    assert result.accessible
    assert len(scrapemm["calls"]) == 1
    assert scrapemm["calls"][0]["output_format"] == "multimodal"
    assert result.content is scrapemm["content"]
    assert result.method == "firecrawl"


@pytest.mark.asyncio
async def test_the_methods_are_not_restricted(scrapemm):
    """Every scrapeMM backend serves the multimodal format, so the best method per
    domain is left to scrapeMM."""
    await retrieve_source("https://x.com/a/status/1")
    assert scrapemm["calls"][0]["methods"] == "auto"


@pytest.mark.asyncio
async def test_the_date_is_read_off_the_html_of_that_same_response(scrapemm):
    scrapemm["meta_date"] = datetime(2024, 5, 10)
    result = await retrieve_source("https://example.org/a")

    assert scrapemm["meta_input"] == [scrapemm["html"]]
    assert result.available_since == datetime(2024, 5, 10)
    assert result.dating_method == "meta"


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


# --- Sources without an HTML page ------------------------------------------

@pytest.mark.asyncio
async def test_a_source_without_html_is_still_retrieved(scrapemm):
    """Social media, archives and video platforms are served by scrapeMM's API
    integrations, which have no HTML page - the LLM dates those."""
    scrapemm["response"] = ScrapingResponse(
        url="https://x.com/a/status/1",
        content=ScrapedContent(multimodal=MultimodalSequence(PAGE)),
        method="x", output_format="multimodal")
    scrapemm["llm_date"] = datetime(2024, 5, 12)
    result = await retrieve_source("https://x.com/a/status/1")

    assert result.accessible
    assert result.method == "x"
    assert scrapemm["meta_input"] == []  # nothing to read meta tags off
    assert result.dating_method == "llm"


# --- Failures --------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_failure_is_reported(scrapemm):
    scrapemm["response"] = failed_response(Exception("retrieval failed"))
    result = await retrieve_source("https://example.org/gone")

    assert result.accessible is False
    assert "retrieval failed" in result.error


@pytest.mark.asyncio
async def test_insufficient_content_is_not_accessible(scrapemm):
    scrapemm["content"] = MultimodalSequence("404")
    result = await retrieve_source("https://example.org/empty")
    assert result.accessible is False


@pytest.mark.asyncio
async def test_a_meta_date_is_kept_even_when_the_content_is_unusable(scrapemm):
    """The page came back but holds nothing usable; its meta date is still valid."""
    scrapemm["meta_date"] = datetime(2024, 5, 10)
    scrapemm["content"] = MultimodalSequence("404")
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
    """A rate-limited source is deferrable, not inaccessible - retrying would only
    burn the same quota."""
    scrapemm["response"] = failed_response(ScrapeRateLimitError("slow down"))
    result = await retrieve_source("https://example.org/a")

    assert result.accessible is False
    assert result.rate_limited is True
    assert len(scrapemm["calls"]) == 1


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
    scrapemm["response"] = "not a ScrapingResponse"
    result = await retrieve_source("https://example.org/a")
    assert result.accessible is False
    assert "ScrapingResponse" in result.error


@pytest.mark.asyncio
async def test_scrapemm_quota_aborts_the_run(scrapemm):
    """Unlike a rate limit, an exhausted quota is not a per-source outcome - and
    scrapeMM reports it inside the response rather than by raising."""
    from scrapemm.common.exceptions import QuotaExceededError as ScrapeQuota

    from veritas.models import QuotaExceededError

    scrapemm["response"] = failed_response(ScrapeQuota("no credits left"))
    with pytest.raises(QuotaExceededError):
        await retrieve_source("https://example.org/a")


# --- Archive.today's access check -------------------------------------------

@pytest.mark.asyncio
async def test_archive_today_gate_is_reported_and_not_retried(scrapemm):
    """A source gated behind Archive.today's access check is deferrable, not
    inaccessible - like a rate limit, but resolved only once a human passes the
    check (see `scripts/retry_deferred_archive_today.py`)."""
    from scrapemm.common.exceptions import CaptchaEncounteredError

    scrapemm["response"] = failed_response(CaptchaEncounteredError("captcha"),
                                           url="https://archive.ph/abcde")
    result = await retrieve_source("https://archive.ph/abcde")

    assert result.accessible is False
    assert result.gated is True
    assert result.rate_limited is False
    assert len(scrapemm["calls"]) == 1


@pytest.mark.asyncio
async def test_a_captcha_on_a_non_archive_today_domain_is_not_deferred(scrapemm):
    """Only Archive.today's access check is deferred: it is the one gate
    scrapeMM buffers and later answers from a persistent cache. A CAPTCHA
    anywhere else is recorded as a plain, permanent failure, as before."""
    from scrapemm.common.exceptions import CaptchaEncounteredError

    scrapemm["response"] = failed_response(CaptchaEncounteredError("captcha"),
                                           url="https://example.org/gated")
    result = await retrieve_source("https://example.org/gated")

    assert result.accessible is False
    assert result.gated is False


@pytest.mark.asyncio
async def test_raised_archive_today_captcha_is_caught(scrapemm, monkeypatch):
    from scrapemm.common.exceptions import CaptchaEncounteredError

    async def raising(*a, **k):
        raise CaptchaEncounteredError("captcha")

    monkeypatch.setattr(retrieval_module, "retrieve", raising)
    result = await retrieve_source("https://archive.ph/abcde")
    assert result.gated is True


@pytest.mark.asyncio
async def test_raised_captcha_on_another_domain_is_not_gated(scrapemm, monkeypatch):
    from scrapemm.common.exceptions import CaptchaEncounteredError

    async def raising(*a, **k):
        raise CaptchaEncounteredError("captcha")

    monkeypatch.setattr(retrieval_module, "retrieve", raising)
    result = await retrieve_source("https://example.org/gated")
    assert result.gated is False


# --- An unreachable scrapeMM server aborts the run, like an exhausted quota ----

@pytest.mark.asyncio
async def test_raised_server_error_aborts_the_run(scrapemm, monkeypatch):
    """Total unreachability is always raised directly out of `retrieve()`
    (never placed in `response.errors`), so this is the path that matters."""
    from scrapemm.common.exceptions import ServerError

    from veritas.models import QuotaExceededError

    async def raising(*a, **k):
        raise ServerError("Could not reach the scrapeMM server at http://localhost:8080: timeout")

    monkeypatch.setattr(retrieval_module, "retrieve", raising)
    with pytest.raises(QuotaExceededError):
        await retrieve_source("https://example.org/a")


@pytest.mark.asyncio
async def test_server_error_in_response_errors_also_aborts_the_run(scrapemm):
    """The narrower case where the server reports a missing result mid-stream
    rather than failing the whole call - still a run-level condition.

    This also regression-tests a bug `test_scrapemm_quota_aborts_the_run`
    happened not to catch: `_failed()` raises `veritas.models.QuotaExceededError`
    directly (not scrapeMM's own exception class), and `retrieve_source`'s
    `except Exception` clause used to catch it right back and record it as a
    plain per-source failure instead of letting it propagate - see the
    `except QuotaExceededError: raise` clause above the others."""
    from scrapemm.common.exceptions import ServerError

    from veritas.models import QuotaExceededError

    scrapemm["response"] = failed_response(ServerError("The server did not return a result for this URL."))
    with pytest.raises(QuotaExceededError):
        await retrieve_source("https://example.org/a")
