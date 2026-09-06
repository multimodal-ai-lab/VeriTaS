"""Stage 2.1 - Accessibility and publication time of an evidence source (Spec §3.1).

All retrieval runs through scrapeMM; nothing here fetches a URL on its own. The
order is:

1. retrieve the source through scrapeMM with ``format="html"``;
2. read the publication time off the HTML's meta tags;
3. convert that same HTML into a ``MultimodalSequence`` with scrapeMM's own
   ``to_multimodal_sequence``, so no second download is needed;
4. only if step 2 found nothing, have a cheap LLM read a publication time off the
   converted content.

scrapeMM's ``html`` format is only served by its Firecrawl and Decodo backends, and
step 1 is restricted to those two accordingly. Sources that scrapeMM handles through
a dedicated API integration (social media, archiving services, video platforms)
therefore come back unsuccessful and are recorded as inaccessible - there is no
second attempt in scrapeMM's default ``multimodal_sequence`` format.

A source that merely rate-limited us is *not* inaccessible: the retrieval reports
``rate_limited`` and the caller defers the item (see `filtering.filter_single`).
An exhausted scrapeMM quota is a run-level condition and is raised as VeriTaS'
``QuotaExceededError``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import aiohttp
from ezmm import MultimodalSequence
from scrapemm import RateLimitError as ScrapeRateLimitError, retrieve
from scrapemm.common import ScrapingResponse
from scrapemm.common.exceptions import QuotaExceededError as ScrapeQuotaExceededError
from scrapemm.retrieval import postprocess_media
from scrapemm.util import preprocess_url, to_multimodal_sequence

from veritas.common import Prompt
from veritas.common.appearance import is_sufficient_content
from veritas.gold_evidence import (
    dating_model,
    max_source_content_length,
    max_video_size,
    reasoning_effort_dating,
)
from veritas.gold_evidence.llm import FATAL_ERRORS, resolve_model
from veritas.gold_evidence.models import to_naive
from veritas.models import QuotaExceededError
from veritas.pipeline.stage_3 import extract_date_meta
from veritas.util.parsing import determine_date
from veritas.util.scraping import HEADERS

logger = logging.getLogger("VeriTaS")

DATING_PROMPT_PATH = "veritas/gold_evidence/prompts/determine_publication_time.md.j2"

#: scrapeMM serves `format="html"` only through these backends.
HTML_METHODS = ["firecrawl", "decodo"]


@dataclass
class SourceRetrieval:
    """Outcome of retrieving one evidence source."""

    accessible: bool
    content: MultimodalSequence | None = None
    available_since: datetime | None = None
    #: How `available_since` was obtained: 'meta' | 'llm' | None
    dating_method: str | None = None
    #: The scrapeMM retrieval method that succeeded.
    method: str | None = None
    #: Which of the two scrapeMM formats produced the content: 'html' | 'multimodal_sequence'
    format: str | None = None
    error: str | None = None
    rate_limited: bool = False


def new_session() -> aiohttp.ClientSession:
    """Session used by `to_multimodal_sequence` to download the page's media."""
    return aiohttp.ClientSession(headers=HEADERS)


async def retrieve_source(locator: str,
                          session: aiohttp.ClientSession | None = None,
                          determine_time: bool = True) -> SourceRetrieval:
    """Retrieves an evidence source and determines when it became publicly available."""
    if session is None:
        async with new_session() as own_session:
            return await retrieve_source(locator, own_session, determine_time)

    try:
        url = preprocess_url(locator)
    except Exception:
        url = locator

    try:
        result = await _retrieve_via_html(url, session)
    except ScrapeRateLimitError as e:
        # Per-source condition: the item is deferred, not judged.
        return SourceRetrieval(accessible=False, error=str(e), rate_limited=True)
    except ScrapeQuotaExceededError as e:
        # Run-level condition: continuing would silently mark every remaining
        # source inaccessible, so it aborts the run like any other quota error.
        raise QuotaExceededError(f"scrapeMM quota exhausted: {e}") from e
    except Exception as e:
        logger.debug(f"Retrieval of evidence source {url} failed: {type(e).__name__}: {e}")
        return SourceRetrieval(accessible=False, error=f"{type(e).__name__}: {e}")

    # Step 4: the LLM only sees sources whose meta tags carried no publication time.
    if result.accessible and determine_time and result.available_since is None:
        result.available_since = await determine_publication_time_llm(url, result.content)
        if result.available_since is not None:
            result.dating_method = "llm"

    return result


async def _retrieve_via_html(url: str, session: aiohttp.ClientSession) -> SourceRetrieval:
    """Steps 1-3: scrapeMM's HTML format, its meta tags, and its HTML converter."""
    # Step 1
    response = await retrieve(url, show_progress=False, format="html",
                              methods=HTML_METHODS, prioritize="completeness",
                              max_video_size=max_video_size)
    if not isinstance(response, ScrapingResponse):
        return SourceRetrieval(accessible=False, format="html",
                               error="scrapeMM did not return a ScrapingResponse.")
    if not response.successful:
        return _failed(response, format="html")

    html = str(response.content)

    # Step 2
    available_since = _date_from_meta(html)

    # Step 3
    content = await _to_multimodal_sequence(html, url, session)

    if not is_sufficient_content(content):
        return SourceRetrieval(accessible=False, format="html", method=response.method,
                               available_since=available_since,
                               dating_method="meta" if available_since else None,
                               error="Retrieved content is insufficient.")

    return SourceRetrieval(
        accessible=True,
        content=content,
        available_since=available_since,
        dating_method="meta" if available_since else None,
        method=response.method,
        format="html",
    )


def _failed(response: ScrapingResponse, *, format: str) -> SourceRetrieval:
    """Turns an unsuccessful scrapeMM response into a SourceRetrieval.

    scrapeMM reports most conditions inside the response rather than by raising, so
    an exhausted quota is recognized here too and aborts the run either way."""
    error = None
    rate_limited = False
    if response.errors:
        first = list(response.errors.values())[0]
        if isinstance(first, ScrapeQuotaExceededError):
            raise QuotaExceededError(f"scrapeMM quota exhausted: {first}")
        error = str(first) or type(first).__name__
        rate_limited = isinstance(first, ScrapeRateLimitError)
    return SourceRetrieval(accessible=False, method=response.method, format=format,
                           error=error, rate_limited=rate_limited)


async def _to_multimodal_sequence(html: str, url: str,
                                  session: aiohttp.ClientSession) -> MultimodalSequence | None:
    """Step 3, via scrapeMM's own converter, so the images and videos referenced by
    the page are downloaded and inlined exactly as `scrapemm.retrieve` would.

    Only the arguments scrapeMM passes internally are used: any extra keyword would
    be forwarded down to `aiohttp`'s `session.get`. `max_video_size` in particular
    belongs to `retrieve` and is rejected here, so it is applied in step 1."""
    try:
        content = await to_multimodal_sequence(html, session=session, url=url)
    except Exception as e:
        logger.debug(f"Could not convert HTML of {url} to a MultimodalSequence: {e}")
        return None
    if content is None:
        return None
    try:
        # Moves media out of temp files into the ezMM store and normalizes videos,
        # which `scrapemm.retrieve` does for its own results.
        postprocess_media(content)
    except Exception as e:
        logger.debug(f"Media postprocessing failed for {url}: {e}")
    return content


def _date_from_meta(html: str) -> datetime | None:
    """Step 2: publication time from the page's standard meta tags (reuses stage 3)."""
    try:
        return to_naive(extract_date_meta(html))
    except Exception:
        return None


async def determine_publication_time_llm(url: str,
                                         content: MultimodalSequence | None) -> datetime | None:
    """Step 4: reads an explicitly stated publication time off the retrieved content
    (post time, dateline, "published on ..."). Returns None whenever no clear date
    is stated - in particular for tools."""
    if not content:
        return None

    from veritas.models import gpt_nano

    model = resolve_model(dating_model, gpt_nano)
    prompt = Prompt(
        DATING_PROMPT_PATH,
        url=url,
        content=str(content)[:max_source_content_length],
    )
    try:
        response = await model.generate(prompt, extract="last_code_span",
                                        resolve_media=False,
                                        reasoning_effort=reasoning_effort_dating)
    except FATAL_ERRORS:
        raise
    except Exception as e:
        logger.debug(f"Could not determine publication time for {url}: {e}")
        return None

    if not response:
        return None
    answer = str(response).strip()
    if not answer or answer.lower() in ("none", "null", "unknown", "n/a"):
        return None
    return to_naive(determine_date(answer))
