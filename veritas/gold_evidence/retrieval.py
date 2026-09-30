"""Stage 2.1 - Accessibility and publication time of an evidence source (Spec §3.1).

All retrieval runs through scrapeMM; nothing here fetches a URL on its own. A single
scrapeMM call per source settles accessibility, content and dating:

1. retrieve the source through scrapeMM with ``output_format="multimodal"``;
2. read the publication time off the raw HTML the very same response carries
   (``response.content.html``), if the used method had access to it - JSON-LD,
   Open Graph, microdata and other publication meta tags (see `dating`);
3. only if step 2 found nothing, have a cheap LLM read a publication time off the
   retrieved content, shown alongside the date-bearing markup of the HTML that step
   2 could not attribute with certainty (``<time>`` elements and the like).

scrapeMM produces every format preceding the requested one, so the multimodal
sequence and the page's raw HTML come out of *one* retrieval - there is no second
call and no separate HTML-to-sequence conversion. Consequently retrieval is no
longer restricted to scrapeMM's HTML-capable backends: sources served by a dedicated
API integration (social media, archiving services, video platforms) are retrieved
like any other. Those carry no HTML page, so they are dated in step 3.

A source that merely rate-limited us is *not* inaccessible: the retrieval reports
``rate_limited`` and the caller defers that source (see `filtering.settle_source`).
Likewise, a source gated behind Archive.today's access check reports ``gated``
and is deferred the same way - scrapeMM does not solve the CAPTCHA, but it
buffers the URL and answers it from a persistent cache once a human passes the
check, so retrying later (here, or on demand via
`scripts/retry_deferred_archive_today.py`) succeeds without a second retrieval.
An exhausted scrapeMM quota is a run-level condition and is raised as VeriTaS'
``QuotaExceededError`` - and so is a scrapeMM server that cannot be reached at
all (``ServerError``): every remaining source would fail the exact same way, so
this aborts the run instead of recording each one as a false "inaccessible"
verdict for what the outage merely prevented from being checked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

from ezmm import MultimodalSequence
from scrapemm import RateLimitError as ScrapeRateLimitError, retrieve
from scrapemm.common import ScrapingResponse
from scrapemm.common.exceptions import CaptchaEncounteredError
from scrapemm.common.exceptions import QuotaExceededError as ScrapeQuotaExceededError
from scrapemm.common.exceptions import ServerError as ScrapeServerError

from veritas.common import Prompt
from veritas.common.appearance import is_sufficient_content
from veritas.gold_evidence import (
    dating_model,
    max_source_content_length,
    max_video_size,
    reasoning_effort_dating,
)
from veritas.gold_evidence.dating import (
    extract_publication_time,
    parse_publication_value,
    publication_date_hints,
)
from veritas.gold_evidence.llm import FATAL_ERRORS, resolve_model
from veritas.models import QuotaExceededError
from veritas.util.url import is_archive_today_url, preprocess_url

logger = logging.getLogger("VeriTaS")

DATING_PROMPT_PATH = "veritas/gold_evidence/prompts/determine_publication_time.md.j2"

#: The format the content is needed in. scrapeMM fills the preceding formats (raw
#: HTML, Markdown) in along the way, whenever the used method had access to them.
OUTPUT_FORMAT = "multimodal"


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
    error: str | None = None
    rate_limited: bool = False
    #: Archive.today's access check is up; deferred like a rate limit (see
    #: module docstring). Never set for a CAPTCHA on any other domain.
    gated: bool = False
    #: Date-bearing markup of the HTML for the LLM in step 3 (see
    #: `dating.publication_date_hints`). Transient - never persisted.
    date_hints: list[str] = field(default_factory=list)


async def retrieve_source(locator: str, determine_time: bool = True) -> SourceRetrieval:
    """Retrieves an evidence source and determines when it became publicly available."""
    try:
        url = preprocess_url(locator)
    except Exception:
        url = locator

    try:
        result = await _retrieve(url)
    except QuotaExceededError:
        # Raised by `_failed()` below when scrapeMM reported the fatal condition
        # inside `response.errors` rather than by raising it directly - let it
        # propagate exactly like the two clauses below, instead of falling into
        # the generic `except Exception` and being recorded as a per-source
        # failure.
        raise
    except ScrapeRateLimitError as e:
        # Per-source condition: the item is deferred, not judged.
        return SourceRetrieval(accessible=False, error=str(e), rate_limited=True)
    except CaptchaEncounteredError as e:
        # Only Archive.today's access check is deferred (see module docstring);
        # a CAPTCHA on any other domain is a plain, permanent failure.
        return SourceRetrieval(accessible=False, error=str(e), gated=is_archive_today_url(url))
    except ScrapeQuotaExceededError as e:
        # Run-level condition: continuing would silently mark every remaining
        # source inaccessible, so it aborts the run like any other quota error.
        raise QuotaExceededError(f"scrapeMM quota exhausted: {e}") from e
    except ScrapeServerError as e:
        # Same reasoning: the server is unreachable, not this one source.
        raise QuotaExceededError(f"scrapeMM server unreachable: {e}") from e
    except Exception as e:
        logger.debug(f"Retrieval of evidence source {url} failed: {type(e).__name__}: {e}")
        return SourceRetrieval(accessible=False, error=f"{type(e).__name__}: {e}")

    # Step 3: the LLM only sees sources whose meta tags carried no publication time.
    if result.accessible and determine_time and result.available_since is None:
        result.available_since = await determine_publication_time_llm(
            url, result.content, hints=result.date_hints)
        if result.available_since is not None:
            result.dating_method = "llm"

    return result


async def _retrieve(url: str) -> SourceRetrieval:
    """Steps 1-2: the one scrapeMM call and the meta tags of the HTML it carries."""
    # Step 1
    response = await retrieve(url, show_progress=False, output_format=OUTPUT_FORMAT,
                              prioritize="completeness", max_video_size=max_video_size)
    if not isinstance(response, ScrapingResponse):
        return SourceRetrieval(accessible=False,
                               error="scrapeMM did not return a ScrapingResponse.")
    if not response.success:
        return _failed(response)

    content = response.content.multimodal

    # Step 2: free of charge - the same response carries the page's raw HTML
    # whenever the used method had access to it.
    html = response.content.html
    available_since = _date_from_meta(html)
    # Only needed if step 3 will run; cheap, but no reason to parse the page twice.
    date_hints = _date_hints(html) if available_since is None else []

    if not is_sufficient_content(content):
        return SourceRetrieval(accessible=False, method=response.method,
                               available_since=available_since,
                               dating_method="meta" if available_since else None,
                               error="Retrieved content is insufficient.")

    return SourceRetrieval(
        accessible=True,
        content=content,
        available_since=available_since,
        dating_method="meta" if available_since else None,
        method=response.method,
        date_hints=date_hints,
    )


def _failed(response: ScrapingResponse) -> SourceRetrieval:
    """Turns an unsuccessful scrapeMM response into a SourceRetrieval.

    scrapeMM reports most conditions inside the response rather than by raising, so
    an exhausted quota is recognized here too and aborts the run either way."""
    error = None
    rate_limited = False
    gated = False
    if response.errors:
        first = list(response.errors.values())[0]
        if isinstance(first, ScrapeQuotaExceededError):
            raise QuotaExceededError(f"scrapeMM quota exhausted: {first}")
        if isinstance(first, ScrapeServerError):
            raise QuotaExceededError(f"scrapeMM server unreachable: {first}")
        error = str(first) or type(first).__name__
        rate_limited = isinstance(first, ScrapeRateLimitError)
        gated = isinstance(first, CaptchaEncounteredError) and is_archive_today_url(response.url)
    return SourceRetrieval(accessible=False, method=response.method,
                           error=error, rate_limited=rate_limited, gated=gated)


def _date_from_meta(html: str | None) -> datetime | None:
    """Step 2: publication time from the page's machine-readable markup - JSON-LD,
    meta tags, microdata (see `dating.extract_publication_time` for the priority).
    Recorded as ``dating_method="meta"`` like the meta-tag-only reading it replaces
    (stage 3's `extract_date_meta`, whose tags are all covered), so the stored data
    stays comparable. Sources without an HTML page - those scrapeMM serves through
    an API integration - have no markup and are left to the LLM in step 3."""
    if not html:
        return None
    try:
        return extract_publication_time(html)
    except Exception as e:
        logger.debug(f"Reading the publication time off the HTML failed: {e}")
        return None


def _date_hints(html: str | None) -> list[str]:
    if not html:
        return []
    try:
        return publication_date_hints(html)
    except Exception as e:
        logger.debug(f"Collecting date hints from the HTML failed: {e}")
        return []


async def determine_publication_time_llm(url: str,
                                         content: MultimodalSequence | None,
                                         hints: list[str] | None = None) -> datetime | None:
    """Step 3: reads an explicitly stated publication time off the retrieved content
    (post time, dateline, "published on ..."). Returns None whenever no clear date
    is stated - in particular for tools.

    `hints` are date-bearing HTML fragments with their context: the dateline is often
    part of the page chrome that scrapeMM's content extraction strips, so the text
    alone regularly states no date although the page does. The answer is parsed as
    strictly as the markup (`dating.parse_publication_value`): an answer lacking the
    day, or one that is implausible, counts as no date rather than being completed."""
    if not content:
        return None

    from veritas.models import gpt_nano

    model = resolve_model(dating_model, gpt_nano)
    prompt = Prompt(
        DATING_PROMPT_PATH,
        url=url,
        content=str(content)[:max_source_content_length],
        hints=hints or [],
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
    return parse_publication_value(answer)
