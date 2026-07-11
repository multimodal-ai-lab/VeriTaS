"""
Custom web search service with date filtering.

This module provides a search tool that can be used by OpenAI and Gemini providers
via function calling, with proper constraints for fact-checking benchmarks:
1. Date restriction: Only return results from before the claim date
2. Content retrieval: Fetch page content (via scrapeMM or simple static requests)

Scrape modes:
- "scrapemm": Scrape via scrapeMM, a meta-scraper whose backends (Firecrawl,
  social-media integrations, Decodo) are selected via the `scrape_methods` argument
- "lite": Simple static HTTP requests + BeautifulSoup (faster, text-only)
- "none": No scraping, only use search snippets
"""

import asyncio
import atexit
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Literal

import requests
from bs4 import BeautifulSoup

# Optional scrapeMM import (needed for "scrapemm" mode). scrapeMM is a meta-scraper
# whose own backends include Firecrawl ("firecrawl"), social-media APIs
# ("integrations") and the Decodo web-scraping API ("decodo"); the desired subset is
# selected via the `scrape_methods` argument and forwarded to retrieve(methods=...).
# Catches ImportError and EOFError (raised in non-interactive environments like SLURM)
try:
    from scrapemm import retrieve
    from scrapemm.common import ScrapingResponse
    SCRAPEMM_AVAILABLE = True
except (ImportError, EOFError, OSError):
    SCRAPEMM_AVAILABLE = False

ScrapeMode = Literal["scrapemm", "lite", "none"]


class SerperCreditsExhaustedError(BaseException):
    """Raised when Serper API credits are exhausted. Inherits BaseException so it
    propagates past bare `except Exception` handlers and terminates the run."""


class _AsyncRunner:
    """Manages a single background thread with an event loop for running async code.

    This avoids creating new threads/event loops for each scrape operation,
    preventing file descriptor exhaustion under concurrent load.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._loop = None
        self._thread = None
        self._started = False
        self._start_lock = threading.Lock()

    def _start_loop(self):
        """Start the background event loop thread."""
        with self._start_lock:
            if self._started:
                return
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            self._started = True
            atexit.register(self._shutdown)

    def _run_loop(self):
        """Run the event loop forever in the background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _shutdown(self):
        """Shutdown the event loop cleanly."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=5)

    def run(self, coro):
        """Run a coroutine and return its result (blocking)."""
        if not self._started:
            self._start_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()


# Global async runner instance
_async_runner = _AsyncRunner()


def _run_async(coro):
    """Run an async coroutine from sync code using a shared background event loop."""
    return _async_runner.run(coro)

try:
    from config import serperapi_key as VERITAS_SERPAPI_KEY
except ImportError:
    VERITAS_SERPAPI_KEY = None

@dataclass
class SearchResult:
    """A single search result with optional full content."""
    title: str
    url: str
    snippet: str
    content: str | None = None  # Full page content (markdown)
    date: str | None = None
    source: str | None = None


@dataclass
class SearchResponse:
    """Response from the search service."""
    query: str
    results: list[SearchResult] = field(default_factory=list)
    total_results: int = 0
    error: str | None = None


class SearchService:
    """
    Web search service with date filtering and content retrieval.

    Uses SerpAPI for search. Content scraping supports multiple modes:
    - "scrapemm": Full scraping via scrapeMM
    - "lite": Static HTTP requests + BeautifulSoup (faster, text-only)
    - "none": No scraping, only use search snippets
    """

    def __init__(
        self,
        serpapi_key: str | None = None,
        max_content_length: int = 8000,
        scrape_mode: ScrapeMode = "lite",
        scrape_methods: list[str] | str | None = "firecrawl",
    ):
        """
        Initialize the search service.

        Args:
            serpapi_key: SerpAPI key. If None, uses config/env var.
            max_content_length: Maximum characters of content to include per result.
            scrape_mode: How to fetch page content - "scrapemm", "lite", or "none".
            scrape_methods: For scrape_mode="scrapemm", which scrapeMM backends to
                           use, in order. A subset of {"integrations", "firecrawl",
                           "decodo"}, or "auto" to let scrapeMM choose per domain.
                           Defaults to ["firecrawl"] (Firecrawl only). Forwarded to
                           scrapeMM's retrieve(methods=...).
        """
        self.serpapi_key = self._resolve_serpapi_key(serpapi_key)
        self.max_content_length = max_content_length
        self.scrape_mode = scrape_mode
        self.scrape_methods = self._normalize_scrape_methods(scrape_methods)

        if scrape_mode == "scrapemm" and not SCRAPEMM_AVAILABLE:
            raise ImportError("scrapeMM is required for scrape_mode='scrapemm'. Install it or use 'lite' mode.")

        # Session for lite mode (connection pooling)
        self._session: requests.Session | None = None

    @staticmethod
    def _normalize_scrape_methods(methods: list[str] | str | None) -> list[str] | Literal["auto"]:
        """Normalize `scrape_methods` into the form scrapeMM's retrieve() expects:
        either the literal "auto" or a non-empty list[str] of backend names."""
        if methods is None or methods == "auto" or methods == ["auto"]:
            return "auto"
        if isinstance(methods, str):
            return [methods]
        methods = [m for m in methods if m]
        return methods or "auto"

    def _resolve_serpapi_key(self, key: str | None) -> str | None:
        """Resolve SerpAPI key from parameter, config, or environment."""
        if key:
            return key
        if VERITAS_SERPAPI_KEY:
            return VERITAS_SERPAPI_KEY
        return os.environ.get("SERPAPI_API_KEY")

    def _format_date_for_serper(self, before_date: datetime | date | str) -> str:
        """
        Format date for Serper.dev's tbs parameter.
        Format: cdr:1,cd_max:MM/DD/YYYY
        """
        if isinstance(before_date, str):
            dt = datetime.fromisoformat(before_date.replace("Z", "+00:00"))
        elif isinstance(before_date, date) and not isinstance(before_date, datetime):
            dt = datetime.combine(before_date, datetime.min.time())
        else:
            dt = before_date

        return f"cdr:1,cd_max:{dt.month}/{dt.day}/{dt.year}"

    async def _scrape_url_async(self, url: str) -> str | None:
        """Scrape a single URL using scrapeMM."""
        try:
            response: ScrapingResponse = await retrieve(
                url,
                show_progress=False,
                format="multimodal_sequence",
                methods=self.scrape_methods,
            )

            if response.successful and response.content:
                # Convert MultimodalSequence to string (markdown-like format)
                content = str(response.content)
                if len(content) > self.max_content_length:
                    content = content[:self.max_content_length] + "\n\n[Content truncated...]"
                return content
        except Exception:
            # Silently fail - content scraping is optional
            pass
        return None

    async def _scrape_urls_async(self, urls: list[str]) -> list[str | None]:
        """Scrape multiple URLs concurrently using scrapeMM."""
        if not urls:
            return []

        try:
            responses: list[ScrapingResponse] = await retrieve(
                urls,
                show_progress=False,
                format="multimodal_sequence",
                methods=self.scrape_methods,
            )

            results = []
            for response in responses:
                if response.successful and response.content:
                    content = str(response.content)
                    if len(content) > self.max_content_length:
                        content = content[:self.max_content_length] + "\n\n[Content truncated...]"
                    results.append(content)
                else:
                    results.append(None)
            return results
        except Exception:
            # If batch scraping fails, return None for all
            return [None] * len(urls)

    def _scrape_url_scrapemm(self, url: str) -> str | None:
        """Scrape a single URL using scrapeMM (sync wrapper)."""
        return _run_async(self._scrape_url_async(url))

    def _scrape_urls_scrapemm(self, urls: list[str]) -> list[str | None]:
        """Scrape multiple URLs using scrapeMM (sync wrapper)."""
        return _run_async(self._scrape_urls_async(urls))

    # -------------------------------------------------------------------------
    # Lite scraping methods (static HTTP requests + BeautifulSoup)
    # -------------------------------------------------------------------------

    def _get_session(self) -> requests.Session:
        """Get or create a requests session for connection pooling."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            })
        return self._session

    def _extract_text_from_html(self, html: str) -> str:
        """Extract readable text from HTML using BeautifulSoup."""
        soup = BeautifulSoup(html, "html.parser")

        # Remove script, style, nav, footer, header elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()

        # Get text with some structure preserved
        text_parts = []
        for element in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th", "blockquote"]):
            text = element.get_text(strip=True)
            if text and len(text) > 20:  # Skip very short fragments
                text_parts.append(text)

        return "\n\n".join(text_parts)

    def _scrape_url_lite(self, url: str) -> str | None:
        """Scrape a single URL using simple HTTP request + BeautifulSoup."""
        try:
            session = self._get_session()
            response = session.get(url, timeout=10, allow_redirects=True)
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("Content-Type", "").lower()
            if "html" not in content_type:
                return None  # Skip non-HTML (PDFs, images, etc.)

            content = self._extract_text_from_html(response.text)
            if content:
                if len(content) > self.max_content_length:
                    content = content[:self.max_content_length] + "\n\n[Content truncated...]"
                return content
        except Exception:
            # Silently fail - content scraping is optional
            pass
        return None

    def _scrape_urls_lite(self, urls: list[str]) -> list[str | None]:
        """Scrape multiple URLs in parallel using simple HTTP requests."""
        if not urls:
            return []

        results = [None] * len(urls)
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_idx = {
                executor.submit(self._scrape_url_lite, url): i
                for i, url in enumerate(urls)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    pass
        return results

    # -------------------------------------------------------------------------
    # Unified scraping interface
    # -------------------------------------------------------------------------

    def _scrape_urls(self, urls: list[str]) -> list[str | None]:
        """Scrape URLs using the configured scrape_mode."""
        if self.scrape_mode == "none":
            return [None] * len(urls)
        elif self.scrape_mode == "lite":
            return self._scrape_urls_lite(urls)
        elif self.scrape_mode == "scrapemm":
            return self._scrape_urls_scrapemm(urls)
        else:
            return [None] * len(urls)

    def search(
        self,
        query: str,
        before_date: datetime | date | str | None = None,
        num_results: int = 10,
        max_scrape: int = 3,
        scrape_content: bool = True,
        language: str = "en",
        country: str = "us",
    ) -> SearchResponse:
        """
        Perform a web search with date filtering and content retrieval.

        Args:
            query: The search query.
            before_date: Only return results from before this date.
            num_results: Maximum number of results to return (before scraping).
            max_scrape: Maximum number of top results to scrape for full content.
            scrape_content: Whether to fetch full page content (uses configured scrape_mode).
            language: Language code for search results.
            country: Country code for search results.

        Returns:
            SearchResponse with results, content, and metadata.
        """
        if not self.serpapi_key:
            return SearchResponse(
                query=query,
                error="No Serper API key configured. Set serperapi_key in config/globals.yaml or SERPER_API_KEY env var."
            )

        # Build Serper.dev request
        headers = {
            "X-API-KEY": self.serpapi_key,
            "Content-Type": "application/json",
        }

        payload = {
            "q": query,
            "num": min(num_results * 2, 20),  # Request extra to account for filtering
            "hl": language,
            "gl": country,
        }

        if before_date:
            payload["tbs"] = self._format_date_for_serper(before_date)

        # Retry logic for transient network errors
        import time
        max_retries = 3
        data = None

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "https://google.serper.dev/search",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
                if response.status_code == 402:
                    raise SerperCreditsExhaustedError(
                        "Serper API credits exhausted (HTTP 402). Terminating run."
                    )
                response.raise_for_status()
                data = response.json()
                break  # Success, exit retry loop

            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))  # Backoff: 1s, 2s
                    continue
                # All retries failed
                error_detail = str(e)
                return SearchResponse(query=query, error=f"Search failed after {max_retries} retries: {error_detail}")

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                    continue
                return SearchResponse(query=query, error="Search request timed out after retries")

            except requests.exceptions.RequestException as e:
                error_detail = str(e)
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_detail = f"{str(e)} - Response: {e.response.text[:500]}"
                    except Exception:
                        pass
                return SearchResponse(query=query, error=f"Search request failed: {error_detail}")

            except Exception as e:
                return SearchResponse(query=query, error=f"Unexpected error: {str(e)}")

        if data is None:
            return SearchResponse(query=query, error="Search failed: no response received")

        if "error" in data:
            message = data.get("message", str(data["error"]))
            if "credit" in message.lower() or "insufficient" in message.lower():
                raise SerperCreditsExhaustedError(
                    f"Serper API credits exhausted: {message}. Terminating run."
                )
            return SearchResponse(query=query, error=message)

        organic_results = data.get("organic", [])

        # Build results
        results = []

        for item in organic_results:
            url = item.get("link", "")

            results.append(SearchResult(
                title=item.get("title", ""),
                url=url,
                snippet=item.get("snippet", ""),
                date=item.get("date"),
                source=item.get("source"),
                content=None,  # Will be filled by scraping
            ))

            if len(results) >= num_results:
                break

        # Scrape content for top results only (to limit API calls)
        if scrape_content and results:
            urls_to_scrape = [result.url for result in results[:max_scrape]]
            contents = self._scrape_urls(urls_to_scrape)
            for i, content in enumerate(contents):
                if content:
                    results[i].content = content

        return SearchResponse(
            query=query,
            results=results,
            total_results=data.get("search_information", {}).get("total_results", len(results)),
        )

    def format_results_for_llm(self, response: SearchResponse, include_content: bool = True) -> str:
        """Format search results as a string for inclusion in LLM context."""
        if response.error:
            return f"Search error: {response.error}"

        if not response.results:
            return f"No results found for query: {response.query}"

        lines = [f"Search results for: {response.query}\n"]

        for i, result in enumerate(response.results, 1):
            lines.append(f"{'='*60}")
            lines.append(f"Result {i}: {result.title}")
            lines.append(f"URL: {result.url}")
            if result.date:
                lines.append(f"Date: {result.date}")

            if include_content and result.content:
                lines.append(f"\n--- Page Content for {result.url} ---")
                lines.append(result.content)
                lines.append(f"\n--- End of page content for {result.url} ---")
            else:
                lines.append(f"Summary: {result.snippet}")

            lines.append("")

        return "\n".join(lines)


# Tool definitions for function calling
OPENAI_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information and retrieve page content. Results are limited to content published before the claim date.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant information about the claim."
                }
            },
            "required": ["query"]
        }
    }
}

ANTHROPIC_SEARCH_TOOL = {
    "name": "web_search",
    "description": "Search the web for information and retrieve page content. Results are limited to content published before the claim date.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant information about the claim."
            }
        },
        "required": ["query"]
    }
}

GEMINI_SEARCH_TOOL_DECLARATION = {
    "name": "web_search",
    "description": "Search the web for information and retrieve page content. Results are limited to content published before the claim date.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant information about the claim."
            }
        },
        "required": ["query"]
    }
}


def get_search_tool_schema() -> dict:
    """Get the JSON schema for the search tool (OpenAI format)."""
    return OPENAI_SEARCH_TOOL


def create_search_service(
    serpapi_key: str | None = None,
    max_content_length: int = 8000,
    scrape_mode: ScrapeMode = "lite",
    scrape_methods: list[str] | str | None = "firecrawl",
) -> SearchService:
    """Factory function to create a SearchService instance."""
    return SearchService(
        serpapi_key=serpapi_key,
        max_content_length=max_content_length,
        scrape_mode=scrape_mode,
        scrape_methods=scrape_methods,
    )
