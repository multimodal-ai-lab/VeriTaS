"""
Retrieval tools for the fact-checking baseline.

The model verifies a claim with two tools:
- web_search: Google results (Serper, via the scrapeMM server) published before the
  claim date. Returns titles, URLs, dates and snippets only.
- fetch_url: The text content of a web page (via the scrapeMM server), truncated to
  MAX_FETCH_CHARS characters.

scrapeMM is the client of a scrapeMM server. Its URL and API key are read from the
`scrapemm` section of config.yaml (see veritas/__init__.py), or else from the
SCRAPEMM_API_URL / SCRAPEMM_API_KEY environment variables.

Cutoff semantics: the claim's own day is excluded, not just capped. Search APIs filter
at day granularity with an inclusive end date, so a cutoff on the claim day itself would
admit content published hours *after* the claim. scrapeMM's `SerperQuery(before=day)`
leaves out the given day and everything after it, so the claim day is passed as `before`
as is (see `build_search_query`).
"""

import asyncio
import atexit
import random
import threading
import time
from datetime import date, datetime
from typing import Iterable, Literal

from scrapemm import QuotaExceededError, RateLimitError, ServerError, retrieve
from scrapemm.common.outcome import decisive_error
from scrapemm.search import SerperOrganicResult, SerperQuery, search

try:
    import veritas  # noqa: F401  # Configures the scrapeMM client from config.yaml
except ImportError:
    pass

MAX_FETCH_CHARS = 25_000  # Maximum characters of page content returned by fetch_url
DEFAULT_MAX_SEARCHES = 5  # Maximum web_search calls per claim
DEFAULT_MAX_FETCHES = 5  # Maximum fetch_url calls per claim
NUM_SEARCH_RESULTS = 10
SEARCH_RETRIES = 3

ScrapeMethods = list[str] | Literal["auto"]


class SearchQuotaExceededError(BaseException):
    """Raised when the search provider's credits are used up. Inherits BaseException so it
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


# =============================================================================
# Tool definitions
# =============================================================================

WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": (
        "Returns title, URL, publication date and a short preview of the top ten Google search results for a query."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query to search for.",
            }
        },
        "required": ["query"],
    },
}

FETCH_URL_TOOL = {
    "name": "fetch_url",
    "description": (
        f"Returns the content of a web page in a readable markdown format. The content is truncated to {MAX_FETCH_CHARS}."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The full URL of the web page to fetch.",
            }
        },
        "required": ["url"],
    },
}

TOOL_DEFINITIONS = [WEB_SEARCH_TOOL, FETCH_URL_TOOL]


def openai_tools() -> list[dict]:
    """The tools in OpenAI (chat completions) format."""
    return [{"type": "function", "function": tool} for tool in TOOL_DEFINITIONS]


def anthropic_tools() -> list[dict]:
    """The tools in Anthropic format."""
    return [
        {"name": tool["name"], "description": tool["description"], "input_schema": tool["parameters"]}
        for tool in TOOL_DEFINITIONS
    ]


# =============================================================================
# Helpers
# =============================================================================

def normalize_scrape_methods(methods: list[str] | str | None) -> ScrapeMethods:
    """Normalize `methods` into the form scrapeMM's retrieve() expects:
    either the literal "auto" or a non-empty list[str] of backend names."""
    if methods is None or methods == "auto" or methods == ["auto"]:
        return "auto"
    if isinstance(methods, str):
        return [methods]
    methods = [m for m in methods if m]
    return methods or "auto"


def parse_claim_day(claim_date: str | datetime | date | None) -> date | None:
    """Return the calendar day a claim was made on, or None if no date is given.

    Raises ValueError for a date that cannot be parsed, so that a malformed date never
    silently degrades into an unfiltered search.
    """
    if claim_date is None or claim_date == "":
        return None
    if isinstance(claim_date, datetime):
        # Must precede the `date` branch: datetime is a subclass of date.
        return claim_date.date()
    if isinstance(claim_date, date):
        return claim_date
    if isinstance(claim_date, str):
        return datetime.fromisoformat(claim_date.strip().replace("Z", "+00:00")).date()
    raise ValueError(f"Unsupported claim date: {claim_date!r}")


def build_search_query(query: str, claim_date: str | datetime | date | None) -> SerperQuery:
    """Build the Serper query for a web_search call.

    The claim day itself is passed as `before`: scrapeMM leaves out that day and
    everything after it (Google's `cd_max` becomes the day before the claim).
    """
    return SerperQuery(
        q=query,
        num=NUM_SEARCH_RESULTS,
        hl="en",
        gl="us",
        before=parse_claim_day(claim_date),
    )


def format_search_results(query: str, results: list[SerperOrganicResult]) -> str:
    """Format web_search results for the model."""
    if not results:
        return f"No results found for query: {query}"

    lines = [f"Search results for: {query}", ""]
    for i, result in enumerate(results, 1):
        lines.append(f"[{i}] {result.title or '(no title)'}")
        lines.append(f"URL: {result.link}")
        if result.date:
            lines.append(f"Date: {result.date}")
        if result.snippet:
            lines.append(f"Snippet: {result.snippet}")
        lines.append("")
    return "\n".join(lines).rstrip()


def format_page(url: str, content: str, max_chars: int = MAX_FETCH_CHARS) -> str:
    """Format fetched page content for the model, truncated to `max_chars` characters."""
    if len(content) > max_chars:
        content = content[:max_chars] + "\n\n[Content truncated...]"
    return f"Content of {url}:\n\n{content}"


# =============================================================================
# Tool execution
# =============================================================================

class ToolSession:
    """Executes the tool calls of a single fact-check.

    Holds the per-claim state (call counts, citations), so create one per claim:
    providers are shared across worker threads.
    """

    def __init__(
        self,
        claim_date: str | datetime | date | None = None,
        scrape_methods: list[str] | str | None = "auto",
        max_searches: int = DEFAULT_MAX_SEARCHES,
        max_fetches: int = DEFAULT_MAX_FETCHES,
        max_fetch_chars: int = MAX_FETCH_CHARS,
    ):
        """
        Args:
            claim_date: Date of the claim. web_search only returns content published
                        before this day. Raises ValueError if it cannot be parsed.
            scrape_methods: Which scrapeMM backends fetch_url uses, in order. A subset of
                            {"integrations", "browser", "firecrawl", "decodo"}, or "auto"
                            to let the scrapeMM server choose per domain.
            max_searches: Maximum number of web_search calls.
            max_fetches: Maximum number of fetch_url calls.
            max_fetch_chars: Maximum characters of page content returned by fetch_url.
        """
        self.claim_day = parse_claim_day(claim_date)
        if self.claim_day is None:
            print("    Warning: No claim date given, web_search results are not date-filtered")
        self.scrape_methods = normalize_scrape_methods(scrape_methods)
        self.max_searches = max_searches
        self.max_fetches = max_fetches
        self.max_fetch_chars = max_fetch_chars

        self.n_searches = 0
        self.n_fetches = 0
        self.citations: list[str] = []  # Search result URLs and fetched URLs, in order

    def call(self, name: str, args: dict | None) -> str:
        """Execute a tool call and return the result text for the model."""
        if not isinstance(args, dict):
            return f"Error: Invalid arguments for {name}. Expected a JSON object."
        if name == "web_search":
            return self.web_search(str(args.get("query") or ""))
        if name == "fetch_url":
            return self.fetch_url(str(args.get("url") or ""))
        return f"Error: Unknown tool '{name}'. Available tools: web_search, fetch_url."

    def web_search(self, query: str) -> str:
        """Search the web for content published before the claim date."""
        if not query.strip():
            return "Error: The search query must not be empty."
        if self.n_searches >= self.max_searches:
            return (f"Search limit of {self.max_searches} web_search calls reached. "
                    f"Continue with the information gathered so far.")
        self.n_searches += 1

        try:
            response = self._search(query)
        except Exception as e:
            print(f"    Search error: {type(e).__name__}: {e}")
            return f"Search failed: {e}"

        # Only organic results: the knowledge graph, answer box, top stories etc. are not
        # date-filtered by Google and would leak information from after the claim date.
        results = [r for r in response.organic if r.link]
        self._cite(r.link for r in results)
        print(f"    web_search: {query!r} -> {len(results)} results")
        return format_search_results(query, results)

    def _search(self, query: str):
        """Run the search on the scrapeMM server, retrying transient errors."""
        search_query = build_search_query(query, self.claim_day)
        for attempt in range(SEARCH_RETRIES):
            try:
                return _run_async(search(search_query))
            except QuotaExceededError as e:
                raise SearchQuotaExceededError(f"Search quota exceeded: {e}. Terminating run.") from e
            except (RateLimitError, TimeoutError, ServerError):
                if attempt == SEARCH_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt + random.uniform(0, 1))

    def fetch_url(self, url: str) -> str:
        """Fetch the text content of a web page."""
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            return f"Error: '{url}' is not a valid URL. Provide a full URL starting with http:// or https://."
        if self.n_fetches >= self.max_fetches:
            return (f"Fetch limit of {self.max_fetches} fetch_url calls reached. "
                    f"Continue with the information gathered so far.")
        self.n_fetches += 1

        try:
            response = _run_async(retrieve(
                url,
                show_progress=False,
                output_format="markdown",
                methods=self.scrape_methods,
            ))
        except Exception as e:
            print(f"    fetch_url: {url} -> error: {type(e).__name__}: {e}")
            return f"Could not retrieve {url}: {e}"

        content = response.content.markdown if response.success else None
        if not content:
            error = decisive_error(response.errors)
            reason = f"{type(error[1]).__name__}: {error[1]}" if error else "No content retrieved."
            print(f"    fetch_url: {url} -> failed ({reason})")
            return f"Could not retrieve {url}: {reason}"

        self._cite([url])
        print(f"    fetch_url: {url} -> {len(content)} chars")
        return format_page(url, content, self.max_fetch_chars)

    def _cite(self, urls: Iterable[str]):
        for url in urls:
            if url not in self.citations:
                self.citations.append(url)
