"""
Retrieval tools for the fact-checking baseline.

The model verifies a claim with two tools:
- web_search: Google results (Serper, via the scrapeMM server) published before the
  claim date. Returns titles, URLs, dates and snippets only.
- fetch_url: The main content of a web page, without navigation, headers, footers and
  the like (via the scrapeMM server): its text, truncated to MAX_FETCH_CHARS characters,
  and some of its images and videos in their places (see MediaLimits).
  Only URLs the model has seen, in web_search results or in fetched pages, can be
  fetched: any other would bypass the cutoff (a search engine's results page, an
  archived copy, or a fact-check the model remembers).

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
import re
import tempfile
import threading
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Iterator, Literal
from urllib.parse import unquote, urlsplit

from ezmm import set_ezmm_path
from ezmm.common.registry import item_registry
from scrapemm import QuotaExceededError, RateLimitError, ServerError, retrieve
from scrapemm.common.outcome import decisive_error
from scrapemm.search import SerperOrganicResult, SerperQuery, search

from .media import ItemSequence, MediaLimits, is_readable_image, render
from .parsing import extract_urls, remove_links

try:
    import veritas  # noqa: F401  # Configures the scrapeMM client from config.yaml
except ImportError:
    pass

MAX_FETCH_CHARS = 25_000  # Maximum characters of page content returned by fetch_url
DEFAULT_MAX_SEARCHES = 40  # Maximum web_search calls per claim
DEFAULT_MAX_FETCHES = 40  # Maximum fetch_url calls per claim
MAX_VIDEO_BYTES = 50 * 1024 * 1024  # Larger videos are downloaded at a lower resolution
# Seconds a fetch_url call may take. A fetch can hang on the scrapeMM server, whose heartbeats
# keep the client from timing out, and with it the fact-check and the end of the run.
FETCH_TIMEOUT = 180
NUM_SEARCH_RESULTS = 10
MAX_SEARCH_PAGES = 3  # Pages of Google results web_search requests at most to get NUM_SEARCH_RESULTS unblocked ones
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
# The fetched pages' media
# =============================================================================

class _PageMediaFiles:
    """The files of the fetched pages' media, as scrapeMM downloads them into ezMM's store, and
    how many running fact-checks use each. Parallel fact-checks can fetch pages that share a
    file (a site's logo, say), so a file is deleted only once none uses it any more, and only
    in a temporary store (see temporary_media_store): never a persistent store's files, nor the
    claim's own, which ezMM gives a page's identical medium."""

    def __init__(self):
        self._lock = threading.Lock()
        self._users: Counter[Path] = Counter()
        self.temporary_store: Path | None = None  # Set by temporary_media_store

    def acquire(self, paths: set[Path]):
        with self._lock:
            self._users.update(paths)

    def release(self, paths: set[Path]):
        """Stop using the files, deleting those in the temporary store that no fact-check uses."""
        with self._lock:  # Also while deleting, so that no fact-check takes up a file being deleted
            self._users.subtract(paths)
            for path in paths:
                if self._users[path] <= 0:
                    del self._users[path]
                    if self.temporary_store and path.is_relative_to(self.temporary_store):
                        path.unlink(missing_ok=True)


_page_media_files = _PageMediaFiles()


@contextmanager
def temporary_media_store() -> Iterator[Path]:
    """While the block runs, keep ezMM's items, among them the media scrapeMM downloads for
    fetched pages, in a temporary folder on local disk (in $TMPDIR, else /tmp), and delete each
    fact-check's page media when it ends. The folder is deleted afterwards.

    ezMM's store is an SQLite database, and scrapeMM registers every medium of a page in it, one
    write transaction each. On a network file system (the repository's .ezmm/ on /mnt/vast),
    each of them waits for a lock: fetches took minutes. Slurm jobs on several nodes sharing the
    database could also corrupt it.
    """
    previous = item_registry.path
    with tempfile.TemporaryDirectory(prefix="ezmm_") as store:
        _use_store(Path(store))
        _page_media_files.temporary_store = item_registry.path
        try:
            yield item_registry.path
        finally:
            _page_media_files.temporary_store = None
            _use_store(previous)


def _use_store(path: Path):
    """Point ezMM's registry, which scrapeMM's client uses too, to the store at `path`."""
    item_registry.close()  # ezMM can't move a registry with open connections
    item_registry.clear_cache()
    set_ezmm_path(path)


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
        f"Returns the content of a web page truncated to {MAX_FETCH_CHARS} characters."
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


def responses_tools() -> list[dict]:
    """The tools in OpenAI Responses API format. Not strict, as in the other APIs: the
    Responses API makes functions strict by default, which our schemas don't satisfy."""
    return [{"type": "function", "strict": False, **tool} for tool in TOOL_DEFINITIONS]


def chat_completions_tools() -> list[dict]:
    """The tools in OpenAI chat completions format (as served by vLLM)."""
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


def build_search_query(query: str, cutoff_date: str | datetime | date | None, page: int = 1) -> SerperQuery:
    """Build the Serper query for a page of a web_search call's Google results.

    The cutoff day itself is passed as `before`: scrapeMM leaves out that day and
    everything after it (Google's `cd_max` becomes the day before the cutoff).
    """
    return SerperQuery(
        q=query,
        num=NUM_SEARCH_RESULTS,
        page=page,
        hl="en",
        gl="us",
        before=parse_claim_day(cutoff_date),
    )


def format_search_results(query: str, results: list[SerperOrganicResult]) -> str:
    """Format web_search results for the model."""
    if not results:
        return f"**No results found for query: {query}**"

    lines = [f"**Search results for: {query}**", ""]
    for i, result in enumerate(results, 1):
        lines.append(f"[{i}] {result.title or '(no title)'}")
        lines.append(f"URL: {result.link}")
        if result.date:
            lines.append(f"Date: {result.date}")
        if result.snippet:
            lines.append(f"Snippet: {result.snippet}")
        lines.append("")
    return "\n".join(lines).rstrip()+f"\n**End of search results for: {query}**"


def _page_of(url: str) -> str:
    """The normalized URL without its query (see normalize_url): blocked URLs are blocked
    whatever tracking parameters they come with."""
    return normalize_url(url).split("?", 1)[0]


def _ascii_host(host: str) -> str:
    """A host name in lower case and ASCII, as the publishers' fact-check scopes have it
    ("XN--80AA.example.org" or its Unicode form -> "xn--80aa.example.org")."""
    try:
        return host.lower().encode("idna").decode("ascii")
    except UnicodeError:
        return host.lower()


def _scope_of(scope: str) -> tuple[str, str]:
    """A publisher's fact-check scope, or a blocked URL, as the host and the path under which it
    publishes its fact-checks, the path in lower case ("leadstories.com/hoax-alert" ->
    ("leadstories.com", "hoax-alert"), "fullfact.org" -> ("fullfact.org", ""),
    "https://www.example.org/a/b/?x=1" -> ("example.org", "a/b"))."""
    host, _, path = _page_of(scope if "://" in scope else f"https://{scope}").partition("/")
    return _ascii_host(host), path.lower()


# A host, maybe with a path, in a search result's title or snippet, with or without scheme
# ("leadstories.com/hoax-alert/2026/07/..."), but not in an e-mail address
_HOST_AND_PATH = re.compile(r"(?<![\w.@-])(?:https?://)?(?:www\.)?((?:[\w-]+\.)+[^\W\d_]{2,})(?:/([^\s\"'<>()\[\]]*))?")
# The spaces with which Google breaks URLs and slugs in titles and snippets, before or after a
# "-", "/" or "." ("https://maldita. es/desinfo/...", "2026-09-15- maryam-...")
_GOOGLE_BREAK = re.compile(r"(?<=[-/.]) | (?=[-/.])")

# The shortest slug that tells an article apart from all others
MIN_SLUG_LENGTH = 12
# The shortest piece of a slug that counts as the slug where Google cut it short in a search
# result's title or snippet ("...hontiveros-did-not-say-she-would-not-run ...")
MIN_SLUG_PIECE_LENGTH = 20


def _slug_of(url: str) -> str | None:
    """The slug of an article's URL, the last segment of its path with a letter in it
    ("https://www.dogrulukpayi.com/dogruluk-kontrolu/ucuz-mama-kedileri-gay-yapar-mi" ->
    "ucuz-mama-kedileri-gay-yapar-mi"). None if it is too short to tell the article apart, as
    "n-31898"."""
    _, *segments = _page_of(url).split("/")
    slug = next((segment for segment in reversed(segments) if re.search(r"[^\W\d_]", segment)), "")
    return slug if len(slug) >= MIN_SLUG_LENGTH else None


def normalize_url(url: str) -> str:
    """A URL as fetch_url compares it with the URLs the model has seen: without scheme, "www.",
    trailing slash and fragment, host in lower case and percent-encoding decoded
    ("https://www.Example.org/a%20b/#top" -> "example.org/a b"). The model may write a URL it
    saw in any of these forms. "" if the URL has no host."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:  # Malformed, e.g. "https://[x"
        return ""
    host = parts.netloc.lower().removeprefix("www.")
    if not host:
        return ""
    query = f"?{unquote(parts.query)}" if parts.query else ""
    return f"{host}{unquote(parts.path).rstrip('/')}{query}"


# =============================================================================
# Tool execution
# =============================================================================

@dataclass
class ToolResult:
    """What a tool call returns to the model, as an ezMM sequence: text and, of a fetched
    page, the images and videos shown, in their places. Text alone may be given as a string."""

    content: ItemSequence | str
    # The page's media shown, for the trace: their reference, kind and source URL (their files are
    # deleted after the fact-check in a temporary store)
    media: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.content, str):
            self.content = [self.content]

    @property
    def text(self) -> str:
        """The content as text, each image and video replaced by its reference ("<image:40>")."""
        return render(self.content)


class ToolSession:
    """Executes the tool calls of a single fact-check.

    Holds the per-claim state (call counts, the URLs the model saw and retrieved), so
    create one per claim: providers are shared across worker threads.
    """

    def __init__(
        self,
        claim_date: str | datetime | date | None = None,
        scrape_methods: list[str] | str | None = "auto",
        max_searches: int = DEFAULT_MAX_SEARCHES,
        max_fetches: int = DEFAULT_MAX_FETCHES,
        max_fetch_chars: int = MAX_FETCH_CHARS,
        media_limits: MediaLimits = MediaLimits(),
        blocked_urls: Iterable[str] = (),
        claim_id: int | str | None = None,
        cutoff_date: str | datetime | date | None = None,
        blocked_scopes: Iterable[str] = (),
        claim_images: int = 0,
    ):
        """
        Args:
            claim_date: Date of the claim, which the prompt gives. Raises ValueError if it cannot
                        be parsed, as does cutoff_date.
            scrape_methods: Which scrapeMM backends fetch_url uses, in order. A subset of
                            {"integrations", "browser", "firecrawl", "decodo"}, or "auto"
                            to let the scrapeMM server choose per domain.
            max_searches: Maximum number of web_search calls.
            max_fetches: Maximum number of fetch_url calls.
            max_fetch_chars: Maximum characters of page content returned by fetch_url.
            media_limits: How many images and videos of the fetched pages are shown.
            blocked_urls: URLs the tools never show or fetch, and whose links fetched pages
                          lose: the claim's own fact-checks, which would give the answer away.
                          They are blocked like blocked_scopes, with all pages under them
                          (".../amp"), and so is any URL with their slug in its path, on any
                          site (see _slug_of): sites serve an article under several paths, and
                          archives and other sites copy it. A search result whose title or
                          snippet has the slug, or a piece of it, is dropped too. Google's date
                          filter lets them through at times, and fact-checkers' pages link to
                          them in their current versions.
            claim_id: ID of the claim, which tags the session's log lines.
            cutoff_date: web_search only returns content published before this day: the day the
                         claim's fact-check was published. The claim date if not given.
            blocked_scopes: The fact-check scopes of fact-checkers: the hosts, or hosts and
                            paths, under which they publish fact-checks
                            ("leadstories.com/hoax-alert"), e.g. those of all IFCN signatories
                            and EFCSN members. URLs in them, whatever their scheme, "www.", query
                            and fragment, are blocked: search results that are or show them are
                            dropped, fetch_url refuses them and fetched pages lose their links.
                            The rest of the sites is not blocked.
            claim_images: The images the claim's media take in every request (its images, and
                          the frames of its videos for models without native video), which
                          count toward media_limits.request_images.
        """
        self.log_prefix = f"[{claim_id}] " if claim_id is not None else ""
        self.claim_day = parse_claim_day(claim_date)
        self.cutoff_day = parse_claim_day(cutoff_date) if cutoff_date else self.claim_day
        if self.cutoff_day is None:
            self.log("Warning: No claim or cutoff date given, web_search results are not date-filtered")
        self.scrape_methods = normalize_scrape_methods(scrape_methods)
        self.max_searches = max_searches
        self.max_fetches = max_fetches
        self.max_fetch_chars = max_fetch_chars
        self.media_limits = media_limits
        self.claim_images = claim_images
        # Host -> the paths at and under which all is blocked ("" for all of the host), each with
        # the rule that blocks it, for the log (see blocked_by)
        self.blocked_scopes: dict[str, dict[str, str]] = {}
        for entry, rule in [*((url, f"at or under the claim's fact-check {url}") for url in blocked_urls),
                            *((scope, f"in the fact-check scope {scope}") for scope in blocked_scopes)]:
            host, path = _scope_of(entry)
            if host:
                self.blocked_scopes.setdefault(host, {}).setdefault(path, rule)
        self.blocked_slugs = {_slug_of(url) for url in blocked_urls} - {None}

        self.n_searches = 0
        self.n_fetches = 0
        self.encountered_urls: list[str] = []  # URLs of the web_search results, in order
        self.retrieved_urls: list[str] = []  # URLs successfully fetched with fetch_url, in order
        self.linked_urls: list[str] = []  # URLs in the fetched content shown to the model, in order
        self.shown_media: list[dict] = []  # The fetched pages' images and videos shown to the model
        # The web_search results dropped and the fetch_url calls refused because they are or show
        # a blocked URL, in order: {"tool", "url", "rule"} (see blocked_by), and "query" for searches
        self.blocked: list[dict] = []
        self.media_files: set[Path] = set()  # The files of all media of the fetched pages, shown or not

    def log(self, message: str):
        """Print a log line, tagged with the claim it belongs to: claims run in parallel."""
        print(f"    {self.log_prefix}{message}")

    def close(self):
        """End the session: the files of the fetched pages' media get deleted, unless another
        fact-check uses them too (in a temporary store only, see temporary_media_store)."""
        _page_media_files.release(self.media_files)
        self.media_files = set()

    def call(self, name: str, args: dict | None) -> ToolResult:
        """Execute a tool call and return its result for the model."""
        if not isinstance(args, dict):
            return ToolResult(f"Error: Invalid arguments for {name}. Expected a JSON object.")
        if name == "web_search":
            return ToolResult(self.web_search(str(args.get("query") or "")))
        if name == "fetch_url":
            return self.fetch_url(str(args.get("url") or ""))
        return ToolResult(f"Error: Unknown tool '{name}'. Available tools: web_search, fetch_url.")

    def web_search(self, query: str) -> str:
        """Search the web for content published before the cutoff date."""
        if not query.strip():
            return "Error: The search query must not be empty."
        if self.n_searches >= self.max_searches:
            return (f"Search limit of {self.max_searches} web_search calls reached. "
                    f"Continue with the information gathered so far.")
        self.n_searches += 1

        try:
            results, n_blocked, n_pages = self._search(query)
        except Exception as e:
            self.log(f"Search error: {type(e).__name__}: {e}")
            return f"Search failed: {e}"

        _add_unique(self.encountered_urls, (r.link for r in results))
        blocked = f" ({n_blocked} blocked, {n_pages} pages)" if n_blocked else ""
        self.log(f"web_search: {query!r} -> {len(results)} results{blocked}")
        return format_search_results(query, results)

    def _search(self, query: str, want: int = NUM_SEARCH_RESULTS,
                max_pages: int = MAX_SEARCH_PAGES) -> tuple[list[SerperOrganicResult], int, int]:
        """The first `want` results for `query` that neither are nor show a blocked URL, from as
        many pages of Google results as it takes, up to `max_pages`: blocked results leave a page
        with fewer. A page with fewer than NUM_SEARCH_RESULTS results is not always the last, and
        later pages repeat some results, so paging stops at a page that brings no new result.

        Returns the results, the number of blocked results and the number of pages requested.
        """
        results, seen, n_blocked = [], set(), 0
        for page in range(1, max_pages + 1):
            # Only organic results: the knowledge graph, answer box, top stories etc. are not
            # date-filtered by Google and would leak information from after the claim date.
            new = [r for r in self._search_page(query, page).organic if r.link and r.link not in seen]
            if not new:
                return results, n_blocked, page
            seen.update(r.link for r in new)
            # Also the results whose title or snippet shows a blocked URL: mostly the
            # fact-checkers' own posts, which give a fact-check's headline with its URL
            allowed = []
            for r in new:
                if reason := self.blocked_by(r.link):
                    rule = f"a URL {reason}"
                elif reason := self._blocked_in_preview(f"{r.title or ''} {r.snippet or ''}"):
                    rule = f"whose title or snippet {reason}"
                else:
                    allowed.append(r)
                    continue
                self.log(f"web_search: {query!r}: dropped {r.link}, {rule}")
                self.blocked.append({"tool": "web_search", "query": query, "url": r.link, "rule": rule})
            n_blocked += len(new) - len(allowed)
            results += allowed
            if len(results) >= want:
                return results[:want], n_blocked, page
        return results, n_blocked, max_pages

    def _find_blocked_slug(self, text: str) -> tuple[str, re.Match] | None:
        """Where `text` has the slug of a blocked article, in a URL or not ("fact-checks/<slug>"),
        also where Google broke it with a space before or after a "-", "/" or "." ("2026-09-15-
        maryam-..."): `text` decoded and with these spaces taken out, and the slug's match in it.
        None if it has none."""
        text = _GOOGLE_BREAK.sub("", unquote(text))
        for slug in self.blocked_slugs:
            if match := re.search(r"(?<![\w-])" + re.escape(slug) + r"(?![\w-])", text):
                return text, match
        return None

    def _slug_in(self, text: str) -> str | None:
        """Which slug of a blocked article `text` has (see _find_blocked_slug), or which piece of
        one, where Google cut it short at either end in a search result's title or snippet: a word
        or path segment of `text` of at least MIN_SLUG_PIECE_LENGTH characters, as the log gives it
        ("'...-is-false.html', a piece of the slug ..."). Whether a piece is of the blocked article
        or of another one with a similar slug is unknown. None if it has none."""
        if found := self._find_blocked_slug(text):
            return f"the slug {found[1].group(0)} of the claim's fact-check"
        text = _GOOGLE_BREAK.sub("", unquote(text))
        parts = (part.strip(".,;:!()[]\"'·") for part in re.split(r"[\s/?#…]+|\.{2,}", text))
        return next((f"{part!r}, a piece of the slug {slug} of the claim's fact-check"
                     for part in parts for slug in self.blocked_slugs
                     if len(part) >= MIN_SLUG_PIECE_LENGTH and part in slug), None)

    def _blocked_in_preview(self, preview: str) -> str | None:
        """What blocked a search result's title or snippet shows, for the log ("shows
        leadstories.com/hoax-alert/..., a URL in the fact-check scope leadstories.com/hoax-alert"):
        the slug of a blocked URL or a piece of it (see _slug_in), or a blocked URL, with or without
        scheme, also where Google broke it with spaces after or before a ".", "-" or "/"
        ("https://leadstories. com/hoax-alert/...") or cut it short: then if it is the start of a
        blocked URL or of a blocked fact-check scope ("https://leadstories.com/hoa ..."). Whether
        that is the blocked article or another one of the site is unknown. None if it shows none."""
        if slug := self._slug_in(preview):
            return f"has {slug}"
        scopes = [(f"{host}/{path}".rstrip("/"), rule)
                  for host, paths in self.blocked_scopes.items() for path, rule in paths.items()]
        # As it is and with Google's breaks taken out, which also join a sentence to the next
        text = unquote(preview)
        for text in (text, _GOOGLE_BREAK.sub("", text)):
            for m in _HOST_AND_PATH.finditer(text):
                host, path = m.group(1), (m.group(2) or "").rstrip(".,;:!?…/")
                if reason := self.blocked_by(f"https://{host}/{path}"):
                    return f"shows {host}/{path}, a URL {reason}"
                cut_short = m.group(0).endswith(("...", "…")) or re.match(r"\s?(?:\.{3}|…)", text[m.end():])
                start = f"{_ascii_host(host.lower().removeprefix('www.'))}/{path}".lower()
                if cut_short and (rule := next((rule for scope, rule in scopes if scope.startswith(start)), None)):
                    return f"shows {start}…, cut short, the start of a URL {rule}"
        return None

    def blocked_by(self, url: str) -> str | None:
        """Why `url` is blocked, as the log gives it ("in the fact-check scope fullfact.org"): it is
        at or under a blocked URL or in a blocked fact-check scope, whatever its scheme, "www.",
        query and fragment: on the same host (subdomains have scopes of their own) and at or under
        the path, in any case. Or it has the slug of a blocked URL in its path (see _slug_of), on
        any site: sites serve an article under several paths, e.g. in another section, under
        another date or on a subdomain, and archives and other sites copy it. None if it is not
        blocked."""
        host, _, path = _page_of(url).partition("/")
        if slug := next((slug for slug in self.blocked_slugs if slug in path.split("/")), None):
            return f"with the slug {slug} of the claim's fact-check in its path"
        path = path.lower()
        return next((rule for scope_path, rule in self.blocked_scopes.get(_ascii_host(host), {}).items()
                     if not scope_path or path == scope_path or path.startswith(f"{scope_path}/")), None)

    def is_blocked(self, url: str) -> bool:
        """Whether `url` is blocked (see blocked_by)."""
        return self.blocked_by(url) is not None

    def _search_page(self, query: str, page: int):
        """Get a page of Google results for `query` from the scrapeMM server, retrying transient
        errors."""
        search_query = build_search_query(query, self.cutoff_day, page)
        for attempt in range(SEARCH_RETRIES):
            try:
                return _run_async(search(search_query))
            except QuotaExceededError as e:
                raise SearchQuotaExceededError(f"Search quota exceeded: {e}. Terminating run.") from e
            except (RateLimitError, TimeoutError, ServerError):
                if attempt == SEARCH_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt + random.uniform(0, 1))

    def seen_urls(self) -> set[str]:
        """The URLs fetch_url may fetch, normalized: those of the web_search results and those
        in fetched pages."""
        return {normalize_url(url) for url in self.encountered_urls + self.linked_urls} - {""}

    def fetch_url(self, url: str) -> ToolResult:
        """Fetch the content of a web page the model has seen a link to: its text and some of
        its images and videos."""
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            return ToolResult(f"Error: '{url}' is not a valid URL. Provide a full URL starting with http:// or https://.")
        reason = self.blocked_by(url)
        if reason:
            self.blocked.append({"tool": "fetch_url", "url": url, "rule": f"a URL {reason}"})
        if reason or normalize_url(url) not in self.seen_urls():
            # A blocked URL gets the answer of any URL not seen: another answer would tell the
            # model that it guessed the claim's fact-check
            self.log(f"fetch_url: {url} -> " + (f"blocked, a URL {reason}" if reason else "rejected, URL not seen"))
            return ToolResult(f"Error: Cannot fetch {url}. fetch_url only fetches URLs that appeared in web_search "
                              f"results or in fetched pages. Use them as they appeared.")
        if self.n_fetches >= self.max_fetches:
            return ToolResult(f"Fetch limit of {self.max_fetches} fetch_url calls reached. "
                              f"Continue with the information gathered so far.")
        self.n_fetches += 1

        try:
            response = _run_async(asyncio.wait_for(retrieve(
                url,
                show_progress=False,
                output_format="multimodal",  # The text, with the page's images and videos downloaded
                methods=self.scrape_methods,
                strip=True,  # Only the main content, without navigation, footers, cookie banners etc.
                max_video_size=MAX_VIDEO_BYTES,
                downscale_videos=True,
            ), FETCH_TIMEOUT))
        except TimeoutError:
            self.log(f"fetch_url: {url} -> timed out after {FETCH_TIMEOUT} s")
            return ToolResult(f"Could not retrieve {url}: the page took too long to load.")
        except Exception as e:
            self.log(f"fetch_url: {url} -> error: {type(e).__name__}: {e}")
            return ToolResult(f"Could not retrieve {url}: {e}")

        sequence = (response.content.multimodal if response.success else None) or []
        files = {Path(part.file_path) for part in sequence if not isinstance(part, str)} - self.media_files
        _page_media_files.acquire(files)
        self.media_files |= files
        sequence, removed = self._remove_blocked_links(sequence, url)
        parts, media, n_chars = self._page_content(sequence)
        if n_chars == 0:
            error = decisive_error(response.errors)
            reason = f"{type(error[1]).__name__}: {error[1]}" if error else "No content retrieved."
            self.log(f"fetch_url: {url} -> failed ({reason})")
            return ToolResult(f"Could not retrieve {url}: {reason}")

        self.log(f"fetch_url: {url} -> {n_chars} chars, {len(media)} media shown"
                 + (f", {len(removed)} blocked links removed" if removed else ""))
        by_rule: dict[str, list[str]] = {}
        for link, rule in removed:
            by_rule.setdefault(rule, []).append(link)
        for rule, links in by_rule.items():
            links = list(dict.fromkeys(links))
            self.log(f"fetch_url: {url}: removed the links to {len(links)} URLs {rule}: {', '.join(links)}")
        self.shown_media += media
        _add_unique(self.retrieved_urls, [url])
        page_text = " ".join(part for part in parts if isinstance(part, str))
        _add_unique(self.linked_urls, extract_urls(page_text, base_url=url))
        content = [f"**Content of {url}:**\n\n", *parts, f"\n**End of content of {url}:**"]
        return ToolResult(content, media)

    def _remove_blocked_links(self, sequence: Iterable, page_url: str) -> tuple[list, list[tuple[str, str]]]:
        """A fetched page's content without its links to the blocked URLs, with their text and
        title, which may give the fact-check's verdict away (see remove_links). The text of a
        link may hold images, e.g. a teaser's, which go with it.

        Returns the content and the links removed, each as its URL and the rule that blocks it
        (see blocked_by).
        """
        parts = list(sequence)
        if not self.blocked_scopes:
            return parts, []
        # The items stand in the text as placeholders that no link target or URL takes in
        text, items = [], []
        for part in parts:
            if isinstance(part, str):
                text.append(part)
            else:
                text.append(f"<\ue000{len(items)}>")
                items.append(part)
        removed = []

        def is_blocked(url: str) -> bool:
            if reason := self.blocked_by(url):
                removed.append((url, reason))
            return reason is not None

        text, _ = remove_links("".join(text), is_blocked, base_url=page_url)
        if not removed:
            return parts, []
        pieces = re.split(r"<\ue000(\d+)>", text)  # Text, item index, text, ...
        content = [items[int(piece)] if i % 2 else piece for i, piece in enumerate(pieces) if i % 2 or piece]
        return content, removed

    def _page_content(self, sequence: Iterable) -> tuple[list, list[dict], int]:
        """A fetched page's content as the model sees it, in page order: its text, cut at
        max_fetch_chars characters (an image or video counts as long as its reference), with its
        images and videos in their places, as far as the media limits per page, per fact-check
        and per request allow. A note takes the place of the media beyond these, of media shown
        before (on this page or an earlier one) and of images that can't be read.

        Returns the parts (text and ezMM items), the media shown (for the trace) and the
        length of the whole page in characters.
        """
        limits = self.media_limits
        page_limits = {"image": limits.page_images, "video": limits.page_videos}
        total_limits = {"image": limits.total_images, "video": limits.total_videos}
        # The images each takes in a request (a video's frames: a bound for models with native video)
        request_images = {"image": 1, "video": limits.video_frames}
        parts, media = [], []
        n_chars = sum(len(part) if isinstance(part, str) else len(part.reference) for part in sequence)
        budget = self.max_fetch_chars
        for part in sequence:
            length = len(part) if isinstance(part, str) else len(part.reference)
            if length > budget:
                if isinstance(part, str):
                    parts.append(part[:budget])
                parts.append(f"\n\n[Content truncated to {self.max_fetch_chars} characters]")
                break
            budget -= length
            if isinstance(part, str):
                parts.append(part)
                continue
            shown = self.shown_media + media
            if any(m["reference"] == part.reference for m in shown):
                parts.append(f"[the {part.kind} shown above again]")
            elif (part.kind not in page_limits
                  or sum(m["kind"] == part.kind for m in media) >= page_limits[part.kind]
                  or sum(m["kind"] == part.kind for m in shown) >= total_limits[part.kind]
                  or (limits.request_images is not None
                      and self.claim_images + sum(request_images[m["kind"]] for m in shown) + request_images[part.kind]
                      > limits.request_images)):
                parts.append(f"[{part.kind} not shown]")
            elif part.kind == "image" and not is_readable_image(part.file_path, limits.page_image_side):
                parts.append("[image in a format that cannot be shown]")
            else:
                parts.append(part)
                media.append({"reference": part.reference, "kind": part.kind, "source_url": part.source_url})
        return parts, media, n_chars


def _add_unique(target: list[str], urls: Iterable[str]):
    """Append the URLs not yet in `target`, keeping their order."""
    for url in urls:
        if url not in target:
            target.append(url)
