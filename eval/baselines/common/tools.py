"""Retrieval tools for the fact-checking baseline: web_search (Google via Serper, on the
scrapeMM server) and fetch_url (a page's main content, via the scrapeMM server).

fetch_url only fetches URLs the model has seen in web_search results or fetched pages: any
other would bypass the cutoff (a search engine's results page, an archived copy, a
fact-check the model remembers).

The claim's own day is excluded from web_search, not just capped: search APIs filter at
day granularity with an inclusive end date, so a cutoff on the claim day itself would
admit content published hours *after* the claim.
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
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Iterator, Literal
from urllib.parse import unquote, urlsplit

from ezmm import set_ezmm_path
from ezmm.common.registry import item_registry
from scrapemm import QuotaExceededError, RateLimitError, ServerError, retrieve
from scrapemm.common.outcome import decisive_error
from scrapemm.search import SerperOrganicResult, SerperQuery, search

from .google_links import GoogleLinkResolver, is_google_redirect, is_hidden_redirect, plain_target
from .media import ItemSequence, MediaLimits, is_readable_image, render
from .parsing import extract_urls, remove_links

try:
    import veritas  # noqa: F401  # Configures the scrapeMM client from config.yaml
except ImportError:
    pass

MAX_FETCH_CHARS = 25_000
DEFAULT_MAX_SEARCHES = 40
DEFAULT_MAX_FETCHES = 40
MAX_VIDEO_BYTES = 50 * 1024 * 1024  # Larger videos are downscaled, not refused
# A fetch can hang on the scrapeMM server, whose heartbeats keep the client from timing out
FETCH_TIMEOUT = 180
NUM_SEARCH_RESULTS = 10
MAX_SEARCH_PAGES = 3
SEARCH_RETRIES = 3
UNRESOLVED_REDIRECT = "behind a Google redirect that could not be resolved"

ScrapeMethods = list[str] | Literal["auto"]


class SearchQuotaExceededError(BaseException):
    """Inherits BaseException to propagate past the bare `except Exception` handlers and
    terminate the run."""


class _AsyncRunner:
    """One background thread with an event loop for all async calls: a thread and loop per
    scrape exhausted file descriptors under concurrent load."""

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
        with self._start_lock:
            if self._started:
                return
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            self._started = True
            atexit.register(self._shutdown)

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _shutdown(self):
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join(timeout=5)

    def run(self, coro):
        if not self._started:
            self._start_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()


_async_runner = _AsyncRunner()


def _run_async(coro):
    return _async_runner.run(coro)


class _PageMediaFiles:
    """How many running fact-checks use each file of the fetched pages' media. Parallel
    fact-checks can fetch pages that share a file (a site's logo), so a file is deleted only
    once none uses it, and only in a temporary store (see temporary_media_store): never a
    persistent store's files, nor the claim's own, which ezMM gives a page's identical medium."""

    def __init__(self):
        self._lock = threading.Lock()
        self._users: Counter[Path] = Counter()
        self.temporary_store: Path | None = None

    def acquire(self, paths: set[Path]):
        with self._lock:
            self._users.update(paths)

    def release(self, paths: set[Path]):
        with self._lock:  # Also while deleting: no fact-check may take up a file being deleted
            self._users.subtract(paths)
            for path in paths:
                if self._users[path] <= 0:
                    del self._users[path]
                    if self.temporary_store and path.is_relative_to(self.temporary_store):
                        path.unlink(missing_ok=True)


_page_media_files = _PageMediaFiles()


@contextmanager
def temporary_media_store() -> Iterator[Path]:
    """Keep ezMM's store, with the media scrapeMM downloads for fetched pages, in a temporary
    folder on local disk while the block runs, and delete each fact-check's page media when it
    ends.

    ezMM's store is an SQLite database, and scrapeMM registers every medium of a page in it, one
    write transaction each. On a network file system (the repository's .ezmm/ on /mnt/vast),
    each waits for a lock: fetches took minutes. Slurm jobs on several nodes sharing the
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
    """scrapeMM's client uses ezMM's registry too."""
    item_registry.close()  # ezMM can't move a registry with open connections
    item_registry.clear_cache()
    set_ezmm_path(path)


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
    """The Responses API makes functions strict by default, which our schemas don't satisfy."""
    return [{"type": "function", "strict": False, **tool} for tool in TOOL_DEFINITIONS]


def chat_completions_tools() -> list[dict]:
    return [{"type": "function", "function": tool} for tool in TOOL_DEFINITIONS]


def anthropic_tools() -> list[dict]:
    return [
        {"name": tool["name"], "description": tool["description"], "input_schema": tool["parameters"]}
        for tool in TOOL_DEFINITIONS
    ]


def normalize_scrape_methods(methods: list[str] | str | None) -> ScrapeMethods:
    """scrapeMM's retrieve() takes the literal "auto" or a non-empty list of backend names."""
    if methods is None or methods == "auto" or methods == ["auto"]:
        return "auto"
    if isinstance(methods, str):
        return [methods]
    methods = [m for m in methods if m]
    return methods or "auto"


def parse_claim_day(claim_date: str | datetime | date | None) -> date | None:
    """Raises ValueError for a date that cannot be parsed, so that a malformed date never
    silently degrades into an unfiltered search."""
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
    """The cutoff day itself is passed as `before`: scrapeMM leaves out that day and
    everything after it (Google's `cd_max` becomes the day before the cutoff)."""
    return SerperQuery(
        q=query,
        num=NUM_SEARCH_RESULTS,
        page=page,
        hl="en",
        gl="us",
        before=parse_claim_day(cutoff_date),
    )


def format_search_results(query: str, results: list[SerperOrganicResult]) -> str:
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
    """Without the query: blocked URLs are blocked whatever tracking parameters they come with."""
    return normalize_url(url).split("?", 1)[0]


def _ascii_host(host: str) -> str:
    """Lower case and ASCII, as the publishers' fact-check scopes have it
    ("XN--80AA.example.org" or its Unicode form -> "xn--80aa.example.org")."""
    try:
        return host.lower().encode("idna").decode("ascii")
    except UnicodeError:
        return host.lower()


def _scope_of(scope: str) -> tuple[str, str]:
    """A fact-check scope or blocked URL as host and lower-case path ("leadstories.com/hoax-alert"
    -> ("leadstories.com", "hoax-alert"), "https://www.example.org/a/b/?x=1" -> ("example.org", "a/b"))."""
    host, _, path = _page_of(scope if "://" in scope else f"https://{scope}").partition("/")
    return _ascii_host(host), path.lower()


# A host, maybe with a path, with or without scheme ("leadstories.com/hoax-alert/2026/07/..."),
# but not in an e-mail address
_HOST_AND_PATH = re.compile(r"(?<![\w.@-])(?:https?://)?(?:www\.)?((?:[\w-]+\.)+[^\W\d_]{2,})(?:/([^\s\"'<>()\[\]]*))?")
# The spaces with which Google breaks URLs and slugs in titles and snippets, before or after a
# "-", "/" or "." ("https://maldita. es/desinfo/...", "2026-09-15- maryam-...")
_GOOGLE_BREAK = re.compile(r"(?<=[-/.]) | (?=[-/.])")

MIN_SLUG_LENGTH = 12
# A piece this long counts as the slug where Google cut it short in a title or snippet
MIN_SLUG_PIECE_LENGTH = 20


def _slug_of(url: str) -> str | None:
    """The last path segment with a letter in it ("https://www.dogrulukpayi.com/dogruluk-kontrolu/
    ucuz-mama-kedileri-gay-yapar-mi" -> "ucuz-mama-kedileri-gay-yapar-mi"). None if too short to
    tell the article apart, as "n-31898"."""
    _, *segments = _page_of(url).split("/")
    slug = next((segment for segment in reversed(segments) if re.search(r"[^\W\d_]", segment)), "")
    return slug if len(slug) >= MIN_SLUG_LENGTH else None


def normalize_url(url: str) -> str:
    """Without scheme, "www.", trailing slash and fragment, host in lower case and
    percent-encoding decoded ("https://www.Example.org/a%20b/#top" -> "example.org/a b"): the
    model may write a URL it saw in any of these forms. "" if the URL has no host."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:  # Malformed, e.g. "https://[x"
        return ""
    host = parts.netloc.lower().removeprefix("www.")
    if not host:
        return ""
    query = f"?{unquote(parts.query)}" if parts.query else ""
    return f"{host}{unquote(parts.path).rstrip('/')}{query}"


@dataclass
class ToolResult:
    """Text and, of a fetched page, the images and videos shown, in their places."""

    content: ItemSequence | str
    # For the trace: reference, kind and source URL of each medium shown
    media: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if isinstance(self.content, str):
            self.content = [self.content]

    @property
    def text(self) -> str:
        """The content as text, each image and video replaced by its reference ("<image:40>")."""
        return render(self.content)


class ToolSession:
    """The tool calls of a single fact-check: one per claim, providers are shared across
    worker threads."""

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
        link_resolver: GoogleLinkResolver | None = None,
    ):
        """
        Args:
            claim_date: Raises ValueError if it cannot be parsed, as does cutoff_date.
            scrape_methods: scrapeMM backends in order, a subset of {"integrations", "browser",
                            "firecrawl", "decodo"}, or "auto" to let the server choose per domain.
            blocked_urls: The claim's own fact-checks: blocked like blocked_scopes, with all
                          pages under them (".../amp"), and so is any URL with their slug in its
                          path on any site and any search result whose title or snippet has the
                          slug or a piece of it. Google's date filter lets them through
                          at times, and fact-checkers' pages link to them in their current versions.
            cutoff_date: The day the claim's fact-check was published; the claim date if not given.
            blocked_scopes: Hosts, or hosts and paths, under which fact-checkers publish
                            fact-checks ("leadstories.com/hoax-alert"), e.g. of all IFCN
                            signatories and EFCSN members. URLs in them are blocked whatever
                            their scheme, "www.", query and fragment; the rest of the site is not.
            claim_images: The images the claim's media take in every request (video frames for
                          models without native video), which count toward
                          media_limits.request_images.
            link_resolver: One with the Serper API key of config.yaml or the environment if not given.
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
        # Host -> blocked path ("" for the whole host) -> the rule that blocks it, for the log
        self.blocked_scopes: dict[str, dict[str, str]] = {}
        for entry, rule in [*((url, f"at or under the claim's fact-check {url}") for url in blocked_urls),
                            *((scope, f"in the fact-check scope {scope}") for scope in blocked_scopes)]:
            host, path = _scope_of(entry)
            if host:
                self.blocked_scopes.setdefault(host, {}).setdefault(path, rule)
        self.blocked_slugs = {_slug_of(url) for url in blocked_urls} - {None}
        self.link_resolver = link_resolver or GoogleLinkResolver()

        self.n_searches = 0
        self.n_fetches = 0
        self.encountered_urls: list[str] = []  # Of the web_search results
        self.retrieved_urls: list[str] = []
        self.linked_urls: list[str] = []  # In the fetched content shown to the model
        self.shown_media: list[dict] = []
        # {"tool", "url", "rule"} per dropped search result or refused fetch, and "query" for searches
        self.blocked: list[dict] = []
        self.media_files: set[Path] = set()  # Of all media of the fetched pages, shown or not

    def log(self, message: str):
        print(f"    {self.log_prefix}{message}")

    def close(self):
        _page_media_files.release(self.media_files)
        self.media_files = set()

    def call(self, name: str, args: dict | None) -> ToolResult:
        if not isinstance(args, dict):
            return ToolResult(f"Error: Invalid arguments for {name}. Expected a JSON object.")
        if name == "web_search":
            return ToolResult(self.web_search(str(args.get("query") or "")))
        if name == "fetch_url":
            return self.fetch_url(str(args.get("url") or ""))
        return ToolResult(f"Error: Unknown tool '{name}'. Available tools: web_search, fetch_url.")

    def web_search(self, query: str) -> str:
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
        """The first `want` unblocked results, from up to `max_pages` pages of Google results. A
        page with fewer than NUM_SEARCH_RESULTS results is not always the last, and later pages
        repeat some results, so paging stops at a page that brings no new result.

        Returns the results, the number of blocked results and the number of pages requested.
        """
        results, seen, n_blocked = [], set(), 0
        for page in range(1, max_pages + 1):
            # Only organic results: the knowledge graph, answer box, top stories etc. are not
            # date-filtered by Google and would leak information from after the claim date.
            organic = self._resolve_redirects(self._search_page(query, page).organic, query)
            new = [r for r in organic if r.link and r.link not in seen]
            if not new:
                return results, n_blocked, page
            seen.update(r.link for r in new)
            # A title or snippet showing a blocked URL is mostly a fact-checker's own post, which
            # gives the fact-check's headline with its URL
            allowed = []
            for r in new:
                if is_hidden_redirect(r.link):
                    rule = f"a URL {UNRESOLVED_REDIRECT}"
                elif reason := self.blocked_by(r.link):
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

    def _resolve_redirects(self, results: list[SerperOrganicResult], query: str) -> list[SerperOrganicResult]:
        """A redirect Serper does not resolve stays as it is."""
        links = [r.link for r in results]
        n_redirects = sum(map(is_google_redirect, links))
        if not n_redirects:
            return results
        resolved = _run_async(self.link_resolver.resolve(links, log=self.log))
        n_resolved = n_redirects - sum(map(is_hidden_redirect, resolved))
        self.log(f"web_search: {query!r}: resolved {n_resolved} of {n_redirects} Google redirects")
        return [replace(r, link=link) if link != r.link else r for r, link in zip(results, resolved)]

    def _find_blocked_slug(self, text: str) -> tuple[str, re.Match] | None:
        """`text` decoded and with Google's breaks taken out, and the match of a blocked slug in
        it, in a URL or not ("fact-checks/<slug>"). None if it has none."""
        text = _GOOGLE_BREAK.sub("", unquote(text))
        for slug in self.blocked_slugs:
            if match := re.search(r"(?<![\w-])" + re.escape(slug) + r"(?![\w-])", text):
                return text, match
        return None

    def _slug_in(self, text: str) -> str | None:
        """The blocked slug `text` has, or a word or path segment of it that is a piece of one,
        where Google cut the slug short at either end, as the log gives it. Whether a piece is of
        the blocked article or of another one with a similar slug is unknown. None if it has none."""
        if found := self._find_blocked_slug(text):
            return f"the slug {found[1].group(0)} of the claim's fact-check"
        text = _GOOGLE_BREAK.sub("", unquote(text))
        parts = (part.strip(".,;:!()[]\"'·") for part in re.split(r"[\s/?#…]+|\.{2,}", text))
        return next((f"{part!r}, a piece of the slug {slug} of the claim's fact-check"
                     for part in parts for slug in self.blocked_slugs
                     if len(part) >= MIN_SLUG_PIECE_LENGTH and part in slug), None)

    def _blocked_in_preview(self, preview: str) -> str | None:
        """What blocked a title or snippet shows, for the log: a blocked slug or a piece of it
        (see _slug_in), or a blocked URL, also where Google broke it with spaces
        ("https://leadstories. com/hoax-alert/...") or cut it short, then if it is the start of a
        blocked URL or scope ("https://leadstories.com/hoa ..."), whether or not that is the
        blocked article. None if it shows none."""
        if slug := self._slug_in(preview):
            return f"has {slug}"
        scopes = [(f"{host}/{path}".rstrip("/"), rule)
                  for host, paths in self.blocked_scopes.items() for path, rule in paths.items()]
        # Both as it is and without Google's breaks, which also join a sentence to the next
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
        """Why `url` is blocked, as the log gives it, or None: on a blocked host (subdomains have
        scopes of their own) at or under a blocked path, in any case; or with a blocked slug in
        its path on any site (sites serve an article under several paths, archives and other
        sites copy it). A Google redirect with its target in the clear is blocked as its target."""
        if target := plain_target(url):
            reason = self.blocked_by(target)
            return f"{reason}, behind a Google redirect" if reason else None
        host, _, path = _page_of(url).partition("/")
        if slug := next((slug for slug in self.blocked_slugs if slug in path.split("/")), None):
            return f"with the slug {slug} of the claim's fact-check in its path"
        path = path.lower()
        return next((rule for scope_path, rule in self.blocked_scopes.get(_ascii_host(host), {}).items()
                     if not scope_path or path == scope_path or path.startswith(f"{scope_path}/")), None)

    def is_blocked(self, url: str) -> bool:
        return self.blocked_by(url) is not None

    def _search_page(self, query: str, page: int):
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
        return {normalize_url(url) for url in self.encountered_urls + self.linked_urls} - {""}

    def fetch_url(self, url: str) -> ToolResult:
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            return ToolResult(f"Error: '{url}' is not a valid URL. Provide a full URL starting with http:// or https://.")
        reason = self.blocked_by(url)
        target = plain_target(url) or url
        # Search results come resolved, so the model can only have seen such a link in a fetched page
        if not reason and is_hidden_redirect(url) and normalize_url(url) in self.seen_urls():
            (target,) = _run_async(self.link_resolver.resolve([url], log=self.log))
            if is_hidden_redirect(target):
                reason = UNRESOLVED_REDIRECT
            elif reason := self.blocked_by(target):
                reason += ", behind a Google redirect"
        if reason:
            self.blocked.append({"tool": "fetch_url", "url": url, "rule": f"a URL {reason}"})
        if reason or normalize_url(url) not in self.seen_urls():
            # A blocked URL gets the answer of an unseen one: any other would tell the model it
            # guessed the claim's fact-check
            self.log(f"fetch_url: {url} -> " + (f"blocked, a URL {reason}" if reason else "rejected, URL not seen"))
            return ToolResult(f"Error: Cannot fetch {url}. fetch_url only fetches URLs that appeared in web_search "
                              f"results or in fetched pages. Use them as they appeared.")
        if self.n_fetches >= self.max_fetches:
            return ToolResult(f"Fetch limit of {self.max_fetches} fetch_url calls reached. "
                              f"Continue with the information gathered so far.")
        self.n_fetches += 1
        if target != url:
            self.log(f"fetch_url: {url} -> the Google redirect's target {target}")
            url = target

        try:
            response = _run_async(asyncio.wait_for(retrieve(
                url,
                show_progress=False,
                output_format="multimodal",
                methods=self.scrape_methods,
                strip=True,  # Main content only: no navigation, footers, cookie banners
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
        """Without the links to blocked URLs, with their text and title, which may give the
        fact-check's verdict away, and any images in that text (a teaser's, say).

        Returns the content and the links removed, each as its URL and the rule that blocks it.
        """
        parts = list(sequence)
        if not self.blocked_scopes:
            return parts, []
        # Placeholders for the items that no link target or URL takes in
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
        """The text cut at max_fetch_chars (an image or video counts as long as its reference), with
        the media in their places as far as the limits per page, per fact-check and per request
        allow; a note stands for the media beyond these, shown before, or unreadable.

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
    for url in urls:
        if url not in target:
            target.append(url)
