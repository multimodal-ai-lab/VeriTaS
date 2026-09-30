import logging
import re
from urllib.parse import unquote, urlparse

import aiohttp
import requests
import tldextract
from bs4 import BeautifulSoup, Tag
from pydantic import HttpUrl
from scrapemm import retrieve
from scrapemm.common import ScrapingResponse

from veritas.util.scraping import HEADERS

logger = logging.getLogger("VeriTaS")

#: Matches a bare URL inside free text (e.g. a publisher's name field that embeds
#: its homepage). Used to be re-exported by scrapeMM (`scrapemm.util.URL_REGEX`);
#: inlined here since scrapeMM's client package no longer ships that module - the
#: scraping engine it belonged to moved server-side.
URL_REGEX = r"https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9@:%_\+.~#?&//=]*)"


def preprocess_url(url: str) -> str:
    """Decodes a URL and strips unwanted symbols from it, such as surrounding
    whitespace or non-breaking spaces. Used to be re-exported by scrapeMM
    (`scrapemm.util.preprocess_url`); inlined here for the same reason as
    `URL_REGEX` above."""
    return unquote(str(url)).strip()


def get_domain(url: str | HttpUrl | None) -> str | None:
    """Uses tldextract to get out the domain (incl. suffix) from the given URL. The output will be
    of the form 'example.com' or 'domain.co.uk'. No subdomains, no 'www', no 'http'."""
    if url is None:
        return None
    url_str = str(url)
    try:
        return tldextract.extract(url_str).top_domain_under_public_suffix
    except Exception:
        return None


def is_domain_root(url: str | HttpUrl) -> bool:
    """Uses urlparse to determine if the URL points at the domain root."""
    url_str = str(url)
    try:
        if not url.startswith("http"):
            url_str = "https://" + url_str
        parsed = urlparse(url_str)
        return parsed.path in ["/", "", None] and not parsed.query and not parsed.fragment
    except Exception:
        return False


#: Domains whose URLs are mere redirects to the actual source.
SHORTENER_DOMAINS = ["tinyurl.com", "bit.ly", "goo.gl", "youtu.be", "t.ly"]

#: A shortener that does not answer promptly is not worth waiting for; the caller
#: can still use the short URL, which resolves again when it is actually fetched.
UNSHORTEN_TIMEOUT = 15


async def unshorten(url: str, session: aiohttp.ClientSession | None = None) -> str:
    """If the URL is a TinyURL, returns the original (long) URL. Otherwise,
    returns the input URL.

    Pass a `session` to reuse an open connection pool; without one, a short-lived
    session is opened for the single request. A lookup that fails or times out
    yields the input URL rather than raising: the short URL still points at the
    same target, it just names it less explicitly."""
    if get_domain(url) not in SHORTENER_DOMAINS:
        return url

    if session is None:
        async with aiohttp.ClientSession(headers=HEADERS) as own_session:
            return await unshorten(url, own_session)

    try:
        async with session.head(url, allow_redirects=True,
                                timeout=aiohttp.ClientTimeout(total=UNSHORTEN_TIMEOUT)) as response:
            return str(response.url)
    except Exception as e:
        logger.debug(f"Unable to unshorten {url}: {type(e).__name__}: {e}")
        return url


async def resolve_archiving_url(url: str) -> dict | None:
    """Takes the URL from a known archiving service and tries to get the original URL,
    i.e., the URL of the archived source. Returns None if not successful."""
    resolve_method = ARCHIVING_SITES.get(get_domain(url))
    if resolve_method:
        return await resolve_method(url)
    else:
        return None


async def resolve_perma_cc_url(url: str) -> dict | None:
    """Resolves a Perma.cc URL to the original URL.
    TODO: Speed this up. Instead of scraping the page, find a different way.

    Handles two formats:
    1. Regular perma.cc URLs: https://perma.cc/XXXX-XXXX - scrapes the page
    2. Rejouer perma.cc URLs: https://rejouer.perma.cc/replay-web-page/.../mp_/ORIGINAL_URL - extracts from URL

    Args:
        url: The perma.cc archive URL

    Returns:
        The original URL if found, None otherwise
    """
    if "perma.cc" not in url:
        return None

    try:
        # Handle rejouer.perma.cc URLs with embedded original URL
        # Format: https://rejouer.perma.cc/.../mp_/https://example.com
        if "rejouer.perma.cc" in url and "_/" in url:
            resolved_url = url.split("_/")[-1]
            if resolved_url.startswith("http"):
                return dict(original_url=resolved_url)

        # Handle regular perma.cc URLs by scraping via scrapeMM
        response: ScrapingResponse = await retrieve(url, show_progress=False, output_format="html")
        if not response.success:
            return None

        html: str = response.content.html

        # Parse the HTML to extract the original URL
        soup = BeautifulSoup(html, "lxml")

        # The original URL is stored in an input field with id="source_url"
        source_input = soup.find("input", id="source_url")
        if source_input and "value" in source_input.attrs:
            original_url = source_input["value"]
            if isinstance(original_url, str) and original_url.startswith("http"):
                return dict(original_url=original_url)

        logger.info(f"Unable to find original URL in Perma.cc page: {url}")
        return None

    except Exception as e:
        raise
        logger.info(f"Unable to resolve Perma.cc URL: {url}\n{e}")
        return None


async def resolve_archive_org_url(url) -> dict | None:
    match = re.match(
        r"^https?:\/\/web\.archive\.org.*\/(?P<original>https?:\/(?P<second_slash>\/)?.*)",
        url,
    )
    # TODO https://web.archive.org/web/20220224084547/%20https:/twitter.com/suriel/status/1496750577425997831
    # TODO: Download media saved as "detail": https://archive.org/details/picture-1_202505
    if not match:
        # e.g. https://archive.org/details/FOXNEWSW_20220315_230000_Jesse_Watters_Primetime/start/672/end/732?q=happening
        logger.info(f"Unable to resolve archive.org URL: {url}")
        return None
    original_url = match.group("original")
    second_slash = match.group("second_slash")
    if not second_slash:
        # double the //
        original_url = original_url[:6] + "/" + original_url[6:]
    return dict(original_url=original_url)


async def resolve_archive_today_url(url: str) -> dict | None:
    """Resolves an Archive Today URL to the original URL - long form only
    (https://archive.today/2022.04.08-155753/https://example.com), extracted
    from the URL itself via regex, no network request needed.

    Short-code URLs (https://archive.ph/nLdE3) cannot be resolved anymore: this
    used to delegate to scrapeMM's `identify_snapshot()`, an ungated lookup
    through Archive.today's own metadata endpoints, but scrapeMM moved to a
    client/server split and that lookup lives only in the server's internal
    `archive_today` integration now - it is not exposed over the client's HTTP
    API (`/v1/archive-today` only covers the CAPTCHA session/buffer, not
    resolving a snapshot's original URL). A short-code URL's Appearance is
    therefore saved with `url=None`, same as any other appearance whose
    original URL could not be determined."""
    domains = "|".join(re.escape(domain) for domain in ARCHIVE_TODAY_DOMAINS)
    match = re.match(rf"^https?://(?:{domains}).*?[/-](?P<original>https?://?[^#]*)", url)
    if not match:
        return None
    return dict(original_url=match.group("original"))


async def resolve_ghostarchive_url(url: str) -> dict | None:
    """Resolves a Ghostarchive URL to the original URL."""
    if "ghostarchive.org" not in url:
        return None

    # The original URL is not part of the URL itself but in an input field with id="searchInput"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")
    except requests.RequestException as e:
        logger.info(f"Unable to resolve Ghostarchive URL: {url}\n{e}")
        return None

    if not isinstance(input_field := soup.find("input", id="searchInput"), Tag) or "value" not in input_field.attrs:
        logger.info(f"Unable to find original URL in Ghostarchive page: {url}")
        return None

    original_url = input_field["value"]
    if not isinstance(original_url, str) or not original_url.startswith("http"):
        logger.info(f"Original URL in Ghostarchive page is not valid: {original_url}")
        return None

    return dict(original_url=original_url)


def resolve_archiveport_url(url: str) -> dict:
    """Resolves an ArchivePort URL to the original URL. Example:
    https://archiveport.org/?url=https://archive.ph/20241001093811/https://romios.gr/to-vinteo-me-tin-methysmeni-i-ftiagmeni-kamala-kanei-ton-gyro-toy-diadiktyoy/"""
    # TODO


def is_archiving_url(url: str):
    return get_domain(url) in ARCHIVING_SITES


#: Archive.today's mirror domains. They are one and the same service - a single
#: access check, shared session and page cache cover all of them (see scrapeMM's
#: `archive_today` integration) - so anything gated on one of them is gated the
#: same way regardless of which mirror the URL happens to use.
ARCHIVE_TODAY_DOMAINS = (
    "archive.today", "archive.is", "archive.ph", "archive.vn",
    "archive.li", "archive.fo", "archive.md",
)


def is_archive_today_url(url: str | HttpUrl | None) -> bool:
    """True if the URL belongs to one of Archive.today's mirror domains."""
    return get_domain(url) in ARCHIVE_TODAY_DOMAINS


#: Matches an Archive.today URL on any mirror domain, for the SQL `~*` filters
#: that look up appearances/evidence sources still waiting behind the access
#: check (see `scripts/retry_deferred_archive_today.py`).
ARCHIVE_TODAY_URL_SQL_PATTERN = (
    r"^https?://([a-z0-9-]+\.)?(" +
    "|".join(domain.replace(".", r"\.") for domain in ARCHIVE_TODAY_DOMAINS) +
    r")(/|$)"
)


ARCHIVING_SITES = {
    "archive.today": resolve_archive_today_url,
    "archive.is": resolve_archive_today_url,
    "archive.ph": resolve_archive_today_url,
    "archive.vn": resolve_archive_today_url,
    "archive.li": resolve_archive_today_url,
    "archive.fo": resolve_archive_today_url,
    "archive.md": resolve_archive_today_url,
    "perma.cc": resolve_perma_cc_url,
    "ghostarchive.org": resolve_ghostarchive_url,
    "archive.org": resolve_archive_org_url,
    "awesomescreenshot.com": None,  # TODO
    "mvau.lt": None,  # TODO
    "archive.st": None,  # TODO
    "archiveport.org": None,  # TODO  # Meta archiving service by https://factreview.gr/, linking to captures from other archivers
    "sharethefacts.co": None,  # Not available anymore, former project by Duke Reporters' Lab
    # See also https://reporterslab.org/2016/05/12/new-share-facts-widget-helps-facts-rather-falsehoods-go-viral/
}

# Human-friendly service names for archiving domains. Useful for aggregation in stats/plots.
ARCHIVING_SERVICE_NAMES: dict[str, str] = {
    # Archive.today family
    "archive.today": "Archive.today",
    "archive.is": "Archive.today",
    "archive.ph": "Archive.today",
    "archive.vn": "Archive.today",
    "archive.li": "Archive.today",
    "archive.fo": "Archive.today",
    "archive.md": "Archive.today",
    # Others
    "archive.org": "Internet Archive",
    "perma.cc": "Perma.cc",
    "ghostarchive.org": "Ghostarchive",
    "awesomescreenshot.com": "Awesome Screenshot",
    "mvau.lt": "MediaVault",
    "archive.st": "Archive.st",
    "archiveport.org": "ArchivePort",
}


def normalize_archiver_label(domain: str | None) -> str:
    """Map known archiving service domains to a unified service label.

    If the domain is None or empty, returns "(unknown)". If the domain is not
    in our known list, returns the domain unchanged.
    """
    if not domain:
        return "(unknown)"
    d = domain.lower().strip()
    return ARCHIVING_SERVICE_NAMES.get(d, d)
