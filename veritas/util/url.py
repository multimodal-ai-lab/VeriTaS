import asyncio
import logging
import re
from urllib.parse import urlparse

import aiohttp
import requests
import tldextract
from bs4 import BeautifulSoup, Tag
from playwright.async_api import async_playwright
from pydantic import HttpUrl
from scrapemm import retrieve
from scrapemm.common import ScrapingResponse
from scrapemm.util import to_multimodal_sequence

from veritas.util.parsing import perform_extraction
from veritas.util.scraping import HEADERS

logger = logging.getLogger("VeriTaS")


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


def unshorten(url: str) -> str:
    """If the URL is a TinyURL, returns the original (long) URL. Otherwise,
    returns the input URL."""
    domain = get_domain(url)
    if domain in ["tinyurl.com", "bit.ly", "goo.gl", "youtu.be", "t.ly"]:
        resp = requests.head(url, allow_redirects=True)
        return resp.url
    else:
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

        # Handle regular perma.cc URLs by scraping
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10, headers=HEADERS) as response:
                if response.status != 200:
                    logger.info(f"Failed to fetch Perma.cc URL: {url} (status: {response.status})")
                    return None

                html = await response.text()

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


async def get_original_url_playwright(archive_url: str) -> str | None:
    """Attempts to retrieve the original URL from an Archive Today page using Playwright.

    Args:
        archive_url: The archive.today (or similar domain) URL to resolve

    Returns:
        The original URL if found, None otherwise
    """
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            page = await context.new_page()

            # Load the page
            await page.goto(archive_url, wait_until='domcontentloaded', timeout=20000)
            await asyncio.sleep(2)  # Wait for everything to load

            # Get original URL from the input field
            try:
                original_url = await page.input_value('input[type="text"]')
                await browser.close()

                if original_url and original_url.startswith("http"):
                    return original_url
                return None
            except Exception as e:
                logger.debug(f"Could not extract URL from input field: {e}")
                await browser.close()
                return None

    except Exception as e:
        logger.debug(f"Playwright extraction failed for {archive_url}: {e}")
        return None


async def resolve_archive_today_url(url: str) -> dict | None:
    """Resolves an Archive Today URL to the original URL.

    Archive Today uses multiple domains (archive.ph, archive.is, etc.) and two URL formats:
    1. Long format with original URL in path: https://archive.today/2022.04.08-155753/https://example.com
    2. Short code format: https://archive.ph/nLdE3

    For long format URLs, the original URL is extracted via regex (no network request needed).
    For short code format URLs, tries Playwright first, then falls back to scrapeMM.
    Note: Short format requires Cloudflare bypass (Firecrawl/Decodo with Advanced plan).
    """
    from veritas.pipeline import max_video_size

    # First, try regex for long-format URLs (no need for scraping)
    match = re.match(
        r"^https?://(?:archive\.today|archive\.is|archive\.ph|archive\.vn|archive\.li|archive\.fo|archive\.md).*?[/-](?P<original>https?://?[^#]*)",
        url,
    )
    if match:
        result = match.group("original")
        return dict(original_url=result)

    # Otherwise, the URL is in short format. This needs targeted scraping and extraction.
    # Use Decodo (Advanced Scraping API required) to retrieve the HTML of the page and parse it
    result: ScrapingResponse = await retrieve(url, methods=["decodo"], format="html", show_progress=False,
                                              max_video_size=max_video_size)

    # TODO: Move this to ScrapeMM:
    # Extract the original URL from the HTML and return it along with the scraped content (to save requests)
    if result:
        html = result.content
        if html:
            extracted = perform_extraction(html, extraction={"tag": "input", "attrs": {"name": "q"}, "get": "value"})
            original_url = extracted if isinstance(extracted, str) and extracted.startswith("http") else None

            # Turn scraped HTML into multimodal sequence
            async with aiohttp.ClientSession() as session:
                mm_seq = await to_multimodal_sequence(html, remove_urls=False, session=session, url=url)

            return dict(original_url=original_url, scraped_content=mm_seq)


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


def is_archiving_url(url: str):
    return get_domain(url) in ARCHIVING_SITES


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
