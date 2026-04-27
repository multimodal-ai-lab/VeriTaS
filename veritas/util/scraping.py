import logging
from typing import Iterable, Optional

import aiohttp
from aiohttp.http_exceptions import ContentLengthError
from bs4 import BeautifulSoup
from playwright.async_api import TimeoutError, async_playwright, BrowserContext, Error
from pydantic import HttpUrl

from veritas.util.util import run_with_semaphore

logger = logging.getLogger("VeriTaS")


async def get_static_htmls(
        urls: Iterable[str | HttpUrl], session: aiohttp.ClientSession, show_progress=False
) -> list[str | None]:
    tasks = (request_static(url, session) for url in urls)
    htmls = await run_with_semaphore(
        tasks, limit=100, show_progress=show_progress
    )
    return htmls


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/123.0.0.0 Safari/537.36",
}


async def request_static(
        url: str | HttpUrl, session: aiohttp.ClientSession, get_text: bool = True, **kwargs
) -> str | bytes | None:
    """Downloads the static page from the given URL using aiohttp. If `get_text` is True,
    returns the HTML as text. Otherwise, returns the raw binary content (e.g. an image)."""
    # TODO: Handle web archive URLs
    if session is None:
        logger.warning("No aiohttp session provided.")
        return None

    url = str(url)
    try:
        async with session.get(
                url, timeout=10, headers=HEADERS, allow_redirects=True, raise_for_status=True, **kwargs
        ) as response:
            if get_text:
                return await response.text()  # HTML string
            else:
                return await stream(response)  # Binary data
    except TimeoutError:
        pass  # Server too slow
    except UnicodeError:
        pass  # Page not readable
    except (aiohttp.ClientOSError, aiohttp.ClientConnectorError):
        pass  # Page not available anymore
    except ContentLengthError:
        pass  # 'Not enough data for satisfy content length header.'
    except aiohttp.ClientResponseError as e:
        if e.status in [403, 404, 429, 500, 502, 503]:
            # 403: Forbidden access
            # 404: Not found
            # 429: Too many requests
            # 500: Server error
            # 502: Bad gateway
            # 503: Service unavailable (e.g. rate limit)
            pass
        else:
            logger.debug(f"Failed to retrieve page.\n\t{type(e).__name__}: {e}")
    except Exception as e:
        logger.info(f"Failed to retrieve page at {url}.\n\tReason: {type(e).__name__}: {e}")


async def stream(response: aiohttp.ClientResponse, chunk_size: int = 1024) -> bytes:
    data = bytearray()
    async for chunk in response.content.iter_chunked(chunk_size):
        data.extend(chunk)
    return bytes(data)  # Convert to immutable bytes if needed


async def get_dynamic_htmls(urls: list[str], **kwargs):
    """Reads multiple URLs dynamically and concurrently."""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(accept_downloads=False)
            tasks = [get_dynamic_html(url, context, **kwargs) for url in urls]
            results = await run_with_semaphore(tasks, limit=100)
            await browser.close()
            return results

    except Exception as e:
        logger.warning(f"Unable to dynamically read pages: {e}")


async def get_dynamic_html(url: str, context: BrowserContext, buttons_to_click: list[str] = None) -> str | None:
    """Retrieves the HTML of a URL using Playwright. Loads JavaScript contents
    dynamically and can perform button clicks. Specify the buttons using CSS selectors."""
    page = await context.new_page()

    try:
        await page.goto(url, timeout=60000)
        await page.wait_for_load_state(
            "networkidle"
        )  # 'domcontentloaded'
    except (TimeoutError, Error) as e:
        logger.warning(f"\rUnable to load page at URL '{url}'.\n\tReason: {type(e).__name__} {e}")
        return

    if buttons_to_click:
        for button in buttons_to_click:
            # Click buttons
            try:
                # 1. Wait for the button to appear
                await page.wait_for_selector(f"{button}", timeout=5000)

                # 2. Click the button
                await page.click(f"{button}")

                # 3. Optionally wait for new content to load
                await page.wait_for_load_state("networkidle")

            except Exception as e:
                logger.warning(f"Button {button} not found or clickable: {e}")

    # Get the full page HTML after JS execution
    content = await page.content()
    await page.close()
    return content


async def get_all_links(url: str, **kwargs) -> Optional[list[str]]:
    """Returns a list of all links on the page specified by the given URL."""
    if html := (await get_dynamic_htmls([url], **kwargs))[0]:
        soup = BeautifulSoup(html, features="lxml")
        links = soup.find_all('a', href=True)
        return [a['href'] for a in links]
