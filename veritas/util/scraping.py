import logging
from typing import Iterable, Optional

import aiohttp
from aiohttp.http_exceptions import ContentLengthError
from bs4 import BeautifulSoup
from pydantic import HttpUrl
from scrapemm import retrieve

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


async def get_rendered_htmls(urls: list[str]) -> list[str | None]:
    """Retrieves the HTML of pages that need JavaScript to show their content.
    Anything beyond a static request goes through the scrapeMM server, which
    renders the pages in a real browser. Returns None for pages that failed.
    Raises scrapeMM's `ServerError` if the server is unreachable."""
    if not urls:
        return []
    responses = await retrieve(list(urls), output_format="html", show_progress=False)
    return [response.content.html if response.success else None for response in responses]


async def get_all_links(url: str) -> Optional[list[str]]:
    """Returns a list of all links on the (static) page specified by the given URL."""
    async with aiohttp.ClientSession() as session:
        html = await request_static(url, session)
    if html:
        soup = BeautifulSoup(html, features="lxml")
        links = soup.find_all('a', href=True)
        return [a['href'] for a in links]
