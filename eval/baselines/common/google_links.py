"""Google's redirect links in search results, resolved through Serper's /resolve.

Serper gives some organic results as Google redirects with an encrypted target
("https://www.google.com/url?...&q=CAES...", "https://www.google.com/goto?url=CAES..."), which
hides the target from every block rule while scrapeMM follows the redirect when fetching it: the
claim's fact-check came through to the model this way. Serper's /resolve (free, not publicly
documented) answers with the real URLs once Google has released them: it may wait until then,
echoes a link it cannot resolve, and adds `retryAfter` while Google still holds one back.

The scrapeMM server (1.10) has no route for /resolve, so it is called from this machine, with the
Serper API key in config.yaml (`api_secrets.serper`) or the SERPER_API_KEY environment variable.
"""

import asyncio
import os
import time
from typing import Any, Callable, Sequence
from urllib.parse import parse_qs, urlsplit

import aiohttp

try:
    import veritas
except ImportError:
    veritas = None

REDIRECT_PATHS = ("/url", "/goto")


def is_google_redirect(url: str | None) -> bool:
    if not url:
        return False
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    host = (parts.hostname or "").removeprefix("www.")
    return host.startswith("google.") and parts.path in REDIRECT_PATHS


def plain_target(url: str | None) -> str | None:
    """The target a Google redirect has in the clear ("https://www.google.com/url?q=https://example.org/a&sa=U"
    -> "https://example.org/a"). None for other links and encrypted targets ("...&q=CAES...")."""
    if not is_google_redirect(url):
        return None
    params = parse_qs(urlsplit(url).query)
    return next((value for key in ("url", "q") for value in params.get(key, [])
                 if value.startswith(("http://", "https://"))), None)


def is_hidden_redirect(url: str | None) -> bool:
    return is_google_redirect(url) and plain_target(url) is None


def serper_api_key() -> str | None:
    secrets = getattr(veritas, "api_secrets", None) or {}
    return secrets.get("serper") or os.environ.get("SERPER_API_KEY") or None


class GoogleLinkResolver:
    """Every link comes back, in order: a redirect as its target where known, anything else as it
    is, and so are redirects still hidden when the budget runs out or Serper fails."""

    ENDPOINT = "https://google.serper.dev/resolve"
    # Serper refuses larger requests with 413 Request Entity Too Large
    MAX_LINKS_PER_REQUEST = 100

    def __init__(self, api_key: str | None = None, budget_seconds: float = 30.0, retry_seconds: float = 5.0):
        """`budget_seconds` is the wall time one `resolve` may take in total, `retry_seconds` the
        pause between two requests while Google holds links back."""
        self.api_key = api_key or serper_api_key()
        self.budget_seconds = budget_seconds
        self.retry_seconds = retry_seconds

    async def resolve(self, links: Sequence[str | None], log: Callable[[str], None] = print) -> list[str | None]:
        """Only the redirects with an encrypted target are sent to Serper. `log` gets the warnings."""
        resolved = [plain_target(link) or link for link in links]
        pending = [i for i, link in enumerate(resolved) if is_hidden_redirect(link)]
        if not pending:
            return resolved
        if not self.api_key:
            log(f"Warning: no Serper API key (api_secrets.serper in config.yaml, or SERPER_API_KEY): "
                f"{len(pending)} Google redirect(s) stay unresolved")
            return resolved

        deadline = time.monotonic() + self.budget_seconds
        while pending and (remaining := deadline - time.monotonic()) > 0:
            batches = [pending[i:i + self.MAX_LINKS_PER_REQUEST]
                       for i in range(0, len(pending), self.MAX_LINKS_PER_REQUEST)]
            rounds = await asyncio.gather(*(self._round(resolved, batch, remaining, log) for batch in batches))
            pending, retry = [], False
            for found, left, batch_retry in rounds:
                for i, link in found.items():
                    resolved[i] = link
                pending += left
                retry |= batch_retry
            if not pending or not retry:
                break
            await asyncio.sleep(max(0.0, min(self.retry_seconds, deadline - time.monotonic())))
        return resolved

    async def _round(self, links: Sequence[str | None], batch: list[int], timeout: float,
                     log: Callable[[str], None]) -> tuple[dict[int, str], list[int], bool]:
        """Returns the resolved links by index, the indices still pending and whether Serper asks
        to try those again (`retryAfter`). A failed request leaves the whole batch pending and asks
        for no retry."""
        try:
            answer = await self._request([links[i] for i in batch], timeout)
        except TimeoutError:
            return {}, batch, False  # The budget ran out during the request
        except (aiohttp.ClientError, ValueError) as e:
            log(f"Warning: Serper /resolve failed: {type(e).__name__}: {e}")
            return {}, batch, False
        entries = answer.get("organic")
        if not isinstance(entries, list) or len(entries) != len(batch):
            log(f"Warning: Serper /resolve answered {len(batch)} link(s) with {answer!r:.300}")
            return {}, batch, False
        found, left = {}, []
        for i, entry in zip(batch, entries):
            link = entry.get("link") if isinstance(entry, dict) else None
            if link and link.startswith(("http://", "https://")) and not is_google_redirect(link):
                found[i] = link
            else:
                left.append(i)
        # Without `retryAfter` Serper has given up on what is left
        return found, left, "retryAfter" in answer

    async def _request(self, links: list[str], timeout: float) -> dict[str, Any]:
        """Raises TimeoutError after `timeout` seconds, aiohttp.ClientError for a failed or refused
        request and ValueError for an answer that is not JSON."""
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.post(self.ENDPOINT, json={"organic": [{"link": link} for link in links]},
                                    headers={"X-API-KEY": self.api_key}) as response:
                response.raise_for_status()
                return await response.json()
