"""Self-hosted model fact-checker provider (via vLLM).

Self-hosted models are served through vLLM's OpenAI-compatible API, so this provider
runs the same tool-calling loop as the OpenAI provider against a different endpoint.
"""

import os

from openai import OpenAI

# Import config from Veritas
try:
    from veritas import selfhosted as _selfhosted
    VERITAS_SELFHOSTED_URL = _selfhosted.get("url") if _selfhosted else None
    VERITAS_SELFHOSTED_KEY = _selfhosted.get("key") if _selfhosted else None
except ImportError:
    VERITAS_SELFHOSTED_URL = None
    VERITAS_SELFHOSTED_KEY = None

from ..common.tools import DEFAULT_MAX_SEARCHES, DEFAULT_MAX_FETCHES
from .openai import OpenAIFactChecker


class SelfhostedFactChecker(OpenAIFactChecker):
    """Fact-checker using a self-hosted model via vLLM with the web_search and fetch_url tools."""

    provider_name = "selfhosted"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        scrape_methods: list[str] | str | None = "auto",
        max_searches: int = DEFAULT_MAX_SEARCHES,
        max_fetches: int = DEFAULT_MAX_FETCHES,
    ):
        """
        Initialize the self-hosted model fact-checker.

        Args:
            api_key: API key for vLLM endpoint. If None, uses config/env var.
            base_url: Base URL for vLLM endpoint. If None, uses config/env var.
            model: Model identifier (as served by vLLM).
            scrape_methods: Which scrapeMM backends fetch_url uses (subset of
                            integrations/browser/firecrawl/decodo, or "auto").
            max_searches: Maximum number of web_search calls per claim.
            max_fetches: Maximum number of fetch_url calls per claim.
        """
        self.base_url = base_url or VERITAS_SELFHOSTED_URL or os.environ.get("VLLM_BASE_URL")
        super().__init__(
            api_key=api_key,
            model=model,
            scrape_methods=scrape_methods,
            max_searches=max_searches,
            max_fetches=max_fetches,
        )

    def _create_client(self, api_key: str | None) -> OpenAI:
        """Create OpenAI-compatible client for vLLM."""
        if not self.base_url:
            raise ValueError(
                "vLLM base URL required. Either:\n"
                "  1. Add it to config.yaml (selfhosted.url)\n"
                "  2. Set VLLM_BASE_URL environment variable\n"
                "  3. Pass base_url parameter"
            )

        # API key (vLLM often uses a dummy key)
        key_to_use = api_key or VERITAS_SELFHOSTED_KEY or os.environ.get("VLLM_API_KEY", "dummy")

        return OpenAI(api_key=key_to_use, base_url=self.base_url)
