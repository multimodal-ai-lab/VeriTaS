"""Self-hosted model fact-checker provider (via vLLM).

The model verifies the claim with the web_search and fetch_url tools via function calling
in vLLM's OpenAI-compatible chat completions API.
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

from .chat_completions import ChatCompletionsFactChecker


class SelfhostedFactChecker(ChatCompletionsFactChecker):
    """Fact-checker using a self-hosted model via vLLM with the web_search and fetch_url tools."""

    provider_name = "selfhosted"
    default_model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"

    def __init__(self, api_key: str | None = None, base_url: str | None = None, **kwargs):
        """
        Initialize the self-hosted model fact-checker.

        Args:
            api_key: API key for vLLM endpoint. If None, uses config/env var.
            base_url: Base URL for vLLM endpoint. If None, uses config/env var.
            **kwargs: See BaseFactChecker.
        """
        self.base_url = base_url or VERITAS_SELFHOSTED_URL or os.environ.get("VLLM_BASE_URL")
        super().__init__(api_key=api_key, **kwargs)

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
