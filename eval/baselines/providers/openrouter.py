"""OpenRouter fact-checker provider.

The model verifies the claim with the web_search and fetch_url tools via function calling
in OpenRouter's OpenAI-compatible chat completions API.
"""

from openai import OpenAI

from .base import resolve_api_key
from .chat_completions import ChatCompletionsFactChecker


class OpenRouterFactChecker(ChatCompletionsFactChecker):
    """Fact-checker using a model on OpenRouter with the web_search and fetch_url tools."""

    provider_name = "openrouter"
    default_model = "z-ai/glm-5.3-flash"

    def __init__(self, host: str | None = None, **kwargs):
        """
        Initialize the OpenRouter fact-checker.

        Args:
            host: The one endpoint every request goes to, as its tag in the model's endpoint list
                  (https://openrouter.ai/api/v1/models/<model>/endpoints), e.g. "deepinfra/fp8".
                  Required: otherwise OpenRouter spreads the requests over the model's hosts,
                  which differ in quantization and context length.
            **kwargs: See BaseFactChecker.
        """
        if not host:
            raise ValueError("OpenRouter needs a host to pin, e.g. \"deepinfra/fp8\" (see the tags in "
                             "https://openrouter.ai/api/v1/models/<model>/endpoints)")
        self.host = host
        super().__init__(**kwargs)

    def _request_params(self) -> dict:
        return {"extra_body": {
            "provider": {
                "only": [self.host],
                # Without this, OpenRouter silently drops the tools or the reasoning setting if the host lacks them
                "require_parameters": True,
            },
            # OpenRouter's reasoning setting, which it translates for the model's vendor
            "reasoning": {"effort": self.reasoning_effort},
        }}

    def _create_client(self, api_key: str | None) -> OpenAI:
        return OpenAI(
            api_key=resolve_api_key(api_key, "openrouter", "OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
