"""Unified fact-checker that supports multiple providers."""

from datetime import datetime
from typing import Literal

from .common.media import MediaLimits
from .common.tools import DEFAULT_MAX_SEARCHES, DEFAULT_MAX_FETCHES
from .common.types import FactCheckResult
from .providers import (
    BaseFactChecker,
    DEFAULT_REASONING_EFFORT,
    OpenAIFactChecker,
    GeminiFactChecker,
    AnthropicFactChecker,
    SelfhostedFactChecker,
    OpenRouterFactChecker,
)


Provider = Literal["openai", "gemini", "anthropic", "selfhosted", "openrouter"]

# Provider class mapping
PROVIDER_CLASSES: dict[Provider, type[BaseFactChecker]] = {
    "openai": OpenAIFactChecker,
    "gemini": GeminiFactChecker,
    "anthropic": AnthropicFactChecker,
    "selfhosted": SelfhostedFactChecker,
    "openrouter": OpenRouterFactChecker,
}

# Default models for each provider
DEFAULT_MODELS = {provider: cls.default_model for provider, cls in PROVIDER_CLASSES.items()}

DEFAULT_PROVIDERS: list[Provider] = ["openai", "gemini", "anthropic"]


class UnifiedFactChecker:
    """
    Unified fact-checker that can use any supported provider.

    Every provider runs the same baseline: the model verifies the claim with the
    web_search tool (results restricted to before the claim date) and the fetch_url
    tool, and answers with the claim's integrity, its certainty and a justification in JSON,
    which make a label of the 7-class scheme.

    Example usage:
        # Single provider
        fc = UnifiedFactChecker(provider="gemini")
        result = fc.check_claim("The Earth is flat", claim_date="2024-01-15")

        # Multiple providers
        fc = UnifiedFactChecker(providers=["openai", "gemini", "anthropic"])
        result = fc.check_claim("The Earth is flat", claim_date="2024-01-15", provider="openai")
    """

    def __init__(
        self,
        provider: Provider | None = None,
        providers: list[Provider] | None = None,
        model: str | None = None,
        models: dict[Provider, str] | None = None,
        api_keys: dict[str, str] | None = None,
        scrape_methods: list[str] | str | None = "auto",
        max_searches: int = DEFAULT_MAX_SEARCHES,
        max_fetches: int = DEFAULT_MAX_FETCHES,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        media_limits: MediaLimits = MediaLimits(),
        openrouter_host: str | None = None,
    ):
        """
        Initialize the unified fact-checker.

        Args:
            provider: Single provider to use (openai, gemini, anthropic, selfhosted, or openrouter).
            providers: List of providers to initialize (for multi-provider mode).
            model: Model to use for single provider mode.
            models: Dict mapping provider names to model identifiers.
            api_keys: Dict mapping provider/service names to API keys.
                      Keys: "openai", "google", "anthropic", "selfhosted", "openrouter"
            scrape_methods: Which scrapeMM backends fetch_url uses, in order (subset of
                        integrations/browser/firecrawl/decodo, or "auto"). Default "auto".
            max_searches: Maximum number of web_search calls per claim.
            max_fetches: Maximum number of fetch_url calls per claim.
            reasoning_effort: How much the models reason, e.g. "low", "medium" or "high",
                              passed as is to each API's effort setting. Default "medium".
            media_limits: Which images and videos the models get, and in what form.
            openrouter_host: The endpoint OpenRouter sends every request to, e.g. "deepinfra/fp8".
                             Required for openrouter.

        Note: Specify either `provider` or `providers`, not both.
        """
        self.api_keys = api_keys or {}
        self.models = dict(models or {})
        self.scrape_methods = scrape_methods
        self.max_searches = max_searches
        self.max_fetches = max_fetches
        self.reasoning_effort = reasoning_effort
        self.media_limits = media_limits
        self.openrouter_host = openrouter_host
        self._checkers: dict[Provider, BaseFactChecker] = {}

        # Determine which providers to initialize
        if provider and providers:
            raise ValueError("Specify either 'provider' or 'providers', not both")

        if provider:
            providers_to_init = [provider]
            if model:
                self.models[provider] = model
        elif providers:
            providers_to_init = providers
        else:
            providers_to_init = list(DEFAULT_PROVIDERS)

        # Initialize requested providers
        for p in providers_to_init:
            self._init_provider(p)

        self._default_provider = providers_to_init[0]

    def _init_provider(self, provider: Provider) -> None:
        """Initialize a single provider."""
        if provider not in PROVIDER_CLASSES:
            raise ValueError(f"Unknown provider: {provider}. Valid options: {list(PROVIDER_CLASSES.keys())}")

        api_key = self.api_keys.get(provider) or self.api_keys.get(
            {"openai": "openai", "gemini": "google", "anthropic": "anthropic"}.get(provider)
        )
        extra = {"host": self.openrouter_host} if provider == "openrouter" else {}
        self._checkers[provider] = PROVIDER_CLASSES[provider](
            **extra,
            api_key=api_key,
            model=self.models.get(provider),
            scrape_methods=self.scrape_methods,
            max_searches=self.max_searches,
            max_fetches=self.max_fetches,
            reasoning_effort=self.reasoning_effort,
            media_limits=self.media_limits,
        )

    def get_provider(self, provider: Provider | None = None) -> BaseFactChecker:
        """
        Get a specific provider's fact-checker.

        Args:
            provider: Provider name. If None, returns the default provider.

        Returns:
            The fact-checker instance for the specified provider.
        """
        if provider is None:
            provider = self._default_provider

        if provider not in self._checkers:
            raise ValueError(f"Provider '{provider}' not initialized. Available: {list(self._checkers.keys())}")

        return self._checkers[provider]

    def check_claim(
        self,
        claim: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        claim_date: str | datetime | None = None,
        provider: Provider | None = None,
        claim_id: int | str | None = None,
        blocked_urls: list[str] | None = None,
        cutoff_date: str | datetime | None = None,
        blocked_scopes: list[str] | None = None,
    ) -> FactCheckResult:
        """
        Fact-check a claim using a single provider.

        Args:
            claim: The claim text to verify.
            image_paths: Optional list of image file paths to include.
            video_paths: Optional list of video file paths to include.
                         Note: Only Gemini supports native video processing.
                         Other providers will extract frames.
            claim_date: Date of the claim (ISO format string or datetime).
                        web_search only returns content published before this day, unless
                        cutoff_date is given.
            provider: Provider to use. If None, uses the default provider.
            claim_id: ID of the claim, which tags the log lines of its tool calls and retries.
            blocked_urls: URLs the tools never show or fetch, e.g. the claim's own fact-check,
                          which would give the answer away.
            cutoff_date: web_search only returns content published before this day, e.g. the
                         day the claim's fact-check was published.
            blocked_scopes: Hosts, or hosts and paths, whose URLs the tools never show or fetch,
                            e.g. those under which the fact-check's publisher publishes
                            fact-checks ("leadstories.com/hoax-alert").

        Returns:
            FactCheckResult with verdict, reasoning, citations, etc.
        """
        checker = self.get_provider(provider)
        return checker.check_claim(claim, image_paths, video_paths, claim_date, claim_id=claim_id,
                                   blocked_urls=blocked_urls, cutoff_date=cutoff_date,
                                   blocked_scopes=blocked_scopes)
