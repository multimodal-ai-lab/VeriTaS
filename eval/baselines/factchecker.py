"""Unified fact-checker that supports multiple providers."""

from datetime import datetime
from typing import Literal

from .common.tools import DEFAULT_MAX_SEARCHES, DEFAULT_MAX_FETCHES
from .common.types import FactCheckResult
from .providers import (
    BaseFactChecker,
    OpenAIFactChecker,
    GeminiFactChecker,
    AnthropicFactChecker,
    SelfhostedFactChecker,
)
from .providers.base import extract_justification


Provider = Literal["openai", "gemini", "anthropic", "selfhosted"]

# Default models for each provider
DEFAULT_MODELS = {
    "openai": "gpt-5.2",
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-sonnet-4-6",
    "selfhosted": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
}

# Provider class mapping
PROVIDER_CLASSES = {
    "openai": OpenAIFactChecker,
    "gemini": GeminiFactChecker,
    "anthropic": AnthropicFactChecker,
    "selfhosted": SelfhostedFactChecker,
}

DEFAULT_PROVIDERS: list[Provider] = ["openai", "gemini", "anthropic"]


class UnifiedFactChecker:
    """
    Unified fact-checker that can use any supported provider.

    Every provider runs the same baseline: the model verifies the claim with the
    web_search tool (results restricted to before the claim date) and the fetch_url
    tool, and answers on the 7-class label scheme in two steps (DIRECTION + CERTAINTY).

    Example usage:
        # Single provider
        fc = UnifiedFactChecker(provider="gemini")
        result = fc.check_claim("The Earth is flat", claim_date="2024-01-15")

        # Multiple providers
        fc = UnifiedFactChecker(providers=["openai", "gemini", "anthropic"])
        results = fc.check_claim_all_providers("The Earth is flat")
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
    ):
        """
        Initialize the unified fact-checker.

        Args:
            provider: Single provider to use (openai, gemini, anthropic, or selfhosted).
            providers: List of providers to initialize (for multi-provider mode).
            model: Model to use for single provider mode.
            models: Dict mapping provider names to model identifiers.
            api_keys: Dict mapping provider/service names to API keys.
                      Keys: "openai", "google", "anthropic", "selfhosted"
            scrape_methods: Which scrapeMM backends fetch_url uses, in order (subset of
                        integrations/browser/firecrawl/decodo, or "auto"). Default "auto".
            max_searches: Maximum number of web_search calls per claim.
            max_fetches: Maximum number of fetch_url calls per claim.

        Note: Specify either `provider` or `providers`, not both.
        """
        self.api_keys = api_keys or {}
        self.models = models or {}
        self.scrape_methods = scrape_methods
        self.max_searches = max_searches
        self.max_fetches = max_fetches
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

        self._default_provider = providers_to_init[0] if providers_to_init else None

    def _init_provider(self, provider: Provider) -> None:
        """Initialize a single provider."""
        if provider not in PROVIDER_CLASSES:
            raise ValueError(f"Unknown provider: {provider}. Valid options: {list(PROVIDER_CLASSES.keys())}")

        provider_class = PROVIDER_CLASSES[provider]
        model = self.models.get(provider, DEFAULT_MODELS[provider])
        api_key = self.api_keys.get(provider) or self.api_keys.get(
            {"openai": "openai", "gemini": "google", "anthropic": "anthropic"}.get(provider)
        )

        kwargs = {
            "model": model,
            "scrape_methods": self.scrape_methods,
            "max_searches": self.max_searches,
            "max_fetches": self.max_fetches,
        }
        if api_key:
            kwargs["api_key"] = api_key

        self._checkers[provider] = provider_class(**kwargs)

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

    @property
    def available_providers(self) -> list[Provider]:
        """Get list of initialized providers."""
        return list(self._checkers.keys())

    def check_claim(
        self,
        claim: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        claim_date: str | datetime | None = None,
        provider: Provider | None = None,
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
                        web_search only returns content published before this day.
            provider: Provider to use. If None, uses the default provider.

        Returns:
            FactCheckResult with verdict, reasoning, citations, etc.
        """
        checker = self.get_provider(provider)
        result = checker.check_claim(claim, image_paths, video_paths, claim_date)
        if not result.justification:
            # `reasoning` holds the raw model response for every provider.
            result.justification = extract_justification(result.reasoning or "")
        return result

    def check_claim_all_providers(
        self,
        claim: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        claim_date: str | datetime | None = None,
    ) -> dict[Provider, FactCheckResult]:
        """
        Fact-check a claim using all initialized providers.

        Args:
            claim: The claim text to verify.
            image_paths: Optional list of image file paths to include.
            video_paths: Optional list of video file paths to include.
            claim_date: Date of the claim (ISO format string or datetime).

        Returns:
            Dict mapping provider names to FactCheckResults.
        """
        results = {}
        for provider in self._checkers:
            try:
                results[provider] = self.check_claim(
                    claim, image_paths, video_paths, claim_date, provider
                )
            except Exception as e:
                # Store error as result
                results[provider] = FactCheckResult(
                    verdict="Unknown",
                    reasoning=f"Error: {str(e)}",
                    provider=provider,
                )
        return results

    def check_claim_with_consensus(
        self,
        claim: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        claim_date: str | datetime | None = None,
        min_agreement: int = 2,
    ) -> tuple[FactCheckResult | None, dict[Provider, FactCheckResult]]:
        """
        Fact-check a claim using all providers and determine consensus.

        Args:
            claim: The claim text to verify.
            image_paths: Optional list of image file paths to include.
            video_paths: Optional list of video file paths to include.
            claim_date: Date of the claim (ISO format string or datetime).
            min_agreement: Minimum number of providers that must agree
                          for a consensus verdict.

        Returns:
            Tuple of (consensus_result, all_results) where consensus_result
            is None if no consensus was reached.
        """
        all_results = self.check_claim_all_providers(claim, image_paths, video_paths, claim_date)

        # Count verdicts
        verdict_counts: dict[str, int] = {}
        for result in all_results.values():
            verdict = result.verdict
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        # Find consensus
        consensus_verdict = None
        for verdict, count in verdict_counts.items():
            if count >= min_agreement:
                consensus_verdict = verdict
                break

        if consensus_verdict is None:
            return None, all_results

        # Build consensus result
        agreeing_providers = [
            p for p, r in all_results.items() if r.verdict == consensus_verdict
        ]
        combined_citations = []
        combined_reasoning_parts = []

        for provider in agreeing_providers:
            result = all_results[provider]
            combined_reasoning_parts.append(f"[{provider.upper()}]: {result.reasoning}")
            for citation in result.citations:
                if citation not in combined_citations:
                    combined_citations.append(citation)

        consensus_result = FactCheckResult(
            verdict=consensus_verdict,
            reasoning="\n\n".join(combined_reasoning_parts),
            citations=combined_citations,
            provider=",".join(agreeing_providers),
            model=",".join(all_results[p].model for p in agreeing_providers),
        )

        return consensus_result, all_results
