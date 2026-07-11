"""Unified fact-checker that supports multiple providers."""

import inspect
from datetime import datetime
from typing import Literal

from .common.search import ScrapeMode
from .common.types import FactCheckResult, LabelScheme
from .providers import (
    BaseFactChecker,
    OpenAIFactChecker,
    GeminiFactChecker,
    PerplexityFactChecker,
    OpenAICustomSearchFactChecker,
    GeminiCustomSearchFactChecker,
    SelfhostedFactChecker,
    AnthropicFactChecker,
    AnthropicCustomSearchFactChecker,
)


Provider = Literal["openai", "gemini", "perplexity", "selfhosted", "anthropic"]
SevenBinPredictionMode = Literal["direct", "two_step"]

# Default models for each provider
DEFAULT_MODELS = {
    "openai": "gpt-5.2",
    "gemini": "gemini-2.5-flash",
    "perplexity": "sonar-pro",
    "selfhosted": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "anthropic": "claude-sonnet-4-6",
}

# Provider class mapping - standard (built-in search)
PROVIDER_CLASSES = {
    "openai": OpenAIFactChecker,
    "gemini": GeminiFactChecker,
    "perplexity": PerplexityFactChecker,
    # Self-hosted models have no native search, always use custom
    "selfhosted": SelfhostedFactChecker,
    "anthropic": AnthropicFactChecker,
}

# Provider class mapping - custom search (with date filtering)
PROVIDER_CLASSES_CUSTOM_SEARCH = {
    "openai": OpenAICustomSearchFactChecker,
    "gemini": GeminiCustomSearchFactChecker,
    # Perplexity already has good date filtering, no custom version needed
    "perplexity": PerplexityFactChecker,
    # Self-hosted models only support custom search
    "selfhosted": SelfhostedFactChecker,
    "anthropic": AnthropicCustomSearchFactChecker,
}


class UnifiedFactChecker:
    """
    Unified fact-checker that can use any supported provider.

    This class provides a single interface for fact-checking using
    OpenAI, Gemini, Perplexity, or self-hosted Llama as the backend.

    Example usage:
        # Single provider with built-in search
        fc = UnifiedFactChecker(provider="openai")
        result = fc.check_claim("The Earth is flat")

        # Single provider with custom search (date filtering + content retrieval)
        fc = UnifiedFactChecker(provider="gemini", custom_search=True)
        result = fc.check_claim("The Earth is flat", claim_date="2024-01-15")

        # Multiple providers
        fc = UnifiedFactChecker(providers=["openai", "gemini", "perplexity"])
        results = fc.check_claim_all_providers("The Earth is flat")

        # Use 7-class label scheme (with uncertainty)
        from baselines.common.types import get_label_scheme
        fc = UnifiedFactChecker(provider="openai", label_scheme=get_label_scheme(7))
    """

    def __init__(
        self,
        provider: Provider | None = None,
        providers: list[Provider] | None = None,
        model: str | None = None,
        models: dict[Provider, str] | None = None,
        api_keys: dict[str, str] | None = None,
        custom_search: bool = False,
        use_search: bool = True,
        label_scheme: LabelScheme | None = None,
        seven_bin_prediction_mode: SevenBinPredictionMode = "direct",
        scrape_mode: ScrapeMode = "lite",
        scrape_methods: list[str] | str | None = "firecrawl",
    ):
        """
        Initialize the unified fact-checker.

        Args:
            provider: Single provider to use (openai, gemini, perplexity, or selfhosted).
            providers: List of providers to initialize (for multi-provider mode).
            model: Model to use for single provider mode.
            models: Dict mapping provider names to model identifiers.
            api_keys: Dict mapping provider/service names to API keys.
                      Keys: "openai", "google", "perplexity"
            custom_search: If True, use custom search instead of built-in
                          provider search. Enables:
                          - Date filtering (only results before claim date)
                          - Full page content retrieval
            use_search: If True (default), use web search. If False, use only
                       parametric knowledge (no search tools will be used).
            label_scheme: Label scheme to use (3-class or 7-class). Defaults to 3-class.
            seven_bin_prediction_mode: For 7-class schemes, "direct" asks for a
                                      combined label; "two_step" asks for
                                      direction + certainty in one response.
            scrape_mode: For custom_search providers, how to fetch page content -
                        "lite" (fast), "scrapemm" (full), or "none".
            scrape_methods: For scrape_mode="scrapemm", which scrapeMM backends to use
                        (subset of integrations/firecrawl/decodo, or "auto"). Default
                        ["firecrawl"]. Custom-search providers only.

        Note: Specify either `provider` or `providers`, not both.
        """
        self.api_keys = api_keys or {}
        self.models = models or {}
        self.custom_search = custom_search
        self.use_search = use_search
        self.scrape_mode = scrape_mode
        self.label_scheme = label_scheme
        self.seven_bin_prediction_mode = seven_bin_prediction_mode
        self.scrape_methods = scrape_methods
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
            # Default to standard providers (not custom)
            providers_to_init = ["openai", "gemini", "perplexity"]

        # Initialize requested providers
        for p in providers_to_init:
            self._init_provider(p)

        self._default_provider = providers_to_init[0] if providers_to_init else None

    def _init_provider(self, provider: Provider) -> None:
        """Initialize a single provider."""
        # Select the appropriate class based on custom_search flag
        if self.custom_search:
            provider_classes = PROVIDER_CLASSES_CUSTOM_SEARCH
        else:
            provider_classes = PROVIDER_CLASSES

        if provider not in provider_classes:
            raise ValueError(f"Unknown provider: {provider}. Valid options: {list(provider_classes.keys())}")

        provider_class = provider_classes[provider]
        model = self.models.get(provider, DEFAULT_MODELS[provider])
        api_key = self.api_keys.get(provider) or self.api_keys.get(
            {"openai": "openai", "gemini": "google", "perplexity": "perplexity", "anthropic": "anthropic"}.get(provider)
        )

        # Build kwargs based on provider type
        kwargs = {
            "model": model,
            "use_search": self.use_search,
            "label_scheme": self.label_scheme,
            "seven_bin_prediction_mode": self.seven_bin_prediction_mode,
        }
        if api_key:
            kwargs["api_key"] = api_key

        # Only custom-search providers accept scrape_mode / scrape_methods;
        # pass them when supported.
        provider_params = inspect.signature(provider_class.__init__).parameters
        if "scrape_mode" in provider_params:
            kwargs["scrape_mode"] = self.scrape_mode
        if "scrape_methods" in provider_params:
            kwargs["scrape_methods"] = self.scrape_methods

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
                        If custom_search=True, search results will be filtered
                        to only include content from before this date.
            provider: Provider to use. If None, uses the default provider.

        Returns:
            FactCheckResult with verdict, reasoning, citations, etc.
        """
        checker = self.get_provider(provider)
        return checker.check_claim(claim, image_paths, video_paths, claim_date)

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
