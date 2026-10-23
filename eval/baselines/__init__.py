"""Fact-checking baseline for the VeriTaS benchmark, with support for multiple providers."""

from .common.media import MediaLimits
from .common.types import Verdict7, FactCheckResult, FactCheckError, NoAnswerError, LabelScheme
from .providers import (
    BaseFactChecker,
    OpenAIFactChecker,
    GeminiFactChecker,
    AnthropicFactChecker,
    SelfhostedFactChecker,
    OpenRouterFactChecker,
)
from .factchecker import UnifiedFactChecker

__all__ = [
    # Types
    "Verdict7",
    "FactCheckResult",
    "FactCheckError",
    "NoAnswerError",
    "LabelScheme",
    "MediaLimits",
    # Providers (for direct use if needed)
    "BaseFactChecker",
    "OpenAIFactChecker",
    "GeminiFactChecker",
    "AnthropicFactChecker",
    "SelfhostedFactChecker",
    "OpenRouterFactChecker",
    # Unified interface (recommended)
    "UnifiedFactChecker",
]
