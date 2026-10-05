"""Fact-checking baseline for the VeriTaS benchmark, with support for multiple providers."""

from .common.types import Verdict, Verdict7, FactCheckResult, LABELS, LabelScheme, get_label_scheme
from .providers import (
    BaseFactChecker,
    OpenAIFactChecker,
    GeminiFactChecker,
    AnthropicFactChecker,
    SelfhostedFactChecker,
)
from .factchecker import UnifiedFactChecker

__all__ = [
    # Types
    "Verdict",
    "Verdict7",
    "FactCheckResult",
    "LABELS",
    "LabelScheme",
    "get_label_scheme",
    # Providers (for direct use if needed)
    "BaseFactChecker",
    "OpenAIFactChecker",
    "GeminiFactChecker",
    "AnthropicFactChecker",
    "SelfhostedFactChecker",
    # Unified interface (recommended)
    "UnifiedFactChecker",
]
