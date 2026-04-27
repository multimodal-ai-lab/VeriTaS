"""Unified baselines package for fact-checking with multiple providers."""

from .common.types import Verdict, FactCheckResult, LABELS, LabelScheme, get_label_scheme
from .providers import (
    BaseFactChecker,
    OpenAIFactChecker,
    GeminiFactChecker,
    PerplexityFactChecker,
)
from .factchecker import UnifiedFactChecker

__all__ = [
    # Types
    "Verdict",
    "FactCheckResult",
    "LABELS",
    "LabelScheme",
    "get_label_scheme",
    # Providers (for direct use if needed)
    "BaseFactChecker",
    "OpenAIFactChecker",
    "GeminiFactChecker",
    "PerplexityFactChecker",
    # Unified interface (recommended)
    "UnifiedFactChecker",
]
