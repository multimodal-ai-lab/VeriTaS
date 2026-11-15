"""Provider implementations for the fact-checking baseline."""

from .base import BaseFactChecker, DEFAULT_REASONING_EFFORT
from .openai import OpenAIFactChecker
from .gemini import GeminiFactChecker
from .anthropic import AnthropicFactChecker
from .selfhosted import SelfhostedFactChecker
from .openrouter import OpenRouterFactChecker

__all__ = [
    "BaseFactChecker",
    "DEFAULT_REASONING_EFFORT",
    "OpenAIFactChecker",
    "GeminiFactChecker",
    "AnthropicFactChecker",
    "SelfhostedFactChecker",
    "OpenRouterFactChecker",
]
