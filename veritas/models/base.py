from __future__ import annotations

import asyncio
import logging
import time
from datetime import timedelta
from typing import Any, Literal
from typing import Type

from ezmm import MultimodalSequence

from veritas import selfhosted
from veritas.common.prompt import Prompt
from veritas.util.parsing import extract_last

logger = logging.getLogger("VeriTaS")

FAMILIES = ["gpt", "claude", "llama", "gemini"]


class Model:
    """Common interface for model providers."""
    specifier: str
    system_prompt = Prompt("veritas/prompts/system_prompt.md")
    base_backoff = 60  # seconds

    def __init__(self, specifier: str):
        self.specifier = specifier

    async def generate(
            self, prompt: Prompt | str,
            response_format: Any | None = None,
            extract: bool = None,
            retries: int = 20,
            resolve_media: bool = True,
            **kwargs
    ) -> MultimodalSequence | Any:
        """
        Submit the given prompt to the model and return the (extracted) response.

        This method attempts to generate a response based on the provided prompt and
        configuration options. It uses multiple retries and backoff mechanisms to handle
        rate-limiting, quota exhaustion, and server errors during the generation process.
        It also supports response extraction for specific cases.

        :param prompt: The input Prompt object or string for the generation process.
        :param response_format: The desired JSON format of the generated response. If None,
            the response will not be explicitly formatted.
        :param extract: A flag indicating whether to extract specific elements from the
            response, such as "last_code_span". If None, no extraction will occur.
        :param retries: The number of retry attempts in case of transient failures, such
            as rate limits or server errors. Defaults to 20.
        :param resolve_media: A boolean indicating whether to resolve multimedia elements
            in the prompt. Defaults to True. If set to False, the prompt will be treated
            as a plain string.

        :return: A `MultimodalSequence` object or a response in the specified format,
            depending on the parameters provided.

        :raises RateLimitError: If the rate limit is exceeded and all retries are exhausted.
        :raises QuotaExceededError: If the model's quota is exceeded.
        """

        response = None

        if not resolve_media:
            prompt = str(prompt)

        start = time.time()
        for attempt in range(retries):
            try:
                response = await self._generate(prompt, response_format, **kwargs)
                logger.debug(f"Response generation took {time.time() - start:.1f} seconds for {self.specifier}.")
                if response is None:
                    logger.info(f"Model {self.specifier} returned an empty response.")
                break
            except RateLimitError:
                msg = f"⚠️ Rate limit hit for model {self.specifier}."
                logger.warning(msg)
                if attempt == retries - 1:
                    raise RateLimitError(msg)
            except QuotaExceededError:
                raise QuotaExceededError(f"❌ Quota exceeded for model {self.specifier}.")

            await self._backoff(attempt)

        # Format the response and return
        if response is not None:
            if not response_format:
                match extract:
                    case "last_code_span": response = extract_last(response, delimiter="`")
                response = MultimodalSequence(response)
        return response

    async def _generate(
            self, prompt: Prompt | str,
            response_format: Any | None = None,
            reasoning_effort: Literal["none", "low", "medium", "high"] | None = None,
            **kwargs
    ) -> str | Any:
        raise NotImplementedError

    async def _backoff(self, attempt: int):
        """Exponential waiting time for rate limiting."""
        sleep_time = 2 ** attempt * self.base_backoff
        logger.warning(f"Backing off {self.specifier} for "
                       f"{str(timedelta(seconds=sleep_time))} hours...")
        await asyncio.sleep(sleep_time)

    async def embed(self, text: str) -> list[float]:
        """Computes embeddings for the given text."""
        raise NotImplementedError

    @property
    def family(self) -> str | None:
        """Returns the family name of the model, if available."""
        return get_family(self.specifier)


def get_family(specifier: str) -> str | None:
    for family in FAMILIES:
        if family in specifier:
            return family
    return None


class RateLimitError(Exception):
    """Max allowed requests per time unit has been exceeded for the given model."""


class QuotaExceededError(Exception):
    """Available capacity of model calls has ben exhausted."""


def _split_provider(model_name: str) -> tuple[str | None, str]:
    """Split a name like 'openai:gpt-4o' into (provider, model).

    If no explicit provider is present, returns (None, name).
    """
    if ":" in model_name:
        provider, model = model_name.split(":", 1)
        return provider.lower(), model
    return None, model_name


def _infer_model_type(name: str) -> Type[Model]:
    lname = name.lower()
    if lname.startswith("gpt-") or lname.startswith("o3") or lname.startswith("gpt4"):
        return GPT  # type: ignore[return-value]
    if "claude" in lname:
        return Claude  # type: ignore[return-value]
    if lname.startswith("gemini") or "gemini-" in lname:
        # Use public Gemini REST provider for all gemini* models
        return Gemini  # type: ignore[return-value]
    # Default to OpenAI to preserve current behavior; can be adjusted
    return GPT  # type: ignore[return-value]


def init_model(model_name: str) -> Model:
    """Return a model instance inferred from the given model name.

    Examples:
      - "openai:gpt-4o"
      - "anthropic:claude-3-5-sonnet-20240620"
      - "gemini:gemini-1.5-pro"
      - "gpt-4o" (inferred OpenAI)
      - "claude-3-haiku" (inferred Anthropic)
    """
    from veritas.models.gpt import GPT
    from veritas.models.claude import Claude
    from veritas.models.gemini import Gemini

    provider, model = _split_provider(model_name)
    if provider == "openai":
        return GPT(model)
    if provider == "anthropic":
        return Claude(model)
    if provider in ("gemini", "google"):
        # Route to Gemini REST implementation
        return Gemini(model)
    if provider in ("vertex", "vertexai", "gcp"):
        # Backwards-compat: map vertex* to Gemini REST too
        return Gemini(model)
    if provider in ("selfhosted", "openai_compat", "llama"):
        # Self-hosted OpenAI-compatible endpoint (e.g., vLLM, llama.cpp proxy)
        return GPT(model, base_url=selfhosted.get("url"))

    # Fallback
    model_cls = _infer_model_type(model)
    return model_cls(model)
