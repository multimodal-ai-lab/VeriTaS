from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal
from typing import Type

from ezmm import MultimodalSequence

from veritas import models_config, selfhosted
from veritas.common.prompt import Prompt
from veritas.util.parsing import extract_last

logger = logging.getLogger("VeriTaS")

FAMILIES = ["gpt", "claude", "llama", "gemini"]


# Extended thinking budgets (in tokens) per reasoning effort level.
THINKING_BUDGETS = {
    "none": 0,
    "low": 1024,
    "medium": 4096,
    "high": 16384,
    "max": 65536,
}


@dataclass
class Generation:
    """One model response.

    - `content`: the answer, i.e. the text or the parsed object.
    - `reasoning`: the model's own reasoning trace, taken from the dedicated field
      the provider's API returns it in (OpenAI reasoning items, Anthropic thinking
      blocks, Gemini thought parts) - *not* parsed out of the answer text.
      None when the model or the request produced no reasoning.
    """

    content: str | Any | None = None
    reasoning: str | None = None


def as_generation(value: "Generation | str | Any | None") -> Generation:
    """Normalizes whatever `_generate` returned into a `Generation`, so providers
    that do not expose reasoning keep working unchanged."""
    return value if isinstance(value, Generation) else Generation(content=value)


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
            extract: Literal["last_code_span"] = None,
            retries: int = 20,
            resolve_media: bool = True,
            return_reasoning: bool = False,
            **kwargs
    ) -> MultimodalSequence | Any | tuple[MultimodalSequence | Any, str | None]:
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
        :param return_reasoning: If True, returns a `(response, reasoning)` tuple, where
            `reasoning` is the model's own reasoning trace as reported by the provider's
            API, or None if it reported none. Defaults to False, which returns just the
            response.

        :return: A `MultimodalSequence` object or a response in the specified format,
            depending on the parameters provided. A `(response, reasoning)` tuple if
            `return_reasoning` is set.

        :raises RateLimitError: If the rate limit is exceeded and all retries are exhausted.
        :raises QuotaExceededError: If the model's quota is exceeded.
        """

        generation = Generation()

        if not resolve_media:
            prompt = str(prompt)

        start = time.time()
        for attempt in range(retries):
            try:
                generation = as_generation(
                    await self._generate(prompt, response_format, **kwargs))
                logger.debug(f"Response generation took {time.time() - start:.1f} seconds for {self.specifier}.")
                if generation.content is None:
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
        response = generation.content
        if response is not None:
            if not response_format:
                match extract:
                    case "last_code_span": response = extract_last(response, delimiter="`")
                response = MultimodalSequence(response)
        if return_reasoning:
            return response, generation.reasoning
        return response

    async def _generate(
            self, prompt: Prompt | str,
            response_format: Any | None = None,
            reasoning_effort: Literal["none", "low", "medium", "high"] | None = None,
            **kwargs
    ) -> "Generation | str | Any":
        """Provider-specific call. Should return a `Generation` carrying both the
        answer and the reasoning the provider reported; a bare answer is accepted
        too and is treated as having no reasoning."""
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


#: Provider prefixes `init_model` understands. A prefix outside this set is part of
#: the model name itself (e.g. OpenAI fine-tunes like "ft:gpt-4o:org:id").
KNOWN_PROVIDERS = frozenset({
    "openai", "anthropic", "gemini", "google", "vertex", "vertexai", "gcp",
    "selfhosted", "openai_compat", "llama",
})

#: Prefixes that `init_model` routes to the respective class. Used to check that a
#: configured singleton names the provider its class implements.
OPENAI_PROVIDERS = ("openai",)
GEMINI_PROVIDERS = ("gemini", "google", "vertex", "vertexai", "gcp")


def split_known_provider(specifier: str) -> tuple[str | None, str]:
    """Like `_split_provider`, but only splits off a prefix naming a known provider,
    so that colons inside a model name survive."""
    provider, model = _split_provider(specifier)
    if provider in KNOWN_PROVIDERS:
        return provider, model
    return None, specifier


def configured_model_name(key: str,
                          default: str,
                          providers: tuple[str, ...],
                          config: dict | None = None) -> str:
    """The model name configured under `models.<key>` in config.yaml, or `default`.

    The singletons in `veritas.models` are provider-specific (transcription and
    embeddings exist only for OpenAI, video input only for Gemini), so the value may
    carry a provider prefix only if it is one of `providers`; the prefix is stripped,
    since the singleton's class already fixes the provider. A value naming another
    provider (e.g. `gpt_strong: "anthropic:..."`) raises a `ValueError` rather than
    silently constructing a model of the wrong class.

    :param key: The key below `models:`, e.g. "gpt_strong".
    :param default: The name used when the key is missing or null.
    :param providers: The prefixes accepted for this singleton, e.g. `OPENAI_PROVIDERS`.
    :param config: The `models:` section. Defaults to the one from config.yaml.
    """
    config = models_config if config is None else config
    value = config.get(key)
    if value is None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"config.yaml: `models.{key}` must be a non-empty model name, "
                         f"got {value!r}.")
    provider, name = split_known_provider(value.strip())
    if provider is not None and provider not in providers:
        raise ValueError(
            f"config.yaml: `models.{key}` is '{value}', but this model must be served by "
            f"{' / '.join(providers)} (it is used through that provider's API). Name a "
            f"model of that provider, or configure the other model where arbitrary "
            f"providers are accepted (e.g. `models.ensemble` or the `gold_evidence.*_model` keys).")
    return name


def matching_singleton(specifier: str,
                       singletons: "list[tuple[Model, tuple[str, ...]]]") -> "Model | None":
    """Returns the singleton that `specifier` denotes, if any, so that it can be
    reused instead of constructing a second instance of the same model.

    Each singleton comes with the provider prefixes of its class. A specifier matches
    if its model name equals the singleton's and it carries no prefix or one of those."""
    provider, name = split_known_provider(specifier.strip())
    for model, providers in singletons:
        if name == model.specifier and (provider is None or provider in providers):
            return model
    return None


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
