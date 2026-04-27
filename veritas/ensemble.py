"""Ensemble LLM support for querying multiple models, extracting and validating
their responses, and aggregating results."""

from __future__ import annotations

import asyncio
import logging
import traceback
from dataclasses import dataclass
from typing import Any, Iterable, TypeVar, Callable

import anthropic
import openai
from ezmm import MultimodalSequence

from veritas.common.prompt import Prompt
from veritas.models import Model, QuotaExceededError, RateLimitError, gpt_strong, gemini_strong, init_model

T = TypeVar('T')

logger = logging.getLogger("VeriTaS")


@dataclass
class ModelResponse:
    """Response container for each member of the ensemble.

    - `model`: The model that returned this response.
    - `output`: The model's output (text or parsed object).
    - `error`: Exception string if the call failed.
    """

    model: Model
    output: MultimodalSequence | Any | None
    error: Exception | None = None


class Ensemble:
    """Central registry to query multiple LLMs for the same prompt.

    Usage:
        ensemble = Ensemble(["openai:gpt-4o", "anthropic:claude-3-5-sonnet-20240620"])
        results = ensemble.generate_all(prompt)

    Notes:
        - Does not modify or integrate into the existing pipeline/model.
        - Uses the provider factory to resolve names into concrete providers.
        - Both sync and async APIs are available.
    """

    def __init__(self, models: Iterable[str] | None = None):
        self._members: dict[str, Model] = {}
        if models:
            for name in models:
                self.register(name)

    def register(self, model_or_specifier: str | Model):
        """Register a provider by model name or Model object."""
        if isinstance(model_or_specifier, str):
            model = init_model(model_or_specifier)
        else:
            model = model_or_specifier

        # Ensure unique key by appending a counter if needed
        name = model.specifier
        base = name
        i = 2
        while name in self._members:
            name = f"{base}#{i}"
            i += 1

        self._members[name] = model
        logger.info(f"✅ Successfully registered LMM {name} as {type(model).__name__}.")

    def unregister(self, name: str) -> None:
        self._members.pop(name, None)

    @property
    def model_names(self) -> list[str]:
        return list(self._members.keys())

    @property
    def models(self) -> list[Model]:
        return list(self._members.values())

    async def generate(
            self,
            prompt: Prompt,
            aggregation_fn: Callable = None,
            response_format: Any | None = None,
            response_extraction_fn: Callable[[ModelResponse], Any] | None = None,
            *,
            tries: int = 3,
            models: list[str] | None = None,
            **kwargs,
    ) -> list[ModelResponse | None] | Any:
        """Call all ensemble members concurrently.

        :param prompt: The prompt to be submitted to all models.
        :param response_format: Required response format for all model outputs.
        :param response_extraction_fn: Function to extract data from the responses. Can be used
            for validation. If validation fails, the corresponding model will be executed again.
        :param aggregation_fn: Function for merging the individual responses from the models.
        :param tries: Number of retries for each model when validation fails.
        :param models: The ensemble members to call. If None, all available models will be called.
        """
        if aggregation_fn is None:
            aggregation_fn = lambda x: x
        if response_extraction_fn is None:
            response_extraction_fn = lambda x: x

        if models is None:
            models = self.model_names
        elif len(models) == 0:
            return None
        assert isinstance(models, list)

        gathered_responses: dict[str, ModelResponse] = {}

        t = 0
        while len(gathered_responses) < len(models) and t < tries:
            t += 1
            models_to_call = [model for model in models if model not in gathered_responses]
            responses = await self._call_models(models_to_call,
                                                prompt,
                                                response_format=response_format,
                                                **kwargs)

            # Validate (and extract) responses
            for response in responses:
                model_name = response.model.specifier
                if response.error:
                    if isinstance(response.error, QuotaExceededError):
                        raise response.error  # Terminate execution immediately
                    if isinstance(response.error, anthropic.BadRequestError) or \
                            "Audio file processing failed" in str(response.error) or \
                            "Audio file might be corrupted or unsupported" in str(response.error):
                        logger.debug(f"Omitting model {model_name} due to {response.error.__class__.__name__}")
                        models.remove(model_name)
                    if t == tries:
                        logger.warning(f"Failed to get response from model {model_name} "
                                       f"after {t} tries: {response.error}")
                else:
                    try:
                        extracted = response_extraction_fn(response)
                        gathered_responses[model_name] = extracted
                    except AssertionError as e:
                        logger.debug(f"Invalid response from model {model_name}: {e}")
                    except ValueError as e:
                        logger.debug(f"ValueError during response extraction: {e}")
                    except Exception:
                        import traceback
                        logger.warning(f"Failed to extract response from model {model_name}"
                                       f" after {t} tries: {traceback.format_exc()}\nResponse: {response.output}")

        # Ensure the responses are complete
        if len(gathered_responses) == 0:
            raise RuntimeError("Failed to get response from all models.")
        if len(gathered_responses) < len(models):
            missing_models = [model for model in models if model not in gathered_responses]
            logger.warning(f"Missing response(s) of {', '.join(missing_models)} after {tries} tries.")

        # Aggregate responses
        responses = list(gathered_responses.values())
        return aggregation_fn(responses)

    async def _call_models(self,
                           models: list[str],
                           prompt: Prompt,
                           response_format: Any | None,
                           **kwargs) -> list[ModelResponse]:
        """Call the specified models concurrently."""

        async def _call(name: str):
            model = self._members[name]
            try:
                out = await model.generate(
                    prompt, response_format=response_format, **kwargs
                )
                return ModelResponse(model=model, output=out)
            except (anthropic.InternalServerError, openai.InternalServerError) as e:
                logger.info(f"Internal server error in model {name}: {e}")
                return ModelResponse(model=model, output=None, error=e)
            except anthropic.BadRequestError as e:
                if "You have reached your specified API usage limits" in str(e):
                    raise QuotaExceededError(f"Anthropic API usage limit reached.")
                else:
                    logger.info(f"Invalid request to model {name}: {e}")
                    return ModelResponse(model=model, output=None, error=e)
            except anthropic.APIStatusError as e:
                if e.status_code == 413:
                    # Raise to avoid retries
                    raise RuntimeError(f"Input exceeds the context window of {name}.")
                return ModelResponse(model=model, output=None, error=e)
            except QuotaExceededError as e:
                raise QuotaExceededError(f"Quota exceeded for model {name}: {e}") from e
            except RateLimitError as e:
                raise RateLimitError(f"Rate limit reached for model {name}.") from e
            except openai.BadRequestError as e:
                if "Your input exceeds the context window of this model" in e.message:
                    # Raise to avoid retries
                    raise RuntimeError(f"Input exceeds the context window of {name}.")
                logger.info(f"Invalid request to model {name}: {e}")
                return ModelResponse(model=model, output=None, error=e)
            except Exception as e:
                logger.error(f"Failed to call model {name}: {e}")
                logger.debug(traceback.format_exc())
                return ModelResponse(model=model, output=None, error=e)

        # Call models concurrently. Use return_exceptions=True so that if one
        # coroutine raises, the others are still awaited (avoids
        # "coroutine was never awaited" RuntimeWarnings).
        tasks = [_call(name) for name in models]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Re-raise the first fatal exception (quota / rate limit / context window),
        # otherwise filter to actual ModelResponse objects.
        responses: list[ModelResponse] = []
        first_exc: BaseException | None = None
        for name, result in zip(models, results):
            if isinstance(result, ModelResponse):
                responses.append(result)
            elif isinstance(result, BaseException):
                if first_exc is None:
                    first_exc = result
            # result is None -> drop silently
        if first_exc is not None and not responses:
            raise first_exc
        if first_exc is not None and isinstance(first_exc, (QuotaExceededError, RateLimitError)):
            raise first_exc
        return responses


ensemble = Ensemble()  # global singleton instance
ensemble.register(gpt_strong)
ensemble.register("anthropic:claude-sonnet-4-5-20250929")
ensemble.register("selfhosted:meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")
ensemble.register(gemini_strong)

if __name__ == "__main__":
    prompt = Prompt(text="<image:126394> Describe what you see.")
    responses = asyncio.run(ensemble.generate(prompt))
    for response in responses:
        if isinstance(response, ModelResponse):
            print(f"Response of model {response.model.specifier}:\n{response.output}\n")
