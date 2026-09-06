from __future__ import annotations

import logging
from typing import Any, Literal
import asyncio

from ezmm import Item, Video, Image
from google.genai import Client
from google.genai.errors import ClientError, ServerError
from google.genai.types import GenerateContentConfig, FileState, ThinkingConfig
from pydantic import BaseModel

from veritas import api_secrets
from veritas.common.prompt import Prompt
from veritas.models.base import Generation, Model, RateLimitError, QuotaExceededError, THINKING_BUDGETS

logger = logging.getLogger("VeriTaS")

logging.getLogger("google_genai").setLevel(logging.ERROR)


class Gemini(Model):
    def __init__(self, specifier: str):
        api_key = api_secrets.get("google")
        if not api_key:
            raise ValueError(
                "No Gemini API key provided. Add 'google' to config.yaml."
            )
        self.client = Client(api_key=api_key, vertexai=False)
        super().__init__(specifier)

    async def _generate(
            self, prompt: Prompt | str,
            response_format: Any | None = None,
            reasoning_effort: Literal["none", "low", "medium", "high"] | None = None,
            **kwargs
    ) -> Generation | None:
        # Track files uploaded during this call to clean them up afterwards
        uploaded_names: set[str] = set()
        contents = await self.prepare_gemini_message(prompt, uploaded_names=uploaded_names)

        # Compose configuration dict
        config = dict()
        if reasoning_effort:
            thinking_budget = THINKING_BUDGETS[reasoning_effort]
            config["thinking_config"] = ThinkingConfig(thinking_budget=thinking_budget)

        if isinstance(response_format, BaseModel):
            config.update(dict(
                response_mime_type="application/json",
                response_json_schema=response_format.model_json_schema(),
            ))
        else:
            config.update(dict(
                system_instruction=str(self.system_prompt),
                response_modalities=["TEXT"],
            ))

        response = None
        try:
            response = await self.client.aio.models.generate_content(
                model=self.specifier,
                contents=contents,
                config=GenerateContentConfig(**config),
            )
        except ClientError as e:
            if e.code == 429:
                if "You exceeded your current quota, please check your plan and billing details" in e.message:
                    raise QuotaExceededError()
                else:
                    raise RateLimitError()
            else:
                raise
        except ServerError:
            logger.debug(f"Google server error encountered for model {self.specifier}.")
            raise  # The error doesn't seem to resolve after more tries

        finally:
            # Best-effort cleanup of files uploaded for this request
            if uploaded_names:
                for name in uploaded_names:
                    try:
                        await self.client.aio.files.delete(name=name)
                    except Exception:
                        pass

        if response:
            reasoning = self._extract_reasoning(response)
            text = self._extract_text(response)
            if response_format:
                try:
                    content = response_format.model_validate_json(str(text))
                except Exception:
                    content = None
            else:
                content = text
            return Generation(content=content, reasoning=reasoning)

    @staticmethod
    def _iter_parts(response):
        for candidate in getattr(response, "candidates", None) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", None) or []:
                yield part

    @classmethod
    def _extract_reasoning(cls, response) -> str | None:
        """Returns the model's reasoning from Gemini's dedicated thought parts -
        never parsed out of the answer text.

        Only present when the request enabled `include_thoughts` (see
        `reasoning_effort`)."""
        parts = [part.text for part in cls._iter_parts(response)
                 if getattr(part, "thought", False) and getattr(part, "text", None)]
        return "\n\n".join(parts) or None

    @classmethod
    def _extract_text(cls, response) -> str | None:
        """The answer text, with thought parts excluded.

        `response.text` concatenates all parts, so once `include_thoughts` is on it
        can carry the reasoning into the answer and corrupt downstream parsing."""
        parts = [part.text for part in cls._iter_parts(response)
                 if not getattr(part, "thought", False) and getattr(part, "text", None)]
        if parts:
            return "".join(parts)
        try:
            return response.text
        except Exception:
            return None

    async def prepare_gemini_message(self, prompt: Prompt | str, uploaded_names: set[str] | None = None) -> list:
        """Uploads media to the Google Files API:
        https://ai.google.dev/gemini-api/docs/files
        "The Files API lets you store up to 20 GB of files per project, with a
        per-file maximum size of 2 GB. Files are stored for 48 hours."
        """
        if isinstance(prompt, str):
            return [prompt]

        contents = []
        for item in prompt:
            if isinstance(item, str):
                contents.append(item)

            elif isinstance(item, Item):
                try:
                    file = await self.client.aio.files.upload(file=item.file_path)
                    if uploaded_names is not None and file.name:
                        # Track only files uploaded during this call for cleanup
                        uploaded_names.add(str(file.name))
                except ClientError as e:
                    # Catch and escalate resource exhausted error
                    if e.code == 429:
                        msg = "Bytes quota exceeded for the Google AI Files API. Please remove files or wait 48h."
                        logger.error(msg)
                        raise QuotaExceededError(msg)
                    raise
                contents.extend([item.reference, file])

                if isinstance(item, Video):
                    # Wait until file is ready (processing might take some time)
                    while file.state == FileState.PROCESSING:
                        await asyncio.sleep(1)
                        file = await self.client.aio.files.get(name=str(file.name))
                    if file.state == FileState.FAILED:
                        raise ValueError(f"Unable to upload video {item.reference}: {file.state.name}")

        return contents


def to_gemini_setup(system_prompt: Prompt) -> dict[str, Any]:
    """Gemini supports a system instruction on model construction."""
    return {"system_instruction": str(system_prompt)}


def to_gemini_user_content(user_prompt: Prompt, extra_instructions: str | None = None) -> list[dict[str, Any]] | str:
    """Convert Prompt to Gemini content. Keep simple text for now.

    Note: If multimodal content is needed later, expand this to include
    image parts according to Prompt content.
    """
    parts: list[dict[str, Any]] = []
    has_media = False
    for item in user_prompt:
        if isinstance(item, str):
            if item.strip():
                parts.append({"text": item})
        elif isinstance(item, Image):
            has_media = True
            img_b64 = item.get_base64_encoded()
            parts.append({"text": item.reference})
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img_b64}})
        else:
            parts.append({"text": str(item)})
    if extra_instructions:
        parts.append({"text": extra_instructions})
    # If there is no media and only a single text part, Gemini accepts a plain string
    if not has_media and len(parts) == 1 and "text" in parts[0]:
        return parts[0]["text"]
    return parts


gemini_strong = Gemini("gemini-3.1-pro-preview")
gemini_cheap = Gemini("gemini-3-flash-preview")

if __name__ == "__main__":
    prompt = Prompt(text="<video:11> Describe what you see.")
    response = asyncio.run(gemini_strong.generate(prompt, reasoning_effort="medium", resolve_media=False))
    print(response)
