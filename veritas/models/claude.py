from __future__ import annotations

import logging
from typing import Any
import asyncio

from anthropic import AsyncAnthropic
from anthropic.types import (
    MessageParam,
    TextBlockParam,
    ImageBlockParam, Message, TextBlock,
)
from ezmm import Image, Video

from veritas import api_secrets
from veritas.models.gpt import gpt_transcribe
from veritas.models.base import Model
from veritas.common.prompt import Prompt

logging.getLogger("anthropic").setLevel(logging.WARNING)


def _format_image_anthropic(b64: str) -> dict:
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
    }


async def to_anthropic_payload(user_prompt: Prompt | str) -> list[MessageParam]:
    """Build a strongly-typed Anthropic payload (messages).

    - `system` must be a string.
    - `messages` must be a list of MessageParam with content blocks of
      type TextBlockParam | ImageBlockParam.
    """
    if isinstance(user_prompt, str):
        return [{"type": "text", "text": user_prompt}]

    content_blocks: list[TextBlockParam | ImageBlockParam] = []
    for item in user_prompt:
        if isinstance(item, str):
            if item.strip():
                content_blocks.append({"type": "text", "text": item})  # type: ignore[arg-type]
        elif isinstance(item, Image):
            img_b64 = item.get_base64_encoded()
            content_blocks.append({"type": "text", "text": item.reference})  # type: ignore[arg-type]
            content_blocks.append(_format_image_anthropic(img_b64))
        elif isinstance(item, Video):
            content_blocks.append({"type": "text", "text": item.reference})  # type: ignore[arg-type]
            from veritas.pipeline import n_frames_per_video
            frames = item.get_base64_encoded(n_frames=n_frames_per_video)
            content_blocks.extend(
                [_format_image_anthropic(frame) for frame in frames]
            )
            # Get transcription
            with open(item.file_path, "rb") as f:
                transcript = await gpt_transcribe.transcribe(file=f)
            if transcript:
                content_blocks.append({"type": "text", "text": f"Video transcript: _{transcript}_"})
        else:
            content_blocks.append({"type": "text", "text": str(item)})  # type: ignore[arg-type]

    messages: list[MessageParam] = [
        {
            "role": "user",
            "content": content_blocks,  # type: ignore[assignment]
        }
    ]
    return messages


class Claude(Model):
    """Claude LLM from Anthropic."""

    client: AsyncAnthropic

    def __init__(self, specifier: str):
        api_key = api_secrets.get("anthropic")
        if not api_key:
            raise ValueError("No Anthropic API key provided. Add 'anthropic' to config.yaml")
        self.client = AsyncAnthropic(api_key=api_key)
        super().__init__(specifier)

    async def _generate(
            self, prompt: Prompt | str,
            response_format: Any | None = None,
            max_tokens: int = 2048,
            **kwargs
    ) -> str | Any:
        messages = await to_anthropic_payload(prompt)

        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        completion = await self.client.messages.create(
            model=self.specifier,
            system=str(self.system_prompt),
            messages=messages,
            max_tokens=max_tokens,
            **filtered_kwargs,
        )
        if isinstance(completion, Message):
            return "".join(part.text for part in completion.content
                           if isinstance(part, TextBlock))


if __name__ == "__main__":
    claude = Claude("claude-sonnet-4-20250514")
    prompt = Prompt(text="<image:126394> Describe what you see.")
    response = asyncio.run(claude.generate(prompt))
    print(response)
