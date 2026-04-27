import asyncio
import logging
from datetime import datetime
from typing import Any, Literal

import openai
from ezmm import Image, Video
from openai import InternalServerError, AsyncOpenAI, PermissionDeniedError
from openai.types import FileObject
from openai.types.responses import Response, ResponseReasoningItem
from openai.types.upload_create_params import ExpiresAfter

from veritas import api_secrets, selfhosted
from veritas.common import Prompt
from veritas.models.base import Model, RateLimitError, QuotaExceededError

logger = logging.getLogger("VeriTaS")
logging.getLogger("openai").setLevel(logging.WARNING)


class GPT(Model):
    """Wrapper for all OpenAI-compatible LLMs, i.e., not only OpenAI GPT models
    but also self-hosted models like Llama.

    :param base_url: If self-hosted, the base URL of the API of the self-hosted LLM.

    When using as self-hosted, make sure to have `selfhosted_url` and `selfhosted_key`
    configured in `config/globals.yaml`.

    Example model specifier for self-hosted:
    `selfhosted:meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8`."""

    client: AsyncOpenAI

    def __init__(self, specifier: str, base_url: str = None):
        self.base_url = base_url
        self.connect()
        super().__init__(specifier)

    def connect(self):
        api_key = api_secrets.get("openai")
        if not api_key and not self.base_url:
            raise ValueError("No OpenAI API key provided. Add it to globals.yaml or pass a"
                             " base_url to the OpenAIModel constructor.")
        self.client = AsyncOpenAI(api_key=api_key, base_url=self.base_url)

    async def _generate(
            self, prompt: Prompt | str,
            response_format: Any | None = None,
            reasoning_effort: Literal["none", "low", "medium", "high"] | None = None,
            **kwargs
    ) -> str | Any:
        # TODO: Truncate prompt to match context length

        messages = await self._prepare_gpt_messages(prompt)

        reasoning = dict(effort=reasoning_effort, summary="detailed") if reasoning_effort else None

        try:
            if response_format:
                response = await self.client.responses.parse(
                    model=self.specifier, input=messages, text_format=response_format, reasoning=reasoning, **kwargs
                )
                # self._log_reasoning(response)
                return response.output_parsed

            else:
                response = await self.client.responses.create(
                    model=self.specifier, input=messages, reasoning=reasoning, **kwargs
                )
                # self._log_reasoning(response)
                return response.output_text

        except InternalServerError as e:
            if e.status_code == 502:
                logger.critical(f"❌ Could not reach {self.specifier} at {self.base_url}!")
            else:
                logger.warning(f"⚠️ Unexpected error from OpenAI API: {e}")

        except PermissionDeniedError as e:
            logger.warning(f"⚠️ PermissionDeniedError encountered at OpenAI API: {e}")

        except openai.RateLimitError as e:
            if e.type == "insufficient_quota":
                raise QuotaExceededError()
            else:
                raise RateLimitError()

    async def embed(self, text: str) -> list[float]:
        try:
            response = await self.client.embeddings.create(
                model=self.specifier,
                input=text,
                encoding_format="float",
            )
            return response.data[0].embedding
        except openai.RateLimitError:
            raise RateLimitError()
        except openai.BadRequestError as e:
            logger.error(f"Failed to compute embeddings with {self.specifier}: {e}")
            raise

    def _log_reasoning(self, response: Response) -> None:
        """Gets and logs the LLM's reasoning."""
        reasoning = None
        reasoning_items = [r for r in response.output if isinstance(r, ResponseReasoningItem)]
        if reasoning_items:
            summary = [s.text for reasoning_item in reasoning_items for s in reasoning_item.summary]
            reasoning = "\n\n".join(summary)
        if not reasoning:
            reasoning = response.output_text
        logger.debug(f"{self.specifier} Reasoning:\n{reasoning}")

    async def transcribe(self, file, **kwargs) -> str | None:
        try:
            response = await self.client.audio.transcriptions.create(model=self.specifier, file=file, **kwargs)
            return response.text
        except openai.BadRequestError as e:
            logger.debug(f"Could not transcribe video {file.name}: {e}")
            return None

    async def _prepare_gpt_messages(self, prompt: Prompt | str) -> list[dict]:
        content = prompt if isinstance(prompt, str) else await preprocess_prompt(prompt)
        return [
            dict(role="system", content=str(self.system_prompt)),
            dict(role="user", content=content),
        ]

    async def _upload_video(self, video: Video) -> FileObject:
        """WARNING: Videos not supported yet as of Sept 2025.
        Uploads a video to OpenAI's file storage and returns the file reference.
        Checks if it was uploaded before and returns the cached reference if it was."""
        if hasattr(video, "openai_file_reference"):
            uploaded_file: FileObject = video.openai_file_reference
            # Ensure it isn't expired
            if int(datetime.now().timestamp()) < uploaded_file.expires_at - 60:  # Buffer for 60s
                return uploaded_file

        uploaded_file = await self.client.files.create(
            file=open(video.file_path, "rb"),
            purpose="vision",
            expires_after=ExpiresAfter(anchor="created_at", seconds=60 * 60)
        )
        video.openai_file_reference = uploaded_file
        return uploaded_file


def _format_image(base64_encoded: str) -> dict:
    return {
        "type": "input_image",
        "detail": "auto",
        "image_url": f"data:image/jpeg;base64,{base64_encoded}"
    }


async def preprocess_prompt(prompt: Prompt) -> list[dict]:
    content_formatted = []

    for item in prompt:
        if isinstance(item, str):
            content_formatted.append({"type": "input_text", "text": item})

        elif isinstance(item, Image):
            image_encoded = item.get_base64_encoded()
            content_formatted.append({"type": "input_text", "text": item.reference})
            content_formatted.append(_format_image(image_encoded))

        elif isinstance(item, Video):
            content_formatted.append({"type": "input_text", "text": item.reference})
            from veritas.pipeline import n_frames_per_video
            frames = item.get_base64_encoded(n_frames=n_frames_per_video)
            content_formatted.extend(
                [_format_image(frame) for frame in frames]
            )
            # Get transcription
            try:
                with open(item.file_path, "rb") as f:
                    transcript = await gpt_transcribe.transcribe(file=f)
                if transcript:
                    content_formatted.append({"type": "input_text", "text": f"Video transcript: _{transcript}_"})
            except Exception as e:
                logger.info(f"Failed to get transcription for video {item.reference}: {e}")

            # TODO: Use this as soon as videos are supported
            # Skip videos that are larger than 10 MB (or use fallback-method)
            # video_size = item.file_path.stat().st_size
            # if video_size > 10 * 1024 * 1024:
            #     logger.warning(f"Skipping video {item.reference} because its size ({video_size / 1024 / 1024:.1f} MB)"
            #                    f" exceeds the maximum size (10 MB).")
            #     content_formatted.append({"type": "text", "text": f"{item.reference} (not available)"})
            #     continue
            # uploaded_file = await self._upload_video(item)
            # content_formatted.append(
            #     {"type": "file_reference", "file_reference": {"id": uploaded_file.id}}
            # )

    return content_formatted


gpt_strong = GPT("gpt-5.2")
gpt_cheap = GPT("gpt-5-mini")
gpt_nano = GPT("gpt-5-nano")
gpt_transcribe = GPT("gpt-4o-mini-transcribe")
text_embedder = GPT("text-embedding-3-large")

if __name__ == "__main__":
    llm = GPT(specifier="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
              base_url=selfhosted.get("url"))
    prompt = Prompt(text="<image:123456> Describe what you see.")
    response = asyncio.run(llm.generate(prompt))
    print(response)
