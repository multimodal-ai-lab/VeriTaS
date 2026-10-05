"""OpenAI fact-checker provider.

The model verifies the claim with the web_search and fetch_url tools via OpenAI's
function calling (chat completions API).
"""

import json
import os
from datetime import datetime
from pathlib import Path

from openai import OpenAI

# Import API key from Veritas config
try:
    from veritas import api_secrets
    VERITAS_OPENAI_KEY = api_secrets.get("openai") if api_secrets else None
except ImportError:
    VERITAS_OPENAI_KEY = None

from ..common.types import FactCheckResult
from ..common.media import encode_image_base64, extract_video_frames
from ..common.tools import openai_tools, DEFAULT_MAX_SEARCHES, DEFAULT_MAX_FETCHES
from .base import BaseFactChecker, MAX_TURNS


class OpenAIFactChecker(BaseFactChecker):
    """Fact-checker using OpenAI with the web_search and fetch_url tools."""

    provider_name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5.2",
        scrape_methods: list[str] | str | None = "auto",
        max_searches: int = DEFAULT_MAX_SEARCHES,
        max_fetches: int = DEFAULT_MAX_FETCHES,
    ):
        """
        Initialize the OpenAI fact-checker.

        Args:
            api_key: OpenAI API key. If None, uses config/env var.
            model: Model to use (must support function calling).
            scrape_methods: Which scrapeMM backends fetch_url uses (subset of
                            integrations/browser/firecrawl/decodo, or "auto").
            max_searches: Maximum number of web_search calls per claim.
            max_fetches: Maximum number of fetch_url calls per claim.
        """
        super().__init__(
            api_key=api_key,
            model=model,
            scrape_methods=scrape_methods,
            max_searches=max_searches,
            max_fetches=max_fetches,
        )
        self.client = self._create_client(api_key)

    def _create_client(self, api_key: str | None) -> OpenAI:
        """Create OpenAI client."""
        if api_key:
            return OpenAI(api_key=api_key)

        key_to_use = VERITAS_OPENAI_KEY or os.environ.get("OPENAI_API_KEY")

        if not key_to_use:
            raise ValueError(
                "OpenAI API key required. Either:\n"
                "  1. Add it to config.yaml (api_secrets.openai)\n"
                "  2. Set OPENAI_API_KEY environment variable\n"
                "  3. Pass api_key parameter"
            )
        return OpenAI(api_key=key_to_use)

    def check_claim(
        self,
        claim: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        claim_date: str | datetime | None = None,
    ) -> FactCheckResult:
        """
        Fact-check a claim using OpenAI with the web_search and fetch_url tools.

        Args:
            claim: The claim text to verify.
            image_paths: Optional list of image file paths.
            video_paths: Optional list of video file paths (frames extracted).
            claim_date: Date of the claim for temporal filtering.

        Returns:
            FactCheckResult with verdict, reasoning, citations.
        """
        session = self._new_tool_session(claim_date)

        # Collect images (including video frames)
        all_image_paths = list(image_paths) if image_paths else []
        temp_frame_paths = []

        if video_paths:
            for video_path in video_paths:
                frames = extract_video_frames(video_path, max_frames=5)
                all_image_paths.extend(frames)
                temp_frame_paths.extend(frames)

        try:
            messages = self._build_initial_messages(claim, all_image_paths, claim_date)
            tools = openai_tools()
            total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

            for turn in range(MAX_TURNS):
                # The last turn disables tool use to force a verdict
                tool_choice = "none" if turn == MAX_TURNS - 1 else "auto"
                response = self._call_with_retry(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                )

                # Accumulate usage
                if response.usage:
                    total_usage["input_tokens"] += response.usage.prompt_tokens
                    total_usage["output_tokens"] += response.usage.completion_tokens
                    total_usage["total_tokens"] += response.usage.total_tokens

                message = response.choices[0].message

                # Add assistant message to conversation
                messages.append(message)

                # The model is done once it stops calling tools
                if not message.tool_calls:
                    break

                for tool_call in message.tool_calls:
                    try:
                        args = json.loads(tool_call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = None
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": session.call(tool_call.function.name, args),
                    })

            # Extract final response
            final_content = message.content or ""
            verdict = self._extract_verdict(final_content)

            # Add any URLs from response text
            citations = list(session.citations)
            for url in self._extract_urls_from_text(final_content):
                if url not in citations:
                    citations.append(url)

            return FactCheckResult(
                verdict=verdict,
                reasoning=final_content,
                citations=citations,
                model=self.model,
                provider=self.provider_name,
                usage=total_usage,
            )

        finally:
            # Clean up temp files
            for temp_path in temp_frame_paths:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass

    def _build_initial_messages(
        self,
        claim: str,
        image_paths: list[str],
        claim_date: str | datetime | None,
    ) -> list[dict]:
        """Build the initial messages for the conversation."""
        messages = [{"role": "system", "content": self._system_prompt()}]

        # Build user message content
        content = []

        # Add images
        for img_path in image_paths:
            img_data = encode_image_base64(img_path)
            if img_data:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": img_data}
                })

        content.append({"type": "text", "text": self._user_prompt(claim, claim_date)})

        messages.append({"role": "user", "content": content})

        return messages
