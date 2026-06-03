"""Anthropic fact-checker provider."""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

import anthropic as anthropic_sdk

# Import API key from Veritas config (if available)
try:
    from veritas import api_secrets
    VERITAS_ANTHROPIC_KEY = api_secrets.get("anthropic") if api_secrets else None
except ImportError:
    VERITAS_ANTHROPIC_KEY = None

from ..common.types import FactCheckResult, LabelScheme
from ..common.media import encode_image_base64, get_mime_type, extract_video_frames
from .base import BaseFactChecker


def _parse_data_uri(data_uri: str) -> tuple[str, str]:
    """Parse a data URI into (media_type, base64_data)."""
    # Format: data:<media_type>;base64,<data>
    match = re.match(r"data:([^;]+);base64,(.+)", data_uri, re.DOTALL)
    if match:
        return match.group(1), match.group(2)
    return "image/jpeg", data_uri


class AnthropicFactChecker(BaseFactChecker):
    """Fact-checker using Anthropic's Claude models with optional web search."""

    provider_name = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        use_search: bool = True,
        label_scheme: LabelScheme | None = None,
        seven_bin_prediction_mode: Literal["direct", "two_step"] = "direct",
    ):
        """
        Initialize the Anthropic fact-checker.

        Args:
            api_key: Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
            model: Model to use. Options include:
                - "claude-sonnet-4-6": Claude Sonnet 4.6 (default)
                - "claude-opus-4-6": Claude Opus 4.6 (most capable)
                - "claude-haiku-4-5-20251001": Claude Haiku 4.5 (fastest)
            use_search: If True (default), use built-in web search tool.
                       If False, use only parametric knowledge.
            label_scheme: Label scheme to use (3-class or 7-class). Defaults to 3-class.
            seven_bin_prediction_mode: For 7-class schemes, "direct" asks for a
                                      combined label; "two_step" asks for
                                      direction + certainty.
        """
        super().__init__(
            api_key=api_key,
            model=model,
            use_search=use_search,
            label_scheme=label_scheme,
            seven_bin_prediction_mode=seven_bin_prediction_mode,
        )
        self.client = self._create_client(api_key)

    def _create_client(self, api_key: str | None) -> anthropic_sdk.Anthropic:
        """Create Anthropic client."""
        if api_key:
            return anthropic_sdk.Anthropic(api_key=api_key)

        # Try Veritas config first, then environment variable
        key_to_use = VERITAS_ANTHROPIC_KEY or os.environ.get("ANTHROPIC_API_KEY")

        if not key_to_use:
            raise ValueError(
                "Anthropic API key required. Either:\n"
                "  1. Add it to config/globals.yaml (anthropic: <key>)\n"
                "  2. Set ANTHROPIC_API_KEY environment variable\n"
                "  3. Pass api_key parameter"
            )
        return anthropic_sdk.Anthropic(api_key=key_to_use)

    def check_claim(
        self,
        claim: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        claim_date: str | datetime | None = None,
    ) -> FactCheckResult:
        """
        Fact-check a claim using Claude with optional web search and images.

        Args:
            claim: The claim text to verify.
            image_paths: Optional list of image file paths to include.
            video_paths: Optional list of video file paths (frames will be extracted).
            claim_date: Date of the claim (ISO format string or datetime).

        Returns:
            FactCheckResult with verdict, reasoning, citations, etc.
        """
        # Collect all images (including extracted video frames)
        all_image_paths = list(image_paths) if image_paths else []
        temp_frame_paths = []

        # Extract frames from videos (Anthropic doesn't support native video)
        if video_paths:
            for video_path in video_paths:
                frames = extract_video_frames(video_path, max_frames=5)
                all_image_paths.extend(frames)
                temp_frame_paths.extend(frames)

        try:
            # Build user message content
            content = []

            # Add images
            for img_path in all_image_paths:
                img_data = encode_image_base64(img_path)
                if img_data:
                    media_type, b64_data = _parse_data_uri(img_data)
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64_data,
                        },
                    })

            # Add text prompt
            prompt_text = self._get_user_prompt(claim, claim_date)
            content.append({"type": "text", "text": prompt_text})

            system_prompt = self._get_system_prompt()

            # Build API request
            api_kwargs = {
                "model": self.model,
                "max_tokens": 8096,
                "system": system_prompt,
                "messages": [{"role": "user", "content": content}],
            }
            if self.use_search:
                api_kwargs["tools"] = [
                    {"type": "web_search_20250305", "name": "web_search", "max_uses": 5}
                ]

            response = self._call_with_retry(
                self.client.messages.create,
                **api_kwargs,
            )

            assert response is not None, "Anthropic API call failed after retries."

            response_text = self._extract_response_text(response)
            citations = self._extract_citations(response, response_text)
            verdict = self._extract_verdict(response_text)

            # Extract usage info
            usage_info = {}
            if hasattr(response, "usage") and response.usage:
                usage_info = {
                    "input_tokens": getattr(response.usage, "input_tokens", 0),
                    "output_tokens": getattr(response.usage, "output_tokens", 0),
                }

            return FactCheckResult(
                verdict=verdict,
                reasoning=response_text,
                citations=citations,
                model=self.model,
                provider=self.provider_name,
                usage=usage_info,
            )
        finally:
            for temp_path in temp_frame_paths:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass

    def _extract_response_text(self, response) -> str:
        """Extract text content from Anthropic response."""
        text_parts = []
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(block.text)
        return "\n".join(text_parts)

    def _extract_citations(self, response, response_text: str) -> list[str]:
        """Extract citations/URLs from Anthropic response."""
        citations = []

        # Extract URLs from web_search_tool_result blocks
        for block in response.content:
            block_type = getattr(block, "type", None)
            if block_type == "web_search_tool_result":
                content_items = getattr(block, "content", [])
                for item in content_items:
                    url = getattr(item, "url", None)
                    if url and url not in citations:
                        citations.append(url)

        # Also extract URLs from response text
        for url in self._extract_urls_from_text(response_text):
            if url not in citations:
                citations.append(url)

        return citations
