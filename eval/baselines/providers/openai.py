"""OpenAI fact-checker provider."""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal

from openai import OpenAI

# Import API key from Veritas config
try:
    from veritas import openai_key as VERITAS_OPENAI_KEY
except ImportError:
    VERITAS_OPENAI_KEY = None

from ..common.types import FactCheckResult, LabelScheme
from ..common.media import encode_image_base64, extract_video_frames
from .base import BaseFactChecker


class OpenAIFactChecker(BaseFactChecker):
    """Fact-checker using OpenAI's GPT models with web search capability."""

    provider_name = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5.2",
        use_search: bool = True,
        label_scheme: LabelScheme | None = None,
        seven_bin_prediction_mode: Literal["direct", "two_step"] = "direct",
    ):
        """
        Initialize the OpenAI fact-checker.

        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            model: Model to use. Options include:
                - "gpt-5.2": Latest GPT-5.2 model
                - "gpt-4.1": GPT-4.1 model
            use_search: If True (default), use web search. If False, use only
                       parametric knowledge.
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

    def _create_client(self, api_key: str | None) -> OpenAI:
        """Create OpenAI client."""
        if api_key:
            return OpenAI(api_key=api_key)

        # Try Veritas config first, then environment variable
        key_to_use = VERITAS_OPENAI_KEY or os.environ.get("OPENAI_API_KEY")

        if not key_to_use:
            raise ValueError(
                "OpenAI API key required. Either:\n"
                "  1. Add it to config/globals.yaml (openai: <key>)\n"
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
        Fact-check a claim using OpenAI with web search and optional images.

        Args:
            claim: The claim text to verify.
            image_paths: Optional list of image file paths to include.
            video_paths: Optional list of video file paths (frames will be extracted).
            claim_date: Date of the claim (ISO format string or datetime).

        Returns:
            FactCheckResult with verdict, reasoning, citations, etc.
        """
        # Build message content using Responses API format
        content = []

        # Collect all images (including extracted video frames)
        all_image_paths = list(image_paths) if image_paths else []
        temp_frame_paths = []

        # Extract frames from videos (OpenAI doesn't support native video)
        if video_paths:
            for video_path in video_paths:
                frames = extract_video_frames(video_path, max_frames=5)
                all_image_paths.extend(frames)
                temp_frame_paths.extend(frames)

        try:
            # Add images
            for img_path in all_image_paths:
                img_data = encode_image_base64(img_path)
                if img_data:
                    content.append({
                        "type": "input_image",
                        "image_url": img_data
                    })

            # Format the prompt using label-scheme-aware helpers
            prompt_text = self._get_user_prompt(claim, claim_date)
            system_prompt = self._get_system_prompt()

            # Add the text prompt
            content.append({
                "type": "input_text",
                "text": prompt_text
            })

            # Build API request
            input_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ]

            # Call API with retry logic
            # WARNING: OpenAI's web_search_preview does NOT support API-level date filtering.
            # Temporal constraints are enforced via prompt instruction only (soft constraint).
            # For hard temporal filtering, use Perplexity provider instead.
            api_kwargs = {
                "model": self.model,
                "input": input_messages,
            }
            if self.use_search:
                api_kwargs["tools"] = [{"type": "web_search_preview"}]

            response = self._call_with_retry(
                self.client.responses.create,
                **api_kwargs,
            )

            assert response is not None, "OpenAI API call failed after retries."

            # Extract response content and citations
            response_content = self._extract_response_content(response)
            citations = self._extract_citations(response)
            verdict = self._extract_verdict(response_content)

            # Extract usage info
            usage_info = {}
            if hasattr(response, "usage") and response.usage:
                usage_info = {
                    "input_tokens": getattr(response.usage, "input_tokens", 0),
                    "output_tokens": getattr(response.usage, "output_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                }

            return FactCheckResult(
                verdict=verdict,
                reasoning=response_content,
                citations=citations,
                model=self.model,
                provider=self.provider_name,
                usage=usage_info,
            )
        finally:
            # Clean up temporary frame files
            for temp_path in temp_frame_paths:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass

    def _extract_response_content(self, response) -> str:
        """Extract text content from OpenAI response."""
        if hasattr(response, "output"):
            text_parts = []
            for item in response.output:
                if hasattr(item, "type"):
                    if item.type == "message":
                        if hasattr(item, "content"):
                            for content_item in item.content:
                                if hasattr(content_item, "text"):
                                    text_parts.append(content_item.text)
                    elif item.type == "text":
                        if hasattr(item, "text"):
                            text_parts.append(item.text)
            if text_parts:
                return "\n".join(text_parts)

        if hasattr(response, "output_text"):
            return response.output_text

        return str(response)

    def _extract_citations(self, response) -> list[str]:
        """Extract citations/URLs from OpenAI response."""
        citations = []

        if hasattr(response, "output"):
            for item in response.output:
                if hasattr(item, "type") and item.type == "web_search_call":
                    if hasattr(item, "results"):
                        for result in item.results:
                            if hasattr(result, "url"):
                                citations.append(result.url)

        # Also extract URLs from response text
        text = self._extract_response_content(response)
        for url in self._extract_urls_from_text(text):
            if url not in citations:
                citations.append(url)

        return citations
