"""Gemini fact-checker provider."""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import types

# Import API key from Veritas config
try:
    from veritas import google_key as VERITAS_GOOGLE_KEY
except ImportError:
    VERITAS_GOOGLE_KEY = None

from ..common.types import FactCheckResult, LabelScheme
from ..common.media import get_mime_type
from .base import BaseFactChecker


# Video MIME types supported by Gemini
VIDEO_MIME_TYPES = {
    ".mp4": "video/mp4",
    ".mpeg": "video/mpeg",
    ".mpg": "video/mpg",
    ".mov": "video/mov",
    ".avi": "video/avi",
    ".flv": "video/x-flv",
    ".webm": "video/webm",
    ".wmv": "video/wmv",
    ".3gp": "video/3gpp",
    ".3gpp": "video/3gpp",
    ".m4v": "video/mp4",
}


class GeminiFactChecker(BaseFactChecker):
    """Fact-checker using Google Gemini with Google Search grounding and native video support."""

    provider_name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        use_search: bool = True,
        label_scheme: LabelScheme | None = None,
        seven_bin_prediction_mode: Literal["direct", "two_step"] = "direct",
    ):
        """
        Initialize the Gemini fact-checker.

        Args:
            api_key: Google API key. If None, uses GOOGLE_API_KEY env var.
            model: Model to use. Options include:
                - "gemini-2.5-flash": Latest, with native video support
                - "gemini-2.0-flash": Fast and capable
                - "gemini-1.5-pro": More capable for complex tasks
            use_search: If True (default), use Google Search grounding. If False,
                       use only parametric knowledge.
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

    def _create_client(self, api_key: str | None) -> genai.Client:
        """Create Gemini client."""
        if api_key:
            return genai.Client(api_key=api_key)

        # Try Veritas config first, then environment variable
        key_to_use = VERITAS_GOOGLE_KEY or os.environ.get("GOOGLE_API_KEY")

        if not key_to_use:
            raise ValueError(
                "Google API key required. Either:\n"
                "  1. Add it to config/globals.yaml (google: <key>)\n"
                "  2. Set GOOGLE_API_KEY environment variable\n"
                "  3. Pass api_key parameter"
            )
        return genai.Client(api_key=key_to_use)

    def check_claim(
        self,
        claim: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        claim_date: str | datetime | None = None,
    ) -> FactCheckResult:
        """
        Fact-check a claim using Gemini with Google Search grounding and native video support.

        Args:
            claim: The claim text to verify.
            image_paths: Optional list of image file paths to include.
            video_paths: Optional list of video file paths to include (native support).
            claim_date: Date of the claim (ISO format string or datetime).

        Returns:
            FactCheckResult with verdict, reasoning, citations, etc.
        """
        # Build message content parts
        contents = []

        # Add images first if provided
        if image_paths:
            for img_path in image_paths:
                img_part = self._load_image(img_path)
                if img_part:
                    contents.append(img_part)

        # Add videos natively (Gemini 2.5+ supports native video)
        if video_paths:
            for video_path in video_paths:
                video_part = self._load_video(video_path)
                if video_part:
                    contents.append(video_part)

        # Format the prompt using label-scheme-aware helpers
        prompt_text = self._get_user_prompt(claim, claim_date)
        system_prompt = self._get_system_prompt()

        # Add the text prompt
        contents.append(prompt_text)

        # Configure generation with Google Search grounding (if search enabled)
        # WARNING: Google Search grounding does NOT support API-level date filtering.
        # Temporal constraints are enforced via prompt instruction only (soft constraint).
        # For hard temporal filtering, use Perplexity provider instead.
        config_kwargs = {
            "system_instruction": system_prompt,
        }
        if self.use_search:
            config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]

        config = types.GenerateContentConfig(**config_kwargs)

        # Call API with retry logic
        response = self._call_with_retry(
            self.client.models.generate_content,
            model=self.model,
            contents=contents,
            config=config,
        )

        assert response is not None, "Gemini API call failed after retries."

        # Extract response content and citations
        response_content = self._extract_response_content(response)
        citations = self._extract_citations(response)
        verdict = self._extract_verdict(response_content)

        # Extract usage info
        usage_info = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage_info = {
                "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0) or 0,
                "total_tokens": getattr(response.usage_metadata, "total_token_count", 0) or 0,
            }

        return FactCheckResult(
            verdict=verdict,
            reasoning=response_content,
            citations=citations,
            model=self.model,
            provider=self.provider_name,
            usage=usage_info,
        )

    def _load_image(self, image_path: str) -> types.Part | None:
        """Load an image file and return as Gemini Part."""
        path = Path(image_path)
        if not path.exists():
            return None

        mime_type = get_mime_type(path)

        with open(path, "rb") as f:
            image_data = f.read()

        return types.Part.from_bytes(data=image_data, mime_type=mime_type)

    def _load_video(self, video_path: str) -> types.Part | None:
        """
        Load a video file and return as Gemini Part for native video processing.

        For videos > 20MB, uses the File API for upload.
        For smaller videos, uses inline data.
        """
        path = Path(video_path)
        if not path.exists():
            return None

        suffix = path.suffix.lower()
        mime_type = VIDEO_MIME_TYPES.get(suffix, "video/mp4")

        file_size = path.stat().st_size
        size_mb = file_size / (1024 * 1024)

        if size_mb > 20:
            # Use File API for large videos
            try:
                uploaded_file = self.client.files.upload(file=str(path))
                return uploaded_file
            except Exception as e:
                print(f"    Warning: Failed to upload video via File API: {e}")
                return None
        else:
            # Use inline data for smaller videos
            with open(path, "rb") as f:
                video_data = f.read()

            return types.Part(
                inline_data=types.Blob(data=video_data, mime_type=mime_type)
            )

    def _extract_response_content(self, response) -> str:
        """Extract text content from Gemini response, ignoring non-text parts."""
        # Don't use response.text as it warns when there are non-text parts
        # Instead, directly extract text from parts
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and candidate.content:
                parts = candidate.content.parts
                if parts:
                    text_parts = []
                    for part in parts:
                        # Only extract text parts, skip other part types
                        if hasattr(part, "text") and part.text:
                            text_parts.append(part.text)
                    if text_parts:
                        return "\n".join(text_parts)

        return ""

    def _extract_citations(self, response) -> list[str]:
        """Extract citations/URLs from Gemini response with grounding."""
        citations = []

        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "grounding_metadata") and candidate.grounding_metadata:
                grounding = candidate.grounding_metadata

                # Get grounding chunks (sources)
                if hasattr(grounding, "grounding_chunks") and grounding.grounding_chunks:
                    for chunk in grounding.grounding_chunks:
                        if hasattr(chunk, "web") and chunk.web:
                            if hasattr(chunk.web, "uri"):
                                citations.append(chunk.web.uri)

                # Try search_entry_point for web search results
                if hasattr(grounding, "search_entry_point") and grounding.search_entry_point:
                    if hasattr(grounding.search_entry_point, "rendered_content"):
                        html_content = grounding.search_entry_point.rendered_content
                        url_pattern = r'href=["\']([^"\']+)["\']'
                        found_urls = re.findall(url_pattern, html_content)
                        for url in found_urls:
                            if url.startswith("http") and url not in citations:
                                citations.append(url)

        # Also extract URLs from response text
        text = self._extract_response_content(response)
        for url in self._extract_urls_from_text(text):
            if url not in citations:
                citations.append(url)

        return citations
