"""Gemini fact-checker provider.

The model verifies the claim with the web_search and fetch_url tools via Gemini's
function calling. Videos are passed natively.
"""

import os
import time
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import types

# Import API key from Veritas config
try:
    from veritas import api_secrets
    VERITAS_GOOGLE_KEY = api_secrets.get("google") if api_secrets else None
except ImportError:
    VERITAS_GOOGLE_KEY = None

from ..common.types import FactCheckResult
from ..common.media import get_mime_type
from ..common.tools import TOOL_DEFINITIONS, DEFAULT_MAX_SEARCHES, DEFAULT_MAX_FETCHES
from .base import BaseFactChecker, MAX_TURNS


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


def gemini_tool() -> types.Tool:
    """The tools in Gemini format."""
    return types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name=tool["name"],
            description=tool["description"],
            parameters_json_schema=tool["parameters"],
        )
        for tool in TOOL_DEFINITIONS
    ])


class GeminiFactChecker(BaseFactChecker):
    """Fact-checker using Gemini with the web_search and fetch_url tools."""

    provider_name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        scrape_methods: list[str] | str | None = "auto",
        max_searches: int = DEFAULT_MAX_SEARCHES,
        max_fetches: int = DEFAULT_MAX_FETCHES,
    ):
        """
        Initialize the Gemini fact-checker.

        Args:
            api_key: Google API key. If None, uses config/env var.
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

    def _create_client(self, api_key: str | None) -> genai.Client:
        """Create Gemini client."""
        if api_key:
            return genai.Client(api_key=api_key)

        key_to_use = VERITAS_GOOGLE_KEY or os.environ.get("GOOGLE_API_KEY")

        if not key_to_use:
            raise ValueError(
                "Google API key required. Either:\n"
                "  1. Add it to config.yaml (api_secrets.google)\n"
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
        Fact-check a claim using Gemini with the web_search and fetch_url tools.

        Args:
            claim: The claim text to verify.
            image_paths: Optional list of image file paths.
            video_paths: Optional list of video file paths (native support).
            claim_date: Date of the claim for temporal filtering.

        Returns:
            FactCheckResult with verdict, reasoning, citations.
        """
        session = self._new_tool_session(claim_date)

        # Build the user turn: media first, then the prompt
        user_parts = []
        for img_path in image_paths or []:
            img_part = self._load_image(img_path)
            if img_part:
                user_parts.append(img_part)
        for video_path in video_paths or []:
            video_part = self._load_video(video_path)
            if video_part:
                user_parts.append(video_part)
        user_parts.append(types.Part.from_text(text=self._user_prompt(claim, claim_date)))

        history = [types.Content(role="user", parts=user_parts)]

        tools = [gemini_tool()]
        config = types.GenerateContentConfig(
            system_instruction=self._system_prompt(),
            tools=tools,
        )
        # The last turn disables tool use to force a verdict
        final_config = types.GenerateContentConfig(
            system_instruction=self._system_prompt(),
            tools=tools,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=types.FunctionCallingConfigMode.NONE)
            ),
        )

        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        for turn in range(MAX_TURNS):
            response = self._call_with_retry(
                self.client.models.generate_content,
                model=self.model,
                contents=history,
                config=final_config if turn == MAX_TURNS - 1 else config,
            )

            # Accumulate usage
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                total_usage["prompt_tokens"] += getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                total_usage["completion_tokens"] += getattr(response.usage_metadata, "candidates_token_count", 0) or 0
                total_usage["total_tokens"] += getattr(response.usage_metadata, "total_token_count", 0) or 0

            # The model is done once it stops calling functions
            function_calls = self._get_function_calls(response)
            if not function_calls:
                break

            # Keep the whole conversation: the model's turn (incl. thought signatures)
            # followed by the function responses
            history.append(response.candidates[0].content)
            history.append(types.Content(role="user", parts=[
                types.Part.from_function_response(
                    name=func_call.name,
                    response={"result": session.call(func_call.name, dict(func_call.args or {}))},
                )
                for func_call in function_calls
            ]))

        # Extract final response
        final_content = self._extract_response_content(response) or ""
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

    def _get_function_calls(self, response) -> list[types.FunctionCall]:
        """Return the function calls of the response's first candidate."""
        if not getattr(response, "candidates", None):
            return []
        content = response.candidates[0].content
        if not content or not content.parts:
            return []
        return [part.function_call for part in content.parts if part.function_call]

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
        """Load a video file and return as Gemini Part."""
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
                # Wait for file to become ACTIVE (required before use)
                max_wait = 60  # seconds
                wait_interval = 2
                waited = 0
                while waited < max_wait:
                    file_info = self.client.files.get(name=uploaded_file.name)
                    if file_info.state.name == "ACTIVE":
                        return types.Part.from_uri(
                            file_uri=file_info.uri,
                            mime_type=file_info.mime_type or mime_type,
                        )
                    elif file_info.state.name == "FAILED":
                        print(f"    Warning: Video file processing failed")
                        return None
                    time.sleep(wait_interval)
                    waited += wait_interval
                print(f"    Warning: Video file not ready after {max_wait}s, skipping")
                return None
            except Exception as e:
                print(f"    Warning: Failed to upload video via File API: {e}")
                return None
        else:
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
                        # Only extract text parts, skip function_call and other parts
                        if hasattr(part, "text") and part.text:
                            text_parts.append(part.text)
                    if text_parts:
                        return "\n".join(text_parts)

        return ""
