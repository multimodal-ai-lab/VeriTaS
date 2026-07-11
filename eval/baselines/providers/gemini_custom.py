"""Gemini fact-checker provider with custom search tool.

This provider uses Gemini's function calling with a custom search tool that supports:
1. Date filtering: Only results from before the claim date
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import types

# Import API key from Veritas config
try:
    from veritas import api_secrets
    VERITAS_GOOGLE_KEY = api_secrets.get("google") if api_secrets else None
except ImportError:
    VERITAS_GOOGLE_KEY = None

from ..common.types import FactCheckResult, LabelScheme
from ..common.media import get_mime_type
from ..common.search import SearchService, GEMINI_SEARCH_TOOL_DECLARATION, ScrapeMode
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


class GeminiCustomSearchFactChecker(BaseFactChecker):
    """Fact-checker using Gemini with custom search tool supporting date filtering."""

    provider_name = "gemini_custom"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.5-flash",
        serpapi_key: str | None = None,
        max_search_calls: int = 5,
        scrape_content: bool = True,
        scrape_mode: ScrapeMode = "lite",
        scrape_methods: list[str] | str | None = "firecrawl",
        use_search: bool = True,
        label_scheme: LabelScheme | None = None,
        seven_bin_prediction_mode: Literal["direct", "two_step"] = "direct",
    ):
        """
        Initialize the Gemini fact-checker with custom search.

        Args:
            api_key: Google API key. If None, uses config/env var.
            model: Model to use (must support function calling).
            serpapi_key: SerpAPI key for search. If None, uses config/env var.
            max_search_calls: Maximum number of search calls per fact-check.
            scrape_content: Whether to scrape full page content.
            scrape_mode: Scraping method - "lite" (fast), "scrapemm" (full), or "none".
            scrape_methods: For scrape_mode="scrapemm", which scrapeMM backends to use
                       (subset of integrations/firecrawl/decodo, or "auto"). Default ["firecrawl"].
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
        # Only initialize search service if search is enabled
        if use_search:
            self.search_service = SearchService(
                serpapi_key=serpapi_key,
                scrape_mode=scrape_mode,
                max_content_length=5000,
                scrape_methods=scrape_methods,
            )
        else:
            self.search_service = None
        self.max_search_calls = max_search_calls
        self.scrape_content = scrape_content

    def _create_client(self, api_key: str | None) -> genai.Client:
        """Create Gemini client."""
        if api_key:
            return genai.Client(api_key=api_key)

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
        Fact-check a claim using Gemini with custom search tool.

        Args:
            claim: The claim text to verify.
            image_paths: Optional list of image file paths.
            video_paths: Optional list of video file paths (native support).
            claim_date: Date of the claim for temporal filtering.

        Returns:
            FactCheckResult with verdict, reasoning, citations.
        """
        # Build content parts
        contents = []

        # Add images
        if image_paths:
            for img_path in image_paths:
                img_part = self._load_image(img_path)
                if img_part:
                    contents.append(img_part)

        # Add videos (native support)
        if video_paths:
            for video_path in video_paths:
                video_part = self._load_video(video_path)
                if video_part:
                    contents.append(video_part)

        # Handle no-search mode (parametric knowledge only)
        if not self.use_search:
            prompt_text = self._get_user_prompt(claim, claim_date)
            system_prompt = self._get_system_prompt()

            contents.append(prompt_text)

            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
            )

            response = self._call_with_retry(
                self.client.models.generate_content,
                model=self.model,
                contents=contents,
                config=config,
            )

            usage_info = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage_info = {
                    "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0) or 0,
                    "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0) or 0,
                    "total_tokens": getattr(response.usage_metadata, "total_token_count", 0) or 0,
                }

            final_content = self._extract_response_content(response) or ""
            verdict = self._extract_verdict(final_content)

            return FactCheckResult(
                verdict=verdict,
                reasoning=final_content,
                citations=[],
                model=self.model,
                provider=self.provider_name,
                usage=usage_info,
            )

        # Build prompt (with search)
        prompt_text = self._get_custom_search_user_prompt(claim, claim_date)
        system_prompt = self._get_custom_search_system_prompt()

        contents.append(prompt_text)

        # Define the search tool for Gemini
        search_tool = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name="web_search",
                    description="Search the web for information. Results are limited to content published before the claim date.",
                    parameters=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "query": types.Schema(
                                type=types.Type.STRING,
                                description="The search query to find relevant information about the claim."
                            )
                        },
                        required=["query"]
                    )
                )
            ]
        )

        # Configure generation
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[search_tool],
        )

        # Run conversation with function calling loop
        citations = []
        search_count = 0
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        chat_history = []

        # Initial request
        response = self._call_with_retry(
            self.client.models.generate_content,
            model=self.model,
            contents=contents,
            config=config,
        )

        # Accumulate usage
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            total_usage["prompt_tokens"] += getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            total_usage["completion_tokens"] += getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            total_usage["total_tokens"] += getattr(response.usage_metadata, "total_token_count", 0) or 0

        # Handle function calling loop
        while self._has_function_calls(response):
            # Process function calls
            function_responses = []

            for candidate in response.candidates:
                if hasattr(candidate, "content") and candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call") and part.function_call:
                            func_call = part.function_call
                            if func_call.name == "web_search":
                                search_count += 1
                                if search_count > self.max_search_calls:
                                    result_text = "Search limit reached. Please provide your verdict based on the information gathered so far."
                                else:
                                    query = func_call.args.get("query", "")
                                    search_response = self.search_service.search(
                                        query=query,
                                        before_date=claim_date,
                                        num_results=10,
                                        max_scrape=2,
                                        scrape_content=self.scrape_content,
                                    )
                                    result_text = self.search_service.format_results_for_llm(search_response)

                                    # Show content status
                                    content_count = sum(1 for r in (search_response.results or []) if r.content)
                                    print(f"    Search: {len(search_response.results or [])} results, {content_count} with content")

                                    # Collect citations
                                    if search_response.results:
                                        for result in search_response.results:
                                            if result.url and result.url not in citations:
                                                citations.append(result.url)

                                function_responses.append(
                                    types.Part.from_function_response(
                                        name="web_search",
                                        response={"result": result_text}
                                    )
                                )

            if not function_responses:
                break

            # Build conversation history for continuation
            # Add the model's response with function calls
            chat_history = contents + [response.candidates[0].content]

            # Add function responses
            chat_history.append(types.Content(
                role="user",
                parts=function_responses
            ))

            # Continue conversation
            response = self._call_with_retry(
                self.client.models.generate_content,
                model=self.model,
                contents=chat_history,
                config=config,
            )

            # Accumulate usage
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                total_usage["prompt_tokens"] += getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                total_usage["completion_tokens"] += getattr(response.usage_metadata, "candidates_token_count", 0) or 0
                total_usage["total_tokens"] += getattr(response.usage_metadata, "total_token_count", 0) or 0

        # Extract final response
        final_content = self._extract_response_content(response) or ""
        verdict = self._extract_verdict(final_content)

        # Add any URLs from response text
        if final_content:
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

    def _has_function_calls(self, response) -> bool:
        """Check if response contains function calls."""
        if not hasattr(response, "candidates") or not response.candidates:
            return False

        for candidate in response.candidates:
            if hasattr(candidate, "content") and candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        return True
        return False

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
                import time
                uploaded_file = self.client.files.upload(file=str(path))
                # Wait for file to become ACTIVE (required before use)
                max_wait = 60  # seconds
                wait_interval = 2
                waited = 0
                while waited < max_wait:
                    file_info = self.client.files.get(name=uploaded_file.name)
                    if file_info.state.name == "ACTIVE":
                        return uploaded_file
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
