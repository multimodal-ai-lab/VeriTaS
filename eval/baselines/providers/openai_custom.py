"""OpenAI fact-checker provider with custom search tool.

This provider uses OpenAI's function calling with a custom search tool that supports:
1. Date filtering: Only results from before the claim date
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

from openai import OpenAI

# Import API key from Veritas config
try:
    from veritas import api_secrets
    VERITAS_OPENAI_KEY = api_secrets.get("openai") if api_secrets else None
except ImportError:
    VERITAS_OPENAI_KEY = None

from ..common.types import FactCheckResult, LabelScheme
from ..common.media import encode_image_base64, extract_video_frames
from ..common.search import SearchService, OPENAI_SEARCH_TOOL, ScrapeMode
from .base import BaseFactChecker


class OpenAICustomSearchFactChecker(BaseFactChecker):
    """Fact-checker using OpenAI with custom search tool supporting date filtering."""

    provider_name = "openai_custom"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-5.2",
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
        Initialize the OpenAI fact-checker with custom search.

        Args:
            api_key: OpenAI API key. If None, uses config/env var.
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
                scrape_methods=scrape_methods,
                max_content_length=5000,
            )
        else:
            self.search_service = None
        self.max_search_calls = max_search_calls
        self.scrape_content = scrape_content

    def _create_client(self, api_key: str | None) -> OpenAI:
        """Create OpenAI client."""
        if api_key:
            return OpenAI(api_key=api_key)

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
        Fact-check a claim using OpenAI with custom search tool.

        Args:
            claim: The claim text to verify.
            image_paths: Optional list of image file paths.
            video_paths: Optional list of video file paths (frames extracted).
            claim_date: Date of the claim for temporal filtering.

        Returns:
            FactCheckResult with verdict, reasoning, citations.
        """
        # Collect images (including video frames)
        all_image_paths = list(image_paths) if image_paths else []
        temp_frame_paths = []

        if video_paths:
            for video_path in video_paths:
                frames = extract_video_frames(video_path, max_frames=5)
                all_image_paths.extend(frames)
                temp_frame_paths.extend(frames)

        try:
            # Handle no-search mode (parametric knowledge only)
            if not self.use_search:
                messages = self._build_initial_messages_no_search(claim, all_image_paths, claim_date)
                response = self._call_with_retry(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=messages,
                )

                usage_info = {}
                if response.usage:
                    usage_info = {
                        "input_tokens": response.usage.prompt_tokens,
                        "output_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }

                final_content = response.choices[0].message.content or ""
                verdict = self._extract_verdict(final_content)

                return FactCheckResult(
                    verdict=verdict,
                    reasoning=final_content,
                    citations=[],
                    model=self.model,
                    provider=self.provider_name,
                    usage=usage_info,
                )

            # Build initial messages (with search)
            messages = self._build_initial_messages(claim, all_image_paths, claim_date)

            # Run the conversation with function calling
            citations = []
            search_count = 0
            total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

            while True:
                response = self._call_with_retry(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=messages,
                    tools=[OPENAI_SEARCH_TOOL],
                    tool_choice="auto",
                )

                # Accumulate usage
                if response.usage:
                    total_usage["input_tokens"] += response.usage.prompt_tokens
                    total_usage["output_tokens"] += response.usage.completion_tokens
                    total_usage["total_tokens"] += response.usage.total_tokens

                choice = response.choices[0]
                message = choice.message

                # Add assistant message to conversation
                messages.append(message)

                # Check if we need to handle tool calls
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        if tool_call.function.name == "web_search":
                            search_count += 1
                            if search_count > self.max_search_calls:
                                # Limit reached, provide error message
                                tool_result = "Search limit reached. Please provide your verdict based on the information gathered so far."
                            else:
                                # Execute search
                                args = json.loads(tool_call.function.arguments)
                                query = args.get("query", "")
                                search_response = self.search_service.search(
                                    query=query,
                                    before_date=claim_date,
                                    num_results=10,
                                    max_scrape=2,
                                    scrape_content=self.scrape_content,
                                )
                                tool_result = self.search_service.format_results_for_llm(search_response)
                                # Show content status
                                content_count = sum(1 for r in (search_response.results or []) if r.content)
                                print(f"    Search: {len(search_response.results or [])} results, {content_count} with content")
                                if search_response.error:
                                    print(f"    Search error: {search_response.error}")

                                # Collect citations
                                if search_response.results:
                                    for result in search_response.results:
                                        if result.url and result.url not in citations:
                                            citations.append(result.url)

                            # Add tool result to conversation
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": tool_result,
                            })

                # Check if model is done (no more tool calls)
                if choice.finish_reason == "stop" or not message.tool_calls:
                    break

            # Extract final response
            final_content = message.content or ""
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
        system_prompt = self._get_custom_search_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]

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

        # Format prompt with date if available
        prompt_text = self._get_custom_search_user_prompt(claim, claim_date)
        content.append({"type": "text", "text": prompt_text})

        messages.append({"role": "user", "content": content})

        return messages

    def _build_initial_messages_no_search(
        self,
        claim: str,
        image_paths: list[str],
        claim_date: str | datetime | None,
    ) -> list[dict]:
        """Build initial messages for no-search mode (parametric knowledge only)."""
        system_prompt = self._get_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]

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

        # Format prompt with date if available
        prompt_text = self._get_user_prompt(claim, claim_date)
        content.append({"type": "text", "text": prompt_text})

        messages.append({"role": "user", "content": content})

        return messages
