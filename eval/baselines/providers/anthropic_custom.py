"""Anthropic fact-checker provider with custom search tool.

This provider uses Anthropic's tool use API with a custom search tool that supports:
1. Date filtering: Only results from before the claim date
"""

import os
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
from ..common.media import encode_image_base64, extract_video_frames
from ..common.search import SearchService, ANTHROPIC_SEARCH_TOOL, ScrapeMode
from .base import BaseFactChecker
from .anthropic import _parse_data_uri


class AnthropicCustomSearchFactChecker(BaseFactChecker):
    """Fact-checker using Anthropic Claude with custom search tool supporting date filtering."""

    provider_name = "anthropic_custom"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        serpapi_key: str | None = None,
        max_search_calls: int = 5,
        scrape_content: bool = True,
        scrape_mode: ScrapeMode = "lite",
        use_search: bool = True,
        label_scheme: LabelScheme | None = None,
        seven_bin_prediction_mode: Literal["direct", "two_step"] = "direct",
    ):
        """
        Initialize the Anthropic fact-checker with custom search.

        Args:
            api_key: Anthropic API key. If None, uses config/env var.
            model: Model to use (must support tool use).
            serpapi_key: SerpAPI key for search. If None, uses config/env var.
            max_search_calls: Maximum number of search calls per fact-check.
            scrape_content: Whether to scrape full page content.
            scrape_mode: Scraping method - "lite" (fast), "scrapemm" (full), or "none".
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
        if use_search:
            self.search_service = SearchService(
                serpapi_key=serpapi_key,
                scrape_mode=scrape_mode,
                max_content_length=5000,
            )
        else:
            self.search_service = None
        self.max_search_calls = max_search_calls
        self.scrape_content = scrape_content

    def _create_client(self, api_key: str | None) -> anthropic_sdk.Anthropic:
        """Create Anthropic client."""
        if api_key:
            return anthropic_sdk.Anthropic(api_key=api_key)

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
        Fact-check a claim using Anthropic with custom search tool.

        Args:
            claim: The claim text to verify.
            image_paths: Optional list of image file paths.
            video_paths: Optional list of video file paths (frames extracted).
            claim_date: Date of the claim for temporal filtering.

        Returns:
            FactCheckResult with verdict, reasoning, citations.
        """
        all_image_paths = list(image_paths) if image_paths else []
        temp_frame_paths = []

        if video_paths:
            for video_path in video_paths:
                frames = extract_video_frames(video_path, max_frames=5)
                all_image_paths.extend(frames)
                temp_frame_paths.extend(frames)

        try:
            # Build initial user content
            user_content = self._build_user_content(claim, all_image_paths, claim_date)

            # Handle no-search mode (parametric knowledge only)
            if not self.use_search:
                response = self._call_with_retry(
                    self.client.messages.create,
                    model=self.model,
                    max_tokens=8096,
                    system=self._get_system_prompt(),
                    messages=[{"role": "user", "content": user_content}],
                )
                response_text = self._extract_text(response.content)
                verdict = self._extract_verdict(response_text)
                usage_info = self._extract_usage(response)
                return FactCheckResult(
                    verdict=verdict,
                    reasoning=response_text,
                    citations=[],
                    model=self.model,
                    provider=self.provider_name,
                    usage=usage_info,
                )

            # Tool-calling loop
            messages = [{"role": "user", "content": user_content}]
            system_prompt = self._get_custom_search_system_prompt()
            citations = []
            search_count = 0
            total_usage = {"input_tokens": 0, "output_tokens": 0}

            while True:
                response = self._call_with_retry(
                    self.client.messages.create,
                    model=self.model,
                    max_tokens=8096,
                    system=system_prompt,
                    messages=messages,
                    tools=[ANTHROPIC_SEARCH_TOOL],
                )

                # Accumulate usage
                if hasattr(response, "usage") and response.usage:
                    total_usage["input_tokens"] += getattr(response.usage, "input_tokens", 0)
                    total_usage["output_tokens"] += getattr(response.usage, "output_tokens", 0)

                # Append assistant turn to history
                messages.append({"role": "assistant", "content": response.content})

                # Check for tool use blocks
                tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

                if not tool_use_blocks or response.stop_reason == "end_turn":
                    break

                # Process each tool call and collect results
                tool_results = []
                for block in tool_use_blocks:
                    if block.name == "web_search":
                        search_count += 1
                        if search_count > self.max_search_calls:
                            result_text = "Search limit reached. Please provide your verdict based on the information gathered so far."
                        else:
                            query = block.input.get("query", "")
                            search_response = self.search_service.search(
                                query=query,
                                before_date=claim_date,
                                num_results=10,
                                max_scrape=2,
                                scrape_content=self.scrape_content,
                            )
                            result_text = self.search_service.format_results_for_llm(search_response)
                            content_count = sum(1 for r in (search_response.results or []) if r.content)
                            print(f"    Search: {len(search_response.results or [])} results, {content_count} with content")
                            if search_response.error:
                                print(f"    Search error: {search_response.error}")
                            if search_response.results:
                                for result in search_response.results:
                                    if result.url and result.url not in citations:
                                        citations.append(result.url)

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                        })

                # Append tool results as next user turn
                messages.append({"role": "user", "content": tool_results})

            # Extract final text response
            final_text = self._extract_text(response.content)
            verdict = self._extract_verdict(final_text)

            for url in self._extract_urls_from_text(final_text):
                if url not in citations:
                    citations.append(url)

            return FactCheckResult(
                verdict=verdict,
                reasoning=final_text,
                citations=citations,
                model=self.model,
                provider=self.provider_name,
                usage=total_usage,
            )

        finally:
            for temp_path in temp_frame_paths:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass

    def _build_user_content(
        self,
        claim: str,
        image_paths: list[str],
        claim_date: str | datetime | None,
    ) -> list[dict]:
        """Build user message content with optional images."""
        content = []
        for img_path in image_paths:
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
        prompt_text = self._get_custom_search_user_prompt(claim, claim_date)
        content.append({"type": "text", "text": prompt_text})
        return content

    def _extract_text(self, content_blocks) -> str:
        """Extract text from a list of Anthropic content blocks."""
        return "\n".join(
            b.text for b in content_blocks if getattr(b, "type", None) == "text"
        )

    def _extract_usage(self, response) -> dict:
        """Extract token usage from response."""
        if hasattr(response, "usage") and response.usage:
            return {
                "input_tokens": getattr(response.usage, "input_tokens", 0),
                "output_tokens": getattr(response.usage, "output_tokens", 0),
            }
        return {}
