"""Anthropic fact-checker provider.

The model verifies the claim with the web_search and fetch_url tools via Anthropic's
tool use API.
"""

import os
from datetime import datetime
from pathlib import Path

import anthropic as anthropic_sdk

# Import API key from Veritas config (if available)
try:
    from veritas import api_secrets
    VERITAS_ANTHROPIC_KEY = api_secrets.get("anthropic") if api_secrets else None
except ImportError:
    VERITAS_ANTHROPIC_KEY = None

from ..common.types import FactCheckResult
from ..common.media import encode_image_base64, extract_video_frames, parse_data_uri
from ..common.tools import anthropic_tools, DEFAULT_MAX_SEARCHES, DEFAULT_MAX_FETCHES
from .base import BaseFactChecker, MAX_TURNS


class AnthropicFactChecker(BaseFactChecker):
    """Fact-checker using Anthropic Claude with the web_search and fetch_url tools."""

    provider_name = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        scrape_methods: list[str] | str | None = "auto",
        max_searches: int = DEFAULT_MAX_SEARCHES,
        max_fetches: int = DEFAULT_MAX_FETCHES,
    ):
        """
        Initialize the Anthropic fact-checker.

        Args:
            api_key: Anthropic API key. If None, uses config/env var.
            model: Model to use (must support tool use).
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

    def _create_client(self, api_key: str | None) -> anthropic_sdk.Anthropic:
        """Create Anthropic client."""
        if api_key:
            return anthropic_sdk.Anthropic(api_key=api_key)

        key_to_use = VERITAS_ANTHROPIC_KEY or os.environ.get("ANTHROPIC_API_KEY")

        if not key_to_use:
            raise ValueError(
                "Anthropic API key required. Either:\n"
                "  1. Add it to config.yaml (api_secrets.anthropic)\n"
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
        Fact-check a claim using Anthropic with the web_search and fetch_url tools.

        Args:
            claim: The claim text to verify.
            image_paths: Optional list of image file paths.
            video_paths: Optional list of video file paths (frames extracted).
            claim_date: Date of the claim for temporal filtering.

        Returns:
            FactCheckResult with verdict, reasoning, citations.
        """
        session = self._new_tool_session(claim_date)

        all_image_paths = list(image_paths) if image_paths else []
        temp_frame_paths = []

        if video_paths:
            for video_path in video_paths:
                frames = extract_video_frames(video_path, max_frames=5)
                all_image_paths.extend(frames)
                temp_frame_paths.extend(frames)

        try:
            messages = [{"role": "user", "content": self._build_user_content(claim, all_image_paths, claim_date)}]
            system_prompt = self._system_prompt()
            tools = anthropic_tools()
            total_usage = {"input_tokens": 0, "output_tokens": 0}

            for turn in range(MAX_TURNS):
                # The last turn disables tool use to force a verdict
                tool_choice = {"type": "none"} if turn == MAX_TURNS - 1 else {"type": "auto"}
                response = self._call_with_retry(
                    self.client.messages.create,
                    model=self.model,
                    max_tokens=8096,
                    system=system_prompt,
                    messages=messages,
                    tools=tools,
                    tool_choice=tool_choice,
                )

                # Accumulate usage
                if hasattr(response, "usage") and response.usage:
                    total_usage["input_tokens"] += getattr(response.usage, "input_tokens", 0)
                    total_usage["output_tokens"] += getattr(response.usage, "output_tokens", 0)

                # Append assistant turn to history
                messages.append({"role": "assistant", "content": response.content})

                # The model is done once it stops calling tools
                tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
                if not tool_use_blocks:
                    break

                # Append tool results as next user turn
                messages.append({"role": "user", "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": session.call(block.name, block.input),
                    }
                    for block in tool_use_blocks
                ]})

            # Extract final text response
            final_text = self._extract_text(response.content)
            verdict = self._extract_verdict(final_text)

            citations = list(session.citations)
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
                media_type, b64_data = parse_data_uri(img_data)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64_data,
                    },
                })
        content.append({"type": "text", "text": self._user_prompt(claim, claim_date)})
        return content

    def _extract_text(self, content_blocks) -> str:
        """Extract text from a list of Anthropic content blocks."""
        return "\n".join(
            b.text for b in content_blocks if getattr(b, "type", None) == "text"
        )
