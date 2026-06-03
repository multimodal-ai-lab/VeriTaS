"""Perplexity fact-checker provider."""

import os
from datetime import datetime
from pathlib import Path
from typing import Literal

from perplexity import Perplexity

# Import API key from Veritas config (if available)
try:
    from veritas import api_secrets
    VERITAS_PERPLEXITY_KEY = api_secrets.get("perplexity") if api_secrets else None
except ImportError:
    VERITAS_PERPLEXITY_KEY = None

from ..common.types import FactCheckResult, LabelScheme
from ..common.media import encode_image_base64, extract_video_frames
from .base import BaseFactChecker


class PerplexityFactChecker(BaseFactChecker):
    """Fact-checker using Perplexity's sonar models with web search and vision."""

    provider_name = "perplexity"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "sonar-pro",
        use_search: bool = True,
        label_scheme: LabelScheme | None = None,
        seven_bin_prediction_mode: Literal["direct", "two_step"] = "direct",
    ):
        """
        Initialize the Perplexity fact-checker.

        Args:
            api_key: Perplexity API key. If None, uses PERPLEXITY_API_KEY env var.
            model: Model to use. Options:
                - "sonar": Fast, cost-effective
                - "sonar-pro": More capable, better for complex queries
                - "sonar-reasoning-pro": Best reasoning, includes chain-of-thought
            use_search: Must be True for Perplexity. Perplexity's API always
                       performs web search - it cannot operate in parametric-only mode.
            label_scheme: Label scheme to use (3-class or 7-class). Defaults to 3-class.
            seven_bin_prediction_mode: For 7-class schemes, "direct" asks for a
                                      combined label; "two_step" asks for
                                      direction + certainty.
        """
        if not use_search:
            raise ValueError(
                "Perplexity cannot operate without search. Perplexity's API is built around "
                "web search and always performs search on queries. Use a different provider "
                "(openai, gemini, selfhosted) for parametric-only fact-checking."
            )
        super().__init__(
            api_key=api_key,
            model=model,
            use_search=use_search,
            label_scheme=label_scheme,
            seven_bin_prediction_mode=seven_bin_prediction_mode,
        )
        self.client = self._create_client(api_key)

    def _create_client(self, api_key: str | None) -> Perplexity:
        """Create Perplexity client."""
        if api_key:
            return Perplexity(api_key=api_key)

        # Try Veritas config first, then environment variable
        key_to_use = VERITAS_PERPLEXITY_KEY or os.environ.get("PERPLEXITY_API_KEY")

        if not key_to_use:
            raise ValueError(
                "Perplexity API key required. Either:\n"
                "  1. Add it to config/globals.yaml (perplexity: <key>)\n"
                "  2. Set PERPLEXITY_API_KEY environment variable\n"
                "  3. Pass api_key parameter"
            )
        return Perplexity(api_key=key_to_use)

    def check_claim(
        self,
        claim: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        claim_date: str | datetime | None = None,
    ) -> FactCheckResult:
        """
        Fact-check a claim using Perplexity with optional images.

        Args:
            claim: The claim text to verify.
            image_paths: Optional list of image file paths to include.
            video_paths: Optional list of video file paths (frames will be extracted).
            claim_date: Date of the claim (ISO format string or datetime).
                        If provided, search results will be filtered to only
                        include content published before this date.

        Returns:
            FactCheckResult with verdict, reasoning, citations, etc.
        """
        # Build message content
        content = []

        # Collect all images (including extracted video frames)
        all_image_paths = list(image_paths) if image_paths else []
        temp_frame_paths = []

        # Extract frames from videos (Perplexity doesn't support native video)
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
                        "type": "image_url",
                        "image_url": {"url": img_data}
                    })

            # Format the prompt using label-scheme-aware helpers
            prompt_text = self._get_user_prompt(claim, claim_date=None)  # Perplexity uses API-level date filtering
            system_prompt = self._get_system_prompt()

            # Add the text prompt
            content.append({
                "type": "text",
                "text": prompt_text
            })

            # Build API request parameters
            request_params: dict = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
            }

            # Add date filter to prevent data leakage
            if claim_date:
                search_before = self._format_date_filter(claim_date)
                if search_before:
                    request_params["search_before_date_filter"] = search_before

            # Call API with retry logic
            response = self._call_with_retry(
                self.client.chat.completions.create,
                **request_params
            )

            assert response is not None, "Perplexity API call failed after retries."

            response_content = response.choices[0].message.content
            verdict = self._extract_verdict(response_content)

            # Extract citations if available
            citations = []
            if hasattr(response, "citations") and response.citations:
                citations = response.citations

            return FactCheckResult(
                verdict=verdict,
                reasoning=response_content,
                citations=citations,
                model=self.model,
                provider=self.provider_name,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            )
        finally:
            # Clean up temporary frame files
            for temp_path in temp_frame_paths:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass

    def _format_date_filter(self, claim_date: str | datetime) -> str | None:
        """
        Format claim date for Perplexity's search_before_date parameter.

        Perplexity expects format: "%m/%d/%Y" (e.g., "3/1/2025")
        """
        try:
            if isinstance(claim_date, str):
                dt = datetime.fromisoformat(claim_date.replace("Z", "+00:00"))
            else:
                dt = claim_date

            return dt.strftime("%-m/%-d/%Y")  # e.g., "2/5/2024"
        except (ValueError, AttributeError):
            return None
