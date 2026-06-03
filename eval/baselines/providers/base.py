"""Abstract base class for fact-checking providers."""

import random
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal

from ..common.types import LabelScheme, DEFAULT_LABEL_SCHEME, MAX_RETRIES, BASE_DELAY, MAX_DELAY
from ..common.prompts import build_prompts


class BaseFactChecker(ABC):
    """Abstract base class for fact-checking providers."""

    provider_name: str = "base"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "",
        use_search: bool = True,
        label_scheme: LabelScheme | None = None,
        seven_bin_prediction_mode: Literal["direct", "two_step"] = "direct",
    ):
        """
        Initialize the fact-checker.

        Args:
            api_key: API key for the provider.
            model: Model identifier to use.
            use_search: If True (default), use web search. If False, use only
                       parametric knowledge (no search tools).
            label_scheme: Label scheme to use (3-class or 7-class). Defaults to 3-class.
            seven_bin_prediction_mode: For 7-class schemes, "direct" asks for a
                                      single combined label, while "two_step"
                                      asks for direction + certainty.
        """
        self.api_key = api_key
        self.model = model
        self.use_search = use_search
        self.label_scheme = label_scheme or DEFAULT_LABEL_SCHEME
        if seven_bin_prediction_mode not in {"direct", "two_step"}:
            raise ValueError("seven_bin_prediction_mode must be 'direct' or 'two_step'")
        self.seven_bin_prediction_mode = seven_bin_prediction_mode
        self._prompts = build_prompts(self.label_scheme)

    @abstractmethod
    def check_claim(
        self,
        claim: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        claim_date: str | datetime | None = None,
    ):
        """
        Fact-check a claim.

        Args:
            claim: The claim text to verify.
            image_paths: Optional list of image file paths to include.
            video_paths: Optional list of video file paths to include.
                         Note: Only Gemini supports native video. Other providers
                         will extract frames from videos and treat as images.
            claim_date: Date of the claim (ISO format string or datetime).

        Returns:
            FactCheckResult with verdict, reasoning, citations, etc.
        """
        pass

    def _get_system_prompt(self) -> str:
        """Get the appropriate system prompt based on search mode."""
        prompt = ""
        if self.use_search:
            prompt = self._prompts["system_prompt"]
        else:
            prompt = self._prompts["system_prompt_no_search"]
        return prompt + self._get_two_step_prompt_suffix()

    def _get_user_prompt(self, claim: str, claim_date: str | datetime | None = None) -> str:
        """Get the formatted user prompt based on search mode and claim date."""
        if self.use_search:
            if claim_date:
                formatted_date = self._format_date_for_prompt(claim_date)
                return self._prompts["user_prompt_with_date"].format(
                    claim=claim, claim_date=formatted_date
                )
            return self._prompts["user_prompt"].format(claim=claim)
        else:
            if claim_date:
                formatted_date = self._format_date_for_prompt(claim_date)
                return self._prompts["user_prompt_with_date_no_search"].format(
                    claim=claim, claim_date=formatted_date
                )
            return self._prompts["user_prompt_no_search"].format(claim=claim)

    def _get_custom_search_system_prompt(self) -> str:
        """Get the custom search system prompt."""
        if self._is_two_step_7bin_mode():
            return self._prompts["system_prompt_custom_search_two_step"]
        return self._prompts["system_prompt_custom_search"]

    def _get_custom_search_user_prompt(self, claim: str, claim_date: str | datetime | None = None) -> str:
        """Get the formatted custom search user prompt."""
        if self._is_two_step_7bin_mode():
            if claim_date:
                formatted_date = self._format_date_for_prompt(claim_date)
                return self._prompts["user_prompt_custom_search_with_date_two_step"].format(
                    claim=claim, claim_date=formatted_date
                )
            return self._prompts["user_prompt_custom_search_two_step"].format(claim=claim)
        if claim_date:
            formatted_date = self._format_date_for_prompt(claim_date)
            return self._prompts["user_prompt_custom_search_with_date"].format(
                claim=claim, claim_date=formatted_date
            )
        return self._prompts["user_prompt_custom_search"].format(claim=claim)

    def _call_with_retry(self, api_call_fn, *args, **kwargs):
        """
        Call an API function with exponential backoff retry for rate limits.

        Args:
            api_call_fn: The API function to call.
            *args, **kwargs: Arguments to pass to the function.

        Returns:
            The result from the API call.

        Raises:
            The last exception if all retries are exhausted.
        """
        last_exception = None

        for attempt in range(MAX_RETRIES):
            try:
                return api_call_fn(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                # Retry on rate limits and transient server errors
                is_retryable = (
                    "429" in str(e) or "rate limit" in error_str or "resource exhausted" in error_str
                    or "503" in str(e) or "unavailable" in error_str
                    or "500" in str(e) or "internal error" in error_str
                )
                if is_retryable:
                    last_exception = e
                    # Exponential backoff with jitter
                    delay = min(BASE_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_DELAY)
                    print(f"    Transient error ({e}), retrying in {delay:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                    time.sleep(delay)
                else:
                    # Not a retryable error, re-raise immediately
                    raise

        # All retries exhausted
        if last_exception:
            raise last_exception

    def _format_date_for_prompt(self, claim_date: str | datetime) -> str:
        """
        Format claim date for inclusion in the prompt.

        Args:
            claim_date: ISO format datetime string or datetime object.

        Returns:
            Human-readable date string (e.g., "February 5, 2024").
        """
        try:
            if isinstance(claim_date, str):
                # Parse ISO format datetime string (e.g., "2024-02-05T12:22:02")
                dt = datetime.fromisoformat(claim_date.replace("Z", "+00:00"))
            else:
                dt = claim_date

            # Format as human-readable date
            return dt.strftime("%B %d, %Y")  # e.g., "February 5, 2024"
        except (ValueError, AttributeError):
            return str(claim_date)

    def _extract_verdict(self, response: str) -> str:
        """
        Extract verdict from response text using the active label scheme.

        Args:
            response: The response text from the model.

        Returns:
            A verdict string from the active label scheme.
        """
        two_step_verdict = self._extract_two_step_verdict(response)
        if two_step_verdict:
            return two_step_verdict

        pattern = self.label_scheme.verdict_regex_pattern

        # Look for explicit VERDICT: pattern
        match = re.search(
            rf"VERDICT:\s*({pattern})",
            response,
            re.IGNORECASE
        )
        if match:
            return self.label_scheme.normalize_verdict(match.group(1))

        # Fallback: look for verdict words at end of response
        last_lines = response.strip().split("\n")[-3:]
        last_text = " ".join(last_lines).upper()

        # Try to match any label in the last lines (longest match first)
        best_match = None
        best_len = 0
        for label in self.label_scheme.labels:
            if label.upper() in last_text and len(label) > best_len:
                best_match = label
                best_len = len(label)

        if best_match:
            return best_match

        # If no clear verdict found, default to Unknown
        return "Unknown"

    def _is_two_step_7bin_mode(self) -> bool:
        """Whether two-step extraction/prompting should be active."""
        return self.seven_bin_prediction_mode == "two_step" and len(self.label_scheme.labels) == 7

    def _get_direction_triplet(self) -> tuple[str, str, str] | None:
        """
        Return (positive_direction, unknown_label, negative_direction) for 7-bin schemes.

        For integrity this is typically ("Intact", "Unknown", "Compromised").
        """
        if len(self.label_scheme.labels) != 7:
            return None

        unknown = None
        for label in self.label_scheme.labels:
            if label.strip().lower() == "unknown":
                unknown = label
                break
        if unknown is None:
            return None

        positive = self.label_scheme.labels[0].split(" (", 1)[0].strip()
        negative = self.label_scheme.labels[-1].split(" (", 1)[0].strip()
        return positive, unknown, negative

    def _get_two_step_prompt_suffix(self) -> str:
        """Additional prompt instructions for 7-bin two-step mode."""
        if not self._is_two_step_7bin_mode():
            return ""

        triplet = self._get_direction_triplet()
        if not triplet:
            return ""
        positive, unknown, negative = triplet

        return (
            "\n\nFor this run, produce the final 7-bin decision in two steps:\n"
            f"1. DIRECTION: exactly one of [{positive}/{unknown}/{negative}]\n"
            f"   - Use {unknown} only as a last resort when evidence is genuinely insufficient\n"
            "     or strongly contradictory after reasonable analysis.\n"
            f"   - If evidence weakly leans {positive} or {negative}, choose that direction\n"
            "     and express uncertainty in CERTAINTY (do not choose UNKNOWN).\n"
            "2. CERTAINTY: exactly one of [certain/rather certain/rather uncertain]\n"
            f"   - If DIRECTION is {unknown}, use CERTAINTY: [N/A]\n\n"
            "Always end your response with these exact final lines:\n"
            "DIRECTION: <chosen direction>\n"
            "CERTAINTY: <chosen certainty or N/A>"
        )

    def _extract_two_step_verdict(self, response: str) -> str | None:
        """Extract 7-bin verdict from DIRECTION/CERTAINTY fields when enabled."""
        if not self._is_two_step_7bin_mode():
            return None

        triplet = self._get_direction_triplet()
        if not triplet:
            return None
        positive, unknown, negative = triplet

        direction_match = re.search(r"DIRECTION\s*:\s*([^\n\r]+)", response, re.IGNORECASE)
        if not direction_match:
            return None

        direction_raw = direction_match.group(1).strip().strip("[](){}<>`_.,;: ").lower()
        direction_map = {
            positive.lower(): positive,
            unknown.lower(): unknown,
            negative.lower(): negative,
        }
        direction = direction_map.get(direction_raw)
        if not direction:
            return None

        if direction.lower() == unknown.lower():
            return unknown

        certainty_match = re.search(r"(?:CERTAINTY|CONFIDENCE)\s*:\s*([^\n\r]+)", response, re.IGNORECASE)
        if not certainty_match:
            return None

        certainty_raw = certainty_match.group(1).strip().strip("[](){}<>`_.,;: ").lower()
        certainty_map = {
            "certain": "certain",
            "rather certain": "rather certain",
            "rather uncertain": "rather uncertain",
        }
        certainty = certainty_map.get(certainty_raw)
        if not certainty:
            return None

        combined = f"{direction} ({certainty})"
        if combined in self.label_scheme.labels:
            return combined
        return None

    def _extract_urls_from_text(self, text: str) -> list[str]:
        """
        Extract URLs from text.

        Args:
            text: Text to search for URLs.

        Returns:
            List of unique URLs found.
        """
        url_pattern = r'https?://[^\s\)\"\'>\]]+'
        return list(dict.fromkeys(re.findall(url_pattern, text)))
