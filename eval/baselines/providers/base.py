"""Abstract base class for fact-checking providers."""

import random
import re
import time
from abc import ABC, abstractmethod
from datetime import datetime

from ..common.types import LABEL_SCHEME_7, MAX_RETRIES, BASE_DELAY, MAX_DELAY
from ..common.prompts import SYSTEM_PROMPT, build_user_prompt
from ..common.tools import ToolSession, DEFAULT_MAX_SEARCHES, DEFAULT_MAX_FETCHES

# Maximum model calls in a tool-calling loop. The last call is made with tool use
# disabled, forcing the model to give its verdict.
MAX_TURNS = 20


def extract_justification(response: str) -> str:
    """Extract the JUSTIFICATION field from a response ("" when absent)."""
    match = re.search(r"JUSTIFICATION\s*:\s*([^\n\r]+)", response or "", re.IGNORECASE)
    if not match:
        return ""

    # Only leading/trailing wrappers: stripping punctuation like the verdict
    # fields do would break URLs and markdown links inside the justification.
    return match.group(1).strip().strip("[]<>`*_ ").strip()


class BaseFactChecker(ABC):
    """Abstract base class for fact-checking providers.

    Every provider runs the same baseline: the model verifies the claim with the
    web_search and fetch_url tools and answers on the 7-class label scheme in two
    steps (DIRECTION + CERTAINTY).
    """

    provider_name: str = "base"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "",
        scrape_methods: list[str] | str | None = "auto",
        max_searches: int = DEFAULT_MAX_SEARCHES,
        max_fetches: int = DEFAULT_MAX_FETCHES,
    ):
        """
        Initialize the fact-checker.

        Args:
            api_key: API key for the provider.
            model: Model identifier to use.
            scrape_methods: Which scrapeMM backends fetch_url uses (subset of
                            integrations/browser/firecrawl/decodo, or "auto").
            max_searches: Maximum number of web_search calls per claim.
            max_fetches: Maximum number of fetch_url calls per claim.
        """
        self.api_key = api_key
        self.model = model
        self.label_scheme = LABEL_SCHEME_7
        self.scrape_methods = scrape_methods
        self.max_searches = max_searches
        self.max_fetches = max_fetches

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

    def _system_prompt(self) -> str:
        """The system prompt of the baseline."""
        return SYSTEM_PROMPT

    def _user_prompt(self, claim: str, claim_date: str | datetime | None = None) -> str:
        """The user prompt for a claim."""
        formatted_date = self._format_date_for_prompt(claim_date) if claim_date else None
        return build_user_prompt(claim, formatted_date)

    def _new_tool_session(self, claim_date: str | datetime | None) -> ToolSession:
        """Create the tool session for a single claim."""
        return ToolSession(
            claim_date=claim_date,
            scrape_methods=self.scrape_methods,
            max_searches=self.max_searches,
            max_fetches=self.max_fetches,
        )

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

        # Fallback: look for verdict words at end of response. Drop the
        # JUSTIFICATION line first — it sits inside this window and routinely
        # contains label words ("compromised", "intact") that would win here.
        scan_text = re.sub(r"^.*JUSTIFICATION\s*:.*$", "", response, flags=re.IGNORECASE | re.MULTILINE)
        last_lines = scan_text.strip().split("\n")[-3:]
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

    def _get_direction_triplet(self) -> tuple[str, str, str]:
        """Return (positive_direction, unknown_label, negative_direction), i.e. ("Intact", "Unknown", "Compromised")."""
        positive = self.label_scheme.labels[0].split(" (", 1)[0].strip()
        negative = self.label_scheme.labels[-1].split(" (", 1)[0].strip()
        return positive, "Unknown", negative

    def _extract_two_step_verdict(self, response: str) -> str | None:
        """Extract the 7-bin verdict from the DIRECTION/CERTAINTY fields."""
        positive, unknown, negative = self._get_direction_triplet()

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
