"""Shared types and constants for fact-checking baselines."""

from dataclasses import dataclass, field
from typing import Literal

# The verdict the baseline predicts (7-class)
Verdict7 = Literal[
    "Intact (certain)", "Intact (rather certain)", "Intact (rather uncertain)",
    "Unknown",
    "Compromised (rather uncertain)", "Compromised (rather certain)", "Compromised (certain)",
]

# Retry configuration for API rate limits
MAX_RETRIES = 5
BASE_DELAY = 2.0  # seconds
MAX_DELAY = 60.0  # seconds


@dataclass
class LabelScheme:
    """A label set with the numeric value of each label."""

    name: str
    labels: list[str]
    verdict_to_numeric: dict[str, float]


# =============================================================================
# 3-class scheme
# =============================================================================

LABELS_3 = ["Intact", "Unknown", "Compromised"]

LABEL_SCHEME_3 = LabelScheme(
    name="3-class",
    labels=LABELS_3,
    verdict_to_numeric={
        "Intact": 1.0,
        "Unknown": 0.0,
        "Compromised": -1.0,
    },
)

# =============================================================================
# 7-class scheme (with uncertainty)
# =============================================================================

LABELS_7 = [
    "Intact (certain)",
    "Intact (rather certain)",
    "Intact (rather uncertain)",
    "Unknown",
    "Compromised (rather uncertain)",
    "Compromised (rather certain)",
    "Compromised (certain)",
]

LABEL_SCHEME_7 = LabelScheme(
    name="7-class",
    labels=LABELS_7,
    verdict_to_numeric={
        "Intact (certain)": 1.0,
        "Intact (rather certain)": 2 / 3,
        "Intact (rather uncertain)": 1 / 3,
        "Unknown": 0.0,
        "Compromised (rather uncertain)": -1 / 3,
        "Compromised (rather certain)": -2 / 3,
        "Compromised (certain)": -1.0,
    },
)

# The baseline predicts on the 7-class scheme. The 3-class scheme is only used for evaluation,
# where 7-class labels are coarsened to 3 classes (see metrics.py).


@dataclass
class FactCheckResult:
    """Result from a fact-checking operation."""

    verdict: str | None  # Verdict label from the 7-class scheme; None if the model gave no valid answer
    reasoning: str
    citations: list[str] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    usage: dict = field(default_factory=dict)
    justification: str = ""  # The justification of the model's JSON answer; "" when there is none
    encountered_urls: list[str] = field(default_factory=list)  # URLs of the web_search results
    retrieved_urls: list[str] = field(default_factory=list)  # URLs successfully fetched with fetch_url
    linked_urls: list[str] = field(default_factory=list)  # URLs in the content of the fetched pages
    quoted_urls: list[str] = field(default_factory=list)  # URLs cited in the justification
    # The web_search results dropped and fetch_url calls refused because they are or show a
    # blocked URL, with the rule that blocked them (see ToolSession.blocked); the model sees none
    blocked: list[dict] = field(default_factory=list)
    # Everything that happened, in order: the prompt, the model's reasoning, text and tool calls,
    # and the tool results (dicts with "type" and, but for the prompt, the model call's "turn")
    trace: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "verdict": self.verdict,
            "reasoning": self.reasoning,
            "citations": self.citations,
            "model": self.model,
            "provider": self.provider,
            "usage": self.usage,
            "justification": self.justification,
            "encountered_urls": self.encountered_urls,
            "retrieved_urls": self.retrieved_urls,
            "linked_urls": self.linked_urls,
            "quoted_urls": self.quoted_urls,
            "blocked": self.blocked,
            "trace": self.trace,
        }


class FactCheckError(Exception):
    """A fact-check that failed. Holds what it got to before: its trace, URLs and usage."""

    status = "error"  # The claim's status in the results

    def __init__(self, message: str, result: FactCheckResult):
        super().__init__(message)
        self.result = result


class NoAnswerError(FactCheckError):
    """A fact-check whose final response holds no valid answer: no JSON object with a valid
    integrity and certainty, e.g. because the response was cut off or blocked."""

    status = "no_answer"
