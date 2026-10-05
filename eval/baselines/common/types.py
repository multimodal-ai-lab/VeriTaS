"""Shared types and constants for fact-checking baselines."""

from dataclasses import dataclass, field
from typing import Literal

# Verdict type - the three possible outcomes (3-class)
Verdict = Literal["Intact", "Compromised", "Unknown"]

# Extended verdict type including uncertainty levels (7-class)
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
    """Configuration for the label set used in fact-checking."""

    name: str
    labels: list[str]
    verdict_to_numeric: dict[str, float]
    label_descriptions: dict[str, str]

    @property
    def verdict_regex_pattern(self) -> str:
        """Build a regex alternation pattern matching all labels (case-insensitive)."""
        # Sort longest-first so regex matches greedily
        escaped = sorted(
            (label.upper() for label in self.labels),
            key=len, reverse=True,
        )
        # Escape parentheses for regex
        parts = [l.replace("(", r"\(").replace(")", r"\)") for l in escaped]
        return "|".join(parts)

    def normalize_verdict(self, raw: str) -> str:
        """Normalize a raw extracted verdict string to canonical label form."""
        raw_lower = raw.strip().lower()
        for label in self.labels:
            if label.lower() == raw_lower:
                return label
        # Fallback: return Unknown
        return "Unknown"


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
    label_descriptions={
        "Intact": "The claim is factually accurate and any associated media is authentic and properly contextualized",
        "Unknown": "There is insufficient evidence to determine the claim's accuracy",
        "Compromised": "The claim is factually inaccurate, misleading, or contains manipulated/miscontextualized media",
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
    label_descriptions={
        "Intact (certain)": "The claim is factually accurate with strong, unequivocal evidence",
        "Intact (rather certain)": "The claim appears factually accurate with strong but not fully definitive evidence",
        "Intact (rather uncertain)": "The claim weakly appears factually accurate based on limited evidence",
        "Unknown": "Evidence is completely absent or irreconcilably contradictory — use ONLY as a last resort; if any directional lean exists (even weak), use the corresponding 'rather uncertain' bin instead",
        "Compromised (rather uncertain)": "The claim weakly appears inaccurate or misleading based on limited evidence",
        "Compromised (rather certain)": "The claim appears inaccurate or misleading with strong but not fully definitive evidence",
        "Compromised (certain)": "The claim is factually inaccurate, misleading, or contains manipulated/miscontextualized media with strong, unequivocal evidence",
    },
)

# The baseline predicts on the 7-class scheme. The 3-class scheme is kept for evaluation,
# where 7-class predictions are coarsened to 3 classes (see metrics.py).

# Convenience: default scheme and labels (backwards compatible)
LABELS = LABELS_3
DEFAULT_LABEL_SCHEME = LABEL_SCHEME_3

LABEL_SCHEMES = {
    3: LABEL_SCHEME_3,
    7: LABEL_SCHEME_7,
}


def get_label_scheme(n_classes: int = 3) -> LabelScheme:
    """Get a label scheme by number of classes (3 or 7)."""
    if n_classes not in LABEL_SCHEMES:
        raise ValueError(f"Unsupported label scheme: {n_classes}. Choose from {list(LABEL_SCHEMES.keys())}")
    return LABEL_SCHEMES[n_classes]


@dataclass
class FactCheckResult:
    """Result from a fact-checking operation."""

    verdict: str  # Verdict label from the 7-class scheme
    reasoning: str
    citations: list[str] = field(default_factory=list)
    model: str = ""
    provider: str = ""
    usage: dict = field(default_factory=dict)
    justification: str = ""  # Parsed JUSTIFICATION field; "" when the model emitted none

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
        }
