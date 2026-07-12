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

# Convenience: default scheme and labels (backwards compatible)
LABELS = LABELS_3
DEFAULT_LABEL_SCHEME = LABEL_SCHEME_3

LABEL_SCHEMES = {
    3: LABEL_SCHEME_3,
    7: LABEL_SCHEME_7,
}

# =============================================================================
# Property-specific 3-class schemes (for staged evaluation)
# =============================================================================

PROPERTY_LABEL_SCHEMES_3 = {
    "authenticity": LabelScheme(
        name="authenticity-3-class",
        labels=["Pristine", "Unknown", "Fabricated"],
        verdict_to_numeric={
            "Pristine": 1.0,
            "Unknown": 0.0,
            "Fabricated": -1.0,
        },
        label_descriptions={
            "Pristine": "The medium is an unaltered recording of reality",
            "Unknown": "There is insufficient evidence to determine media authenticity",
            "Fabricated": "The medium is synthetic, manipulated, or forged",
        },
    ),
    "contextualization": LabelScheme(
        name="contextualization-3-class",
        labels=["Correct", "Unknown", "Incorrect"],
        verdict_to_numeric={
            "Correct": 1.0,
            "Unknown": 0.0,
            "Incorrect": -1.0,
        },
        label_descriptions={
            "Correct": "The claim accurately reflects the media's context",
            "Unknown": "There is insufficient evidence to determine contextualization",
            "Incorrect": "The claim misrepresents the media's context",
        },
    ),
    "veracity": LabelScheme(
        name="veracity-3-class",
        labels=["True", "Unknown", "False"],
        verdict_to_numeric={
            "True": 1.0,
            "Unknown": 0.0,
            "False": -1.0,
        },
        label_descriptions={
            "True": "The evidence supports the claim's assertions",
            "Unknown": "There is insufficient evidence to determine veracity",
            "False": "The evidence contradicts one or more assertions",
        },
    ),
    "context_coverage": LabelScheme(
        name="context_coverage-3-class",
        labels=["Sufficient", "Unknown", "Insufficient"],
        verdict_to_numeric={
            "Sufficient": 1.0,
            "Unknown": 0.0,
            "Insufficient": -1.0,
        },
        label_descriptions={
            "Sufficient": "The claim includes enough context to avoid false impressions",
            "Unknown": "There is insufficient evidence to determine context coverage",
            "Insufficient": "The claim omits necessary context and creates false impressions",
        },
    ),
}

PROPERTY_LABEL_SCHEMES_7 = {
    "authenticity": LabelScheme(
        name="authenticity-7-class",
        labels=[
            "Pristine (certain)",
            "Pristine (rather certain)",
            "Pristine (rather uncertain)",
            "Unknown",
            "Fabricated (rather uncertain)",
            "Fabricated (rather certain)",
            "Fabricated (certain)",
        ],
        verdict_to_numeric={
            "Pristine (certain)": 1.0,
            "Pristine (rather certain)": 2 / 3,
            "Pristine (rather uncertain)": 1 / 3,
            "Unknown": 0.0,
            "Fabricated (rather uncertain)": -1 / 3,
            "Fabricated (rather certain)": -2 / 3,
            "Fabricated (certain)": -1.0,
        },
        label_descriptions={
            "Pristine (certain)": "Strong evidence the medium is pristine",
            "Pristine (rather certain)": "Likely pristine with strong but non-definitive evidence",
            "Pristine (rather uncertain)": "Weak evidence toward pristine",
            "Unknown": "Insufficient evidence about authenticity",
            "Fabricated (rather uncertain)": "Weak evidence toward fabricated",
            "Fabricated (rather certain)": "Likely fabricated with strong but non-definitive evidence",
            "Fabricated (certain)": "Strong evidence the medium is fabricated",
        },
    ),
    "contextualization": LabelScheme(
        name="contextualization-7-class",
        labels=[
            "Correct (certain)",
            "Correct (rather certain)",
            "Correct (rather uncertain)",
            "Unknown",
            "Incorrect (rather uncertain)",
            "Incorrect (rather certain)",
            "Incorrect (certain)",
        ],
        verdict_to_numeric={
            "Correct (certain)": 1.0,
            "Correct (rather certain)": 2 / 3,
            "Correct (rather uncertain)": 1 / 3,
            "Unknown": 0.0,
            "Incorrect (rather uncertain)": -1 / 3,
            "Incorrect (rather certain)": -2 / 3,
            "Incorrect (certain)": -1.0,
        },
        label_descriptions={
            "Correct (certain)": "Strong evidence the media context is correct",
            "Correct (rather certain)": "Likely correct context with strong but non-definitive evidence",
            "Correct (rather uncertain)": "Weak evidence toward correct context",
            "Unknown": "Insufficient evidence about contextualization",
            "Incorrect (rather uncertain)": "Weak evidence toward incorrect context",
            "Incorrect (rather certain)": "Likely incorrect context with strong but non-definitive evidence",
            "Incorrect (certain)": "Strong evidence the media context is incorrect",
        },
    ),
    "veracity": LabelScheme(
        name="veracity-7-class",
        labels=[
            "True (certain)",
            "True (rather certain)",
            "True (rather uncertain)",
            "Unknown",
            "False (rather uncertain)",
            "False (rather certain)",
            "False (certain)",
        ],
        verdict_to_numeric={
            "True (certain)": 1.0,
            "True (rather certain)": 2 / 3,
            "True (rather uncertain)": 1 / 3,
            "Unknown": 0.0,
            "False (rather uncertain)": -1 / 3,
            "False (rather certain)": -2 / 3,
            "False (certain)": -1.0,
        },
        label_descriptions={
            "True (certain)": "Strong evidence the claim is true",
            "True (rather certain)": "Likely true with strong but non-definitive evidence",
            "True (rather uncertain)": "Weak evidence toward true",
            "Unknown": "Insufficient evidence about veracity",
            "False (rather uncertain)": "Weak evidence toward false",
            "False (rather certain)": "Likely false with strong but non-definitive evidence",
            "False (certain)": "Strong evidence the claim is false",
        },
    ),
    "context_coverage": LabelScheme(
        name="context_coverage-7-class",
        labels=[
            "Sufficient (certain)",
            "Sufficient (rather certain)",
            "Sufficient (rather uncertain)",
            "Unknown",
            "Insufficient (rather uncertain)",
            "Insufficient (rather certain)",
            "Insufficient (certain)",
        ],
        verdict_to_numeric={
            "Sufficient (certain)": 1.0,
            "Sufficient (rather certain)": 2 / 3,
            "Sufficient (rather uncertain)": 1 / 3,
            "Unknown": 0.0,
            "Insufficient (rather uncertain)": -1 / 3,
            "Insufficient (rather certain)": -2 / 3,
            "Insufficient (certain)": -1.0,
        },
        label_descriptions={
            "Sufficient (certain)": "Strong evidence context coverage is sufficient",
            "Sufficient (rather certain)": "Likely sufficient with strong but non-definitive evidence",
            "Sufficient (rather uncertain)": "Weak evidence toward sufficient context",
            "Unknown": "Insufficient evidence about context coverage",
            "Insufficient (rather uncertain)": "Weak evidence toward insufficient context",
            "Insufficient (rather certain)": "Likely insufficient with strong but non-definitive evidence",
            "Insufficient (certain)": "Strong evidence context coverage is insufficient",
        },
    ),
}


def get_label_scheme(n_classes: int = 3) -> LabelScheme:
    """Get a label scheme by number of classes (3 or 7)."""
    if n_classes not in LABEL_SCHEMES:
        raise ValueError(f"Unsupported label scheme: {n_classes}. Choose from {list(LABEL_SCHEMES.keys())}")
    return LABEL_SCHEMES[n_classes]


def get_property_label_scheme(property_name: str, n_classes: int = 3) -> LabelScheme:
    """Get the property-specific label scheme (3-class or 7-class) used for staged evaluation."""
    schemes = PROPERTY_LABEL_SCHEMES_3 if n_classes == 3 else PROPERTY_LABEL_SCHEMES_7 if n_classes == 7 else None
    if schemes is None:
        raise ValueError(f"Unsupported property label scheme class count: {n_classes}. Choose from [3, 7]")
    if property_name not in schemes:
        raise ValueError(
            f"Unsupported property label scheme: {property_name}. "
            f"Choose from {list(schemes.keys())}"
        )
    return schemes[property_name]


@dataclass
class FactCheckResult:
    """Result from a fact-checking operation."""

    verdict: str  # Verdict label (from either 3-class or 7-class scheme)
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
