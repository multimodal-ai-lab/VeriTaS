"""Common utilities for the baselines package."""

from .types import (
    Verdict7,
    FactCheckResult,
    FactCheckError,
    NoAnswerError,
    LabelScheme,
    LABEL_SCHEME_3,
    LABEL_SCHEME_7,
    LABELS_3,
    LABELS_7,
)
from .prompts import PROMPT_TEMPLATE, build_prompt
from .parsing import extract_answer, extract_justification, extract_urls, extract_verdict
from .media import (
    Image,
    MediaLimits,
    load_image,
    get_mime_type,
    parse_media_references,
    resolve_media_path,
    resolve_media_files,
    extract_video_frames,
)
from .metrics import (
    classify_integrity,
    coarsen_label,
    compute_metrics,
    compute_coarsened_metrics,
    compute_regression_metrics,
    print_metrics,
)
from .tools import (
    ToolSession,
    SearchQuotaExceededError,
    TOOL_DEFINITIONS,
    MAX_FETCH_CHARS,
    build_search_query,
    parse_claim_day,
)
from .claims import get_claim_quarter

__all__ = [
    # Types
    "Verdict7",
    "FactCheckResult",
    "FactCheckError",
    "NoAnswerError",
    "LabelScheme",
    "LABEL_SCHEME_3",
    "LABEL_SCHEME_7",
    "LABELS_3",
    "LABELS_7",
    # Prompts
    "PROMPT_TEMPLATE",
    "build_prompt",
    # Parsing
    "extract_answer",
    "extract_justification",
    "extract_urls",
    "extract_verdict",
    # Media
    "Image",
    "MediaLimits",
    "load_image",
    "get_mime_type",
    "parse_media_references",
    "resolve_media_path",
    "resolve_media_files",
    "extract_video_frames",
    # Metrics
    "classify_integrity",
    "coarsen_label",
    "compute_metrics",
    "compute_coarsened_metrics",
    "compute_regression_metrics",
    "print_metrics",
    # Tools
    "ToolSession",
    "SearchQuotaExceededError",
    "TOOL_DEFINITIONS",
    "MAX_FETCH_CHARS",
    "build_search_query",
    "parse_claim_day",
    # Claims
    "get_claim_quarter",
]
