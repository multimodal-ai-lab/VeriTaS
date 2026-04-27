"""Common utilities for the baselines package."""

from .types import (
    Verdict,
    FactCheckResult,
    LABELS,
    LabelScheme,
    LABEL_SCHEME_3,
    LABEL_SCHEME_7,
    LABELS_3,
    LABELS_7,
    DEFAULT_LABEL_SCHEME,
    get_label_scheme,
    get_property_label_scheme,
)
from .prompts import SYSTEM_PROMPT, SYSTEM_PROMPT_NO_SEARCH, build_prompts
from .media import (
    encode_image_base64,
    get_mime_type,
    parse_media_references,
    resolve_media_path,
    resolve_media_files,
    extract_video_frames,
)
from .metrics import (
    classify_integrity,
    compute_metrics,
    compute_coarsened_metrics,
    compute_regression_metrics,
    print_metrics,
    VERDICT_TO_NUMERIC,
)
from .search import (
    SearchService,
    SearchResult,
    SearchResponse,
    OPENAI_SEARCH_TOOL,
    GEMINI_SEARCH_TOOL_DECLARATION,
    create_search_service,
)
from .claims import get_claim_quarter

__all__ = [
    # Types
    "Verdict",
    "FactCheckResult",
    "LABELS",
    "LabelScheme",
    "LABEL_SCHEME_3",
    "LABEL_SCHEME_7",
    "LABELS_3",
    "LABELS_7",
    "DEFAULT_LABEL_SCHEME",
    "get_label_scheme",
    "get_property_label_scheme",
    # Prompts
    "SYSTEM_PROMPT",
    "SYSTEM_PROMPT_NO_SEARCH",
    "build_prompts",
    # Media
    "encode_image_base64",
    "get_mime_type",
    "parse_media_references",
    "resolve_media_path",
    "resolve_media_files",
    "extract_video_frames",
    # Metrics
    "classify_integrity",
    "compute_metrics",
    "compute_coarsened_metrics",
    "compute_regression_metrics",
    "print_metrics",
    "VERDICT_TO_NUMERIC",
    # Search
    "SearchService",
    "SearchResult",
    "SearchResponse",
    "OPENAI_SEARCH_TOOL",
    "GEMINI_SEARCH_TOOL_DECLARATION",
    "create_search_service",
    # Claims
    "get_claim_quarter",
]
