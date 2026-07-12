"""Shared prompts for fact-checking baselines.

All prompt templates use {label_descriptions} and {label_list} placeholders
that get filled in based on the active label scheme (3-class or 7-class).
"""

from .types import LabelScheme, DEFAULT_LABEL_SCHEME


def format_label_descriptions(scheme: LabelScheme) -> str:
    """Format label descriptions as a bullet list for the system prompt."""
    lines = []
    for label in scheme.labels:
        desc = scheme.label_descriptions[label]
        lines.append(f"- {label.upper()}: {desc}")
    return "\n".join(lines)


def format_label_list(scheme: LabelScheme) -> str:
    """Format labels as a slash-separated list for the verdict instruction."""
    return "/".join(label.upper() for label in scheme.labels)


def format_unknown_warning(scheme: LabelScheme) -> str:
    """Return an UNKNOWN-discouragement paragraph for 7-bin prompts, empty string for 3-bin."""
    if len(scheme.labels) != 7:
        return ""
    return (
        "\nIMPORTANT: Reserve UNKNOWN strictly for cases where evidence is completely absent "
        "or irreconcilably contradictory after thorough search. "
        "Any directional lean — even weak — must map to the corresponding 'rather uncertain' bin, NOT to UNKNOWN. "
        "Predicting UNKNOWN when evidence weakly favors one direction is an error."
    )


def format_direction_list(scheme: LabelScheme) -> str:
    """Format 7-bin directions as a slash-separated list (e.g., INTACT/UNKNOWN/COMPROMISED)."""
    if len(scheme.labels) != 7:
        return ""
    positive = scheme.labels[0].split(" (", 1)[0].upper()
    unknown = "UNKNOWN"
    negative = scheme.labels[-1].split(" (", 1)[0].upper()
    return f"{positive}/{unknown}/{negative}"


# =============================================================================
# System prompts (with search)
# =============================================================================

SYSTEM_PROMPT_TEMPLATE = """You are a professional fact-checker. Your task is to verify claims by searching the web for reliable sources and evidence.

For each claim, you must:
1. Search for relevant information from reliable sources
2. Evaluate the factual accuracy of the claim
3. If images are provided, analyze them for authenticity and proper contextualization
4. Provide a final verdict

Your verdict MUST be exactly one of these options:
{label_descriptions}{unknown_warning}

Always end your response with your verdict on a new line in this exact format:
VERDICT: [{label_list}]"""

# =============================================================================
# User prompts (with search)
# =============================================================================

USER_PROMPT_TEMPLATE = """Please fact-check the following claim:

{{claim}}

Search for evidence and provide your analysis, then give your final verdict ({label_list})."""


# Date-aware prompt template (when claim_date is provided)
USER_PROMPT_TEMPLATE_WITH_DATE = """Please fact-check the following claim:

{{claim}}

CRITICAL TEMPORAL CONSTRAINT: This claim was made on {{claim_date}}. You MUST ONLY use information that was publicly available BEFORE this date. DO NOT use:
- Fact-check articles published after {{claim_date}}
- News coverage published after {{claim_date}}
- Any information that was not available to the public before {{claim_date}}

This is essential to prevent data leakage in benchmarking. Treat the date {{claim_date}} as a hard cutoff - imagine you are fact-checking in real-time on that date with only the information available then.

Search for evidence and provide your analysis, then give your final verdict ({label_list})."""

# =============================================================================
# No-search prompts (parametric knowledge only)
# =============================================================================

SYSTEM_PROMPT_NO_SEARCH_TEMPLATE = """You are a professional fact-checker. Your task is to verify claims.

For each claim, you must:
1. Evaluate the factual accuracy of the claim
2. If images are provided, analyze them for authenticity and proper contextualization
3. Provide a final verdict

Your verdict MUST be exactly one of these options:
{label_descriptions}{unknown_warning}

Always end your response with your verdict on a new line in this exact format:
VERDICT: [{label_list}]"""

USER_PROMPT_TEMPLATE_NO_SEARCH = """Please fact-check the following claim:

{{claim}}

Provide your analysis, then give your final verdict ({label_list})."""

USER_PROMPT_TEMPLATE_WITH_DATE_NO_SEARCH = """Please fact-check the following claim:

{{claim}}

This claim was made on {{claim_date}}.

Provide your analysis, then give your final verdict ({label_list})."""

# =============================================================================
# Custom search prompts (Llama, OpenAI custom, Gemini custom)
# =============================================================================

SYSTEM_PROMPT_CUSTOM_SEARCH_TEMPLATE = """You are a professional fact-checker. Your task is to verify claims by searching the web for reliable sources and evidence.

For each claim, you must:
1. Use the web_search tool to find relevant information from reliable sources
2. Evaluate the factual accuracy of the claim based on the search results
3. If images are provided, analyze them for authenticity and proper contextualization
4. Provide a final verdict

MANDATORY: You MUST call web_search at least once before providing any verdict. Never output a verdict without searching first — parametric knowledge alone is not sufficient. If a claim date is provided, use it to filter search results by date; this does NOT mean you should skip searching.

Your verdict MUST be exactly one of these options:
{label_descriptions}{unknown_warning}

Always end your response with your verdict on a new line in this exact format:
VERDICT: [{label_list}]"""

USER_PROMPT_CUSTOM_SEARCH_TEMPLATE = """Please fact-check the following claim:

{{claim}}

Use the web_search tool to find evidence and provide your analysis, then give your final verdict ({label_list})."""

USER_PROMPT_CUSTOM_SEARCH_WITH_DATE_TEMPLATE = """Please fact-check the following claim:

{{claim}}

This claim was made on {{claim_date}}. Use the web_search tool to find evidence and provide your analysis, then give your final verdict ({label_list})."""

# =============================================================================
# Custom search + dedicated 7-bin two-step prompts
# =============================================================================

SYSTEM_PROMPT_CUSTOM_SEARCH_TWO_STEP_TEMPLATE = """You are a professional fact-checker. Your task is to verify claims by searching the web for reliable sources and evidence.

# Method

## Web Search
MANDATORY: You MUST call web_search *at least* once before providing any verdict. Never output a verdict without searching first — parametric knowledge alone is not sufficient. If a claim date is provided, use it to filter search results by date; this does NOT mean you should skip searching.

## Arriving at a verdict
For each claim, follow this decision protocol:
1. Use the web_search tool to gather evidence from credible sources.
2. Evaluate the evidence and choose a DIRECTION from: {direction_list}
3. Use UNKNOWN only as a last resort: choose UNKNOWN only when evidence is genuinely insufficient or strongly contradictory after reasonable search.
4. If there is any directional lean (even weak), do NOT use UNKNOWN. Choose the leaning direction and encode uncertainty via CERTAINTY.
5. If direction is UNKNOWN, do not assign certainty (use N/A).
6. If direction is not UNKNOWN, choose CERTAINTY based on evidence strength:
   - certain: evidence is unequivocal and leaves little room for doubt
   - rather certain: evidence is strong but not fully definitive
   - rather uncertain: evidence weakly supports one side; misclassification risk is high
7. Map direction + certainty to exactly one final 7-bin verdict from:
{label_descriptions}

# Output 
The final verdict MUST be exactly one of: {label_list}
End your response with these exact lines:
  DIRECTION: `<{direction_list}>`
  CERTAINTY: `_<certain/rather certain/rather uncertain/N/A>_`
  VERDICT: [{label_list}]"""

USER_PROMPT_CUSTOM_SEARCH_TWO_STEP_TEMPLATE = """Please fact-check the following claim:

{{claim}}

Use the web_search tool to find evidence and provide your analysis.
Then decide in two steps:
1) direction ({direction_list})
   - Choose UNKNOWN only for true evidence dead-ends or unresolved contradictions.
   - If evidence weakly leans one side, choose that direction (not UNKNOWN).
2) certainty (certain/rather certain/rather uncertain, or N/A if direction is UNKNOWN)
Finally output the mapped 7-bin verdict."""

USER_PROMPT_CUSTOM_SEARCH_TWO_STEP_WITH_DATE_TEMPLATE = """Please fact-check the following claim:

{{claim}}

This claim was made on {{claim_date}}. Use the web_search tool to find evidence and provide your analysis.
Then decide in two steps:
1) direction ({direction_list})
   - Choose UNKNOWN only for true evidence dead-ends or unresolved contradictions.
   - If evidence weakly leans one side, choose that direction (not UNKNOWN).
2) certainty (certain/rather certain/rather uncertain, or N/A if direction is UNKNOWN)
Finally output the mapped 7-bin verdict."""


def build_prompts(scheme: LabelScheme | None = None) -> dict[str, str]:
    """
    Build all prompt strings for a given label scheme.

    Returns a dict with keys:
        - system_prompt
        - user_prompt
        - user_prompt_with_date
        - system_prompt_no_search
        - user_prompt_no_search
        - user_prompt_with_date_no_search
        - system_prompt_custom_search
        - user_prompt_custom_search
        - user_prompt_custom_search_with_date

    The returned user prompts still contain {claim} and {claim_date}
    placeholders to be filled at call time.
    """
    if scheme is None:
        scheme = DEFAULT_LABEL_SCHEME

    label_desc = format_label_descriptions(scheme)
    label_list = format_label_list(scheme)
    direction_list = format_direction_list(scheme)
    unknown_warning = format_unknown_warning(scheme)
    fmt = {
        "label_descriptions": label_desc,
        "label_list": label_list,
        "direction_list": direction_list,
        "unknown_warning": unknown_warning,
    }

    return {
        "system_prompt": SYSTEM_PROMPT_TEMPLATE.format(**fmt),
        "user_prompt": USER_PROMPT_TEMPLATE.format(**fmt),
        "user_prompt_with_date": USER_PROMPT_TEMPLATE_WITH_DATE.format(**fmt),
        "system_prompt_no_search": SYSTEM_PROMPT_NO_SEARCH_TEMPLATE.format(**fmt),
        "user_prompt_no_search": USER_PROMPT_TEMPLATE_NO_SEARCH.format(**fmt),
        "user_prompt_with_date_no_search": USER_PROMPT_TEMPLATE_WITH_DATE_NO_SEARCH.format(**fmt),
        "system_prompt_custom_search": SYSTEM_PROMPT_CUSTOM_SEARCH_TEMPLATE.format(**fmt),
        "user_prompt_custom_search": USER_PROMPT_CUSTOM_SEARCH_TEMPLATE.format(**fmt),
        "user_prompt_custom_search_with_date": USER_PROMPT_CUSTOM_SEARCH_WITH_DATE_TEMPLATE.format(**fmt),
        "system_prompt_custom_search_two_step": SYSTEM_PROMPT_CUSTOM_SEARCH_TWO_STEP_TEMPLATE.format(**fmt),
        "user_prompt_custom_search_two_step": USER_PROMPT_CUSTOM_SEARCH_TWO_STEP_TEMPLATE.format(**fmt),
        "user_prompt_custom_search_with_date_two_step": USER_PROMPT_CUSTOM_SEARCH_TWO_STEP_WITH_DATE_TEMPLATE.format(**fmt),
    }


# =============================================================================
# Backwards-compatible module-level constants (3-class defaults)
# =============================================================================

_default_prompts = build_prompts(DEFAULT_LABEL_SCHEME)

SYSTEM_PROMPT = _default_prompts["system_prompt"]
SYSTEM_PROMPT_NO_SEARCH = _default_prompts["system_prompt_no_search"]
