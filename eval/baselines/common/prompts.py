"""The prompt of the fact-checking baseline.

There is exactly one prompt: the model verifies a claim with the web_search and
fetch_url tools and decides on the 7-class label scheme in two steps (DIRECTION,
then CERTAINTY), followed by a JUSTIFICATION and the combined VERDICT.
"""

from .types import LabelScheme, LABEL_SCHEME_7


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


def format_direction_list(scheme: LabelScheme) -> str:
    """Format 7-bin directions as a slash-separated list (e.g., INTACT/UNKNOWN/COMPROMISED)."""
    positive = scheme.labels[0].split(" (", 1)[0].upper()
    negative = scheme.labels[-1].split(" (", 1)[0].upper()
    return f"{positive}/UNKNOWN/{negative}"


SYSTEM_PROMPT_TEMPLATE = """
You are a professional fact-checker. Your task is to verify claims by searching the web for reliable sources and evidence.

# Method
You will analyse a claim by searching the web and rate it on a spectrum from Intact to Compromised. To do so, you use the `web_search` and `fetch_url` tools.

`web_search` allows you to perform web searches with Google, and `fetch_url` lets you fetch the content of a web page. The short previews of the search results returned by `web_search` can be misleading due to their brevity; therefore, you must use `fetch_url` on relevant search results to grasp the full content of the page.

Your final verdict must be based on the evidence you find and present in your final justification.

## Arriving at a verdict
For each claim, follow this decision protocol:
1. Use the web_search and fetch_url tools to gather evidence from credible sources.
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
  DIRECTION: <{direction_list}>
  CERTAINTY: <certain/rather certain/rather uncertain/N/A>
  JUSTIFICATION: <1-2 sentence justification citing ALL the exact retrieved URLs of the Evidence you build your verdict on as inline markdown hyperlinks>
  VERDICT: [{label_list}]
  """.strip()

SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE.format(
    label_descriptions=format_label_descriptions(LABEL_SCHEME_7),
    label_list=format_label_list(LABEL_SCHEME_7),
    direction_list=format_direction_list(LABEL_SCHEME_7),
)


def build_user_prompt(claim: str, claim_date: str | None = None) -> str:
    """Build the user prompt for a claim. `claim_date` is the human-readable claim date, if known."""
    parts = [f"Please fact-check the following claim:\n{claim}"]
    if claim_date:
        parts.append(f"This claim was made on {claim_date}.")
    parts.append("Provide your analysis, then give your final verdict.")
    return "\n\n".join(parts)
