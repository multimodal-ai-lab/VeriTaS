"""The prompt of the fact-checking baseline.

There is exactly one prompt, sent as the user message (there is no system prompt): the model
fact-checks the claim with the web_search and fetch_url tools and answers with a JSON object
holding the claim's integrity, the certainty and a justification (see parsing.py).
"""

from datetime import date

from ezmm import Item

from .media import ItemSequence

PROMPT_TEMPLATE = """**Your task is to fact-check the claim stated below.** To this end, you are provided with tools you may use for evidence retrieval.

Note that fact-checking organizations are deliberately excluded from search results and from scraping as you should perform your independent research.

Today is {{ today }}.

# Task Details
Analyze the overall informational integrity of the claim. A claim's integrity is
- **intact** if the claim contextualizes all media correctly, has only true assertions, and does not give any false impressions,
- **compromised** otherwise.

# The Claim
{{ claim }}

# Rules
At the end of your investigations, provide a verdict. That is, decide whether the integrity tends to be "compromised" or "intact". Even a slight tendency counts. If your evidence is entirely inconclusive, pick "unknown".

Unless "unknown", state your level of certainty. Choose:
 - "certain" if there is no room for doubt.
 - "rather certain" if there remains a small chance of being wrong, e.g., the evidence is not fully definitive.
 - "rather uncertain" if there is a high risk of the verdict being wrong, e.g., because the evidence only weakly supports one side.

# Output
Output your verdict formatted in JSON as follows:
```json
{
  "integrity": "compromised" | "intact" | "unknown",
  "certainty": "certain" | "rather certain" | "rather uncertain" | null,
  "justification": "<A short explanation (1–4 sentences) justifying your integrity choice and certainty. Include relevant evidence URLs as Markdown hyperlinks.>"
}
```"""


def build_prompt(claim: str, claim_day: date | None = None, media: list[Item] = (),
                 today: date | None = None) -> ItemSequence:
    """The prompt for a claim, as an ezMM sequence. The claim is rendered as the VeriTaS
    pipeline renders it (Claim.__str__): its text in quotes and the day it was published on,
    if known. Its images and videos follow its text, inside the quotes: the dataset gives no
    position for them in it. `today` is the day the prompt says it is (default: the actual day)."""
    today = (today or date.today()).strftime("%B %d, %Y")
    before, after = PROMPT_TEMPLATE.replace("{{ today }}", today).split("{{ claim }}")
    published = f"\nPublished on: {claim_day.strftime('%B %d, %Y')}" if claim_day else ""
    if not media:
        return [f'{before}"{claim}"{published}{after}']
    return [f'{before}"{claim}', *media, f'"{published}{after}']
