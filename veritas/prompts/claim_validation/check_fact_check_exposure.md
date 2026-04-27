# Instructions
You are given a medium (image or video) that is associated with the below claim. **Your task is to determine whether this medium exposes that it has been subject to a fact-check**. The goal is to filter out such media from the benchmark to avoid any shortcuts that could be exploited to identify false claims.

A medium exposes a fact-check if it contains cues hinting at a fact-check—cues that are the result of the editorial fact-checking process. The following cues indicate a fact-check:
- Overlaid text containing fact-checking terms, such as “Fact-Check”, “Fake,” and "AI-generated".
- A logo from the fact-checking organization (like AFP, DW, PolitiFact, etc.).
- Graphical annotations such as red X marks and green check marks
- Blurring for anonymization, typical in journalism.
- If the medium itself is a screenshot of a social media post, it is also an indicator for fact-checking, since this is a common practice of fact-checkers to capture the fact-checking subject.

Typical **original** media include memes, raw footage, vidoes (with subtitles), and any other media typical for social media content. In particular, manipulations which do not imply that the medium was subject to a fact-check can be considered original here.

Your task is not to assess truthfulness, context, or intent—only whether the medium appears to be an asset coming from the claim directly or an altered version processed by a fact-checker.

# Required Output Format
1. **Reasoning Paragraph (one paragraph only)**: Explain step by step why the medium is or is not exposing a fact-check.
2. **Final Decision**: Answer `exposing` if the medium seems to expose a fact-check. Answer `not exposing` otherwise. Return the decision enclosed in backticks.

# The Medium
{medium}

# Your Response
