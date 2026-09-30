"""Trimming a retrieved evidence source down to its main content.

A raw scrapeMM result carries the whole page: navigation, ads, cookie notices,
"related articles", comments, footers. That noise wastes the prompt budget of the
later LLM checks and, above all, makes it hard to tell the source's own media from
the thumbnails and banners around it.

The approach mirrors `veritas.pipeline.stage_3.extract_article_content`: the page's
lines are numbered, a cheap model names the line ranges that make up the main
content, and exactly those lines are returned. The model never rewrites text, so
the result is a verbatim excerpt of the page, and ezMM media references such as
``<image:123>`` survive unchanged.

Unlike stage 3, the model may name *several* ranges. Evidence sources are far more
varied than fact-checking articles: a social media post often has its media above
the author line and its text further down, with UI chrome in between, and news
pages interrupt the article with ad or "read also" blocks. A single span would
either keep that noise or cut off part of the content. Supporting several ranges
costs only a slightly more permissive parser; the ranges are sorted and merged, so
the output always follows page order and contains no line twice.

Cleaning is best effort. Whenever the answer is unusable or the excerpt looks like
it lost the source's substance (see `keeps_substance`), the result is None and the
caller keeps the raw content: a noisy source is merely expensive, a gutted one
would falsify the faithfulness and later-event checks built on it.
"""

from __future__ import annotations

import logging
import re

from veritas.common import Prompt
from veritas.gold_evidence import (
    cleaning_model,
    max_cleaning_input_length,
    reasoning_effort_cleaning,
)
from veritas.gold_evidence.llm import FATAL_ERRORS, resolve_model

logger = logging.getLogger("VeriTaS")

CLEANING_PROMPT_PATH = "veritas/gold_evidence/prompts/clean_source.md.j2"

#: Pages shorter than this (in characters) are returned unchanged without a model
#: call: there is too little room for noise to be worth a call, and API-based
#: retrievals (social media, archives) already arrive short and clean.
MIN_CLEANING_LENGTH = 1_500

#: Below this many non-media characters, an excerpt holds no substance (at best a
#: title or an author line) and is rejected. Deliberately low, so that a short
#: social media post, possibly accompanied by media only, still passes.
MIN_KEPT_CHARS = 40

#: An excerpt keeping less than this fraction of the page's non-media text is
#: rejected, unless it is substantial in absolute terms (`SUBSTANTIAL_CHARS`). A
#: tiny fraction usually means the model picked the wrong region, e.g. only a
#: headline or a teaser. The cost of the rule: a very short post on a very long page
#: (e.g. 200 characters out of 30,000) is kept raw - which is safe, just noisy.
MIN_KEPT_FRACTION = 0.02

#: Excerpts at least this long (non-media characters) are accepted regardless of the
#: fraction: a full article on a page bloated by comments legitimately keeps little.
SUBSTANTIAL_CHARS = 1_000

#: An ezMM item reference inside text, as in `veritas.db.veritas_db.MEDIA_REF_PATTERN`
#: (not imported, to keep this module free of the database layer).
MEDIA_REF_REGEX = re.compile(r"<(?:image|video|audio):[0-9]{1,9}>")

#: Backticked spans in the model's answer.
_CODE_SPAN_REGEX = re.compile(r"`([^`]*)`")

#: A line range such as `12-57`, `12 - 57`, `12–57`, `12 to 57` or `12..57`.
_RANGE_REGEX = re.compile(r"(\d+)\s*(?:-|–|—|\.\.|to)\s*(\d+)", re.IGNORECASE)

#: A code span holding a single line number, as in stage 3's `12` and `57`.
_NUMBER_REGEX = re.compile(r"\s*(\d+)\s*")

Span = tuple[int, int]  # (start, end), both inclusive, 0-based line numbers


# --- Pure helpers -------------------------------------------------------------

def number_lines(lines: list[str]) -> str:
    """The page with each line prefixed by its (0-based) number, as in stage 3.
    Blank lines are numbered too, so that numbers map back to lines one-to-one."""
    return "\n".join(f"{i}: {line}" for i, line in enumerate(lines))


def parse_spans(answer: str | None, n_lines: int) -> list[Span] | None:
    """The line ranges named in the model's answer, sorted, merged and clamped to
    the page. None if the answer names no usable range (garbage, `none`, empty).

    Accepted forms: backticked ranges (`12-57`, several allowed), and stage 3's
    backticked number pairs (`12` and `57`). Without backticks, only explicit ranges
    count, never loose numbers, which could be anything the model mentions in prose.
    Reversed ranges are swapped; ends beyond the page are clamped; ranges starting
    beyond the page are dropped, as they cannot refer to this page at all."""
    if not answer or n_lines <= 0:
        return None

    code_spans = _CODE_SPAN_REGEX.findall(answer)
    raw_spans: list[Span] = []
    if code_spans:
        numbers: list[int] = []
        for code in code_spans:
            ranges = _RANGE_REGEX.findall(code)
            if ranges:
                raw_spans.extend((int(a), int(b)) for a, b in ranges)
            elif match := _NUMBER_REGEX.fullmatch(code):
                numbers.append(int(match.group(1)))
        # Loose numbers pair up in order of appearance; an odd one out is ambiguous.
        raw_spans.extend(zip(numbers[0::2], numbers[1::2]))
    else:
        raw_spans.extend((int(a), int(b)) for a, b in _RANGE_REGEX.findall(answer))

    last = n_lines - 1
    spans = []
    for start, end in raw_spans:
        start, end = min(start, end), max(start, end)
        if start > last:
            continue
        spans.append((start, min(end, last)))
    return merge_spans(spans) or None


def merge_spans(spans: list[Span]) -> list[Span]:
    """Sorts the spans and merges overlapping or directly adjacent ones, so that
    the excerpt follows page order and contains every line at most once."""
    merged: list[Span] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def cut_spans(lines: list[str], spans: list[Span]) -> str:
    """The selected lines, verbatim and in page order."""
    return "\n".join(line for start, end in spans for line in lines[start:end + 1])


def text_length(text: str) -> int:
    """Number of non-whitespace characters outside media references - the measure
    of how much textual substance a piece of content holds."""
    return len(re.sub(r"\s+", "", MEDIA_REF_REGEX.sub("", text)))


def keeps_substance(lines: list[str], spans: list[Span]) -> bool:
    """Whether the excerpt given by `spans` retains the page's substance.

    Rejected are excerpts that
    - keep fewer than `MIN_KEPT_CHARS` non-media characters;
    - keep less than `MIN_KEPT_FRACTION` of the page's text while being shorter
      than `SUBSTANTIAL_CHARS`;
    - drop every media reference although the selected region - from the first to
      the last selected line - contains some, i.e. the ranges skip exactly the media
      between them. Media inside the main content almost always belong to it, and
      losing all of them would strip the source of the media it contributes. The
      region is not widened beyond the selection: the lines just outside it are
      where logos and banners sit, and they would trigger needless fallbacks."""
    if not spans:
        return False
    excerpt = cut_spans(lines, spans)

    kept = text_length(excerpt)
    if kept < MIN_KEPT_CHARS:
        return False
    total = text_length("\n".join(lines))
    if kept < SUBSTANTIAL_CHARS and total > 0 and kept / total < MIN_KEPT_FRACTION:
        return False

    if not MEDIA_REF_REGEX.search(excerpt):
        region = "\n".join(lines[spans[0][0]:spans[-1][1] + 1])
        if MEDIA_REF_REGEX.search(region):
            return False

    return True


def truncate(content: str, limit: int) -> str:
    """Cuts the content to `limit` characters, at the last line break before it if
    there is one, so that no half line - or half media reference - is shown."""
    if len(content) <= limit:
        return content
    cut = content[:limit]
    newline = cut.rfind("\n")
    return cut[:newline] if newline > 0 else cut


def apply_answer(content: str, answer: str | None) -> str | None:
    """The excerpt of `content` the model's answer selects, or None if the answer is
    unusable or the excerpt fails `keeps_substance`."""
    lines = content.splitlines()
    spans = parse_spans(answer, len(lines))
    if spans is None or not keeps_substance(lines, spans):
        return None
    return cut_spans(lines, spans)


# --- Model call ---------------------------------------------------------------

async def clean_source_content(raw_content: str) -> str | None:
    """The source's main content: its own text and media without the surrounding
    page chrome, as a verbatim excerpt of `raw_content`. None means "use the raw
    content" - the answer was unusable, the excerpt lost substance, or the call
    failed. Pages shorter than `MIN_CLEANING_LENGTH` are returned unchanged.

    Run-level errors (`FATAL_ERRORS`) propagate, as everywhere in this package."""
    if not raw_content:
        return None
    if len(raw_content) < MIN_CLEANING_LENGTH:
        return raw_content

    from veritas.models import gpt_nano

    content = truncate(raw_content, max_cleaning_input_length)
    lines = content.splitlines()
    model = resolve_model(cleaning_model, gpt_nano)
    prompt = Prompt(CLEANING_PROMPT_PATH, page=number_lines(lines), n_lines=len(lines))
    try:
        response = await model.generate(prompt, resolve_media=False,
                                        reasoning_effort=reasoning_effort_cleaning)
    except FATAL_ERRORS:
        raise
    except Exception as e:
        logger.debug(f"Cleaning a source failed: {type(e).__name__}: {e}")
        return None

    answer = str(response) if response else None
    cleaned = apply_answer(content, answer)
    if cleaned is None:
        logger.debug(f"Discarded the cleaning answer {answer!r:.200}; keeping the raw content.")
    return cleaned
