"""Trimming a retrieved source to its main content: the excerpt is verbatim, follows
page order, and is discarded whenever it would lose the source's substance."""

from __future__ import annotations

import pytest

from veritas.common import Prompt
from veritas.gold_evidence import cleaning as cleaning_module
from veritas.gold_evidence.cleaning import (
    MIN_CLEANING_LENGTH,
    MIN_KEPT_CHARS,
    SUBSTANTIAL_CHARS,
    apply_answer,
    clean_source_content,
    cut_spans,
    keeps_substance,
    merge_spans,
    number_lines,
    parse_spans,
    text_length,
    truncate,
)
from veritas.models.base import QuotaExceededError, RateLimitError

ARTICLE_LINE = "The mayor signed the decree on 3 May, as the city register shows in detail."


def make_page(n_noise_top: int = 10, n_article: int = 30, n_noise_bottom: int = 10) -> list[str]:
    """A page of navigation, an article and a comment section, in that order."""
    return ([f"Menu item {i}" for i in range(n_noise_top)]
            + [f"{ARTICLE_LINE} ({i})" for i in range(n_article)]
            + [f"Comment {i}: great read!" for i in range(n_noise_bottom)])


# --- Line numbering ---------------------------------------------------------

def test_lines_are_numbered_from_zero():
    assert number_lines(["a", "", "c"]) == "0: a\n1: \n2: c"


def test_numbering_an_empty_page():
    assert number_lines([]) == ""


# --- Parsing the answer -----------------------------------------------------

@pytest.mark.parametrize("answer,expected", [
    ("`12-57`", [(12, 57)]),
    ("`12 - 57`", [(12, 57)]),
    ("`12–57`", [(12, 57)]),           # en dash
    ("`12 to 57`", [(12, 57)]),
    ("`12..57`", [(12, 57)]),
    ("`12` and `57`", [(12, 57)]),     # stage 3's format
    ("`57-12`", [(12, 57)]),           # reversed
    ("`57` and `12`", [(12, 57)]),     # reversed pair
    ("The main content spans lines 12-57.", [(12, 57)]),  # unbackticked range
    ("`3-3`", [(3, 3)]),
])
def test_a_single_range_is_parsed(answer, expected):
    assert parse_spans(answer, 100) == expected


def test_several_ranges_are_parsed_in_page_order():
    assert parse_spans("`46-57` `12-40`", 100) == [(12, 40), (46, 57)]


def test_several_ranges_within_one_code_span():
    assert parse_spans("`12-40, 46-57`", 100) == [(12, 40), (46, 57)]


def test_overlapping_and_adjacent_ranges_are_merged():
    assert parse_spans("`12-40` `30-50` `51-60` `70-80`", 100) == [(12, 60), (70, 80)]


def test_duplicate_ranges_are_merged():
    assert parse_spans("`12-40` `12-40`", 100) == [(12, 40)]


def test_ranges_are_clamped_to_the_page():
    assert parse_spans("`90-250`", 100) == [(90, 99)]


def test_ranges_beyond_the_page_are_dropped():
    assert parse_spans("`10-20` `150-200`", 100) == [(10, 20)]
    assert parse_spans("`150-200`", 100) is None


def test_backticked_ranges_take_precedence_over_prose():
    """Numbers in the model's prose are not line numbers once it used backticks."""
    assert parse_spans("Lines 1-2 are the menu; the article is `12-57`.", 100) == [(12, 57)]


def test_an_odd_loose_number_is_ignored():
    assert parse_spans("`12` and `57` and `80`", 100) == [(12, 57)]


@pytest.mark.parametrize("answer", [
    None,
    "",
    "`none`",
    "none",
    "I cannot determine the main content.",
    "The article starts at line 12.",   # a loose number without backticks
    "`12`",                               # a lone number is no range
    "`abc-def`",
])
def test_garbage_is_rejected(answer):
    assert parse_spans(answer, 100) is None


def test_an_empty_page_has_no_spans():
    assert parse_spans("`0-5`", 0) is None


def test_merge_spans_sorts_and_merges():
    assert merge_spans([(5, 6), (0, 2), (3, 4), (10, 12)]) == [(0, 6), (10, 12)]
    assert merge_spans([]) == []


# --- Cutting ----------------------------------------------------------------

def test_spans_are_cut_verbatim_and_inclusive():
    lines = ["a", "b <image:1>", "c", "d", "e"]
    assert cut_spans(lines, [(1, 2)]) == "b <image:1>\nc"
    assert cut_spans(lines, [(0, 0), (3, 4)]) == "a\nd\ne"


def test_text_length_ignores_media_and_whitespace():
    assert text_length("ab <image:12>\n c <video:3>") == 3


def test_truncation_cuts_at_a_line_break():
    content = "first line\nsecond line <image:123>"
    assert truncate(content, 1_000) == content
    assert truncate(content, 30) == "first line"


def test_truncation_without_line_breaks_cuts_hard():
    assert truncate("x" * 50, 10) == "x" * 10


# --- The substance guard ----------------------------------------------------

def test_the_article_span_keeps_substance():
    lines = make_page()
    assert keeps_substance(lines, [(10, 39)])


def test_no_spans_keep_no_substance():
    assert not keeps_substance(make_page(), [])


def test_a_nearly_empty_excerpt_is_rejected():
    lines = make_page() + ["<image:1>", "Hi"]
    assert not keeps_substance(lines, [(50, 51)])


def test_media_with_a_short_post_text_pass():
    """A social media post may be little more than a few words and a photo."""
    lines = (["Menu"] * 5 + ["<image:1>", "Look at this huge crowd gathering in front of the old town hall today!"]
             + ["Menu"] * 5)
    assert keeps_substance(lines, [(5, 6)])


def test_a_tiny_fraction_of_a_long_page_is_rejected():
    """E.g. only the headline of a long article."""
    lines = make_page(n_article=200)
    assert text_length(lines[10]) >= MIN_KEPT_CHARS
    assert not keeps_substance(lines, [(10, 10)])


def test_a_substantial_excerpt_passes_regardless_of_the_fraction():
    """A full article on a page bloated by thousands of comments."""
    lines = make_page(n_article=20, n_noise_bottom=5_000)
    kept = text_length(cut_spans(lines, [(10, 29)]))
    assert kept >= SUBSTANTIAL_CHARS
    assert kept / text_length("\n".join(lines)) < 0.02
    assert keeps_substance(lines, [(10, 29)])


def test_ranges_skipping_every_medium_of_the_main_content_are_rejected():
    lines = make_page()
    lines[20] = "<image:7>"
    lines[25] = "<video:8>"
    assert not keeps_substance(lines, [(10, 19), (21, 24), (26, 39)])


def test_dropping_some_media_of_the_main_content_is_fine():
    """E.g. an ad image inside the article is left out, the article's photo kept."""
    lines = make_page()
    lines[15] = "<image:7>"
    lines[25] = "<image:8>"
    assert keeps_substance(lines, [(10, 24), (26, 39)])


def test_media_outside_the_selected_region_do_not_matter():
    """Logos in the navigation and thumbnails of related articles are meant to go."""
    lines = make_page()
    lines[2] = "<image:1>"
    lines[45] = "<image:2>"
    assert keeps_substance(lines, [(10, 39)])


def test_apply_answer_cuts_the_selected_lines():
    lines = make_page()
    lines[10] = "<image:5> Photo: the signed decree"
    content = "\n".join(lines)
    assert apply_answer(content, "`10-39`") == "\n".join(lines[10:40])


@pytest.mark.parametrize("answer", ["`none`", "no idea", "`115-116`"])
def test_apply_answer_rejects_unusable_answers(answer):
    """Garbage, `none`, and a span holding only two comments of a long page."""
    content = "\n".join(make_page(n_article=100))
    assert apply_answer(content, answer) is None


# --- The model call ---------------------------------------------------------

class FakeCleaningModel:
    def __init__(self, answer=None, error: Exception | None = None):
        self.answer = answer
        self.error = error
        self.prompts: list[str] = []
        self.kwargs: list[dict] = []

    async def generate(self, prompt, **kwargs):
        self.prompts.append(str(prompt))
        self.kwargs.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.answer


@pytest.fixture
def cleaning_model(monkeypatch):
    def install(answer=None, error: Exception | None = None) -> FakeCleaningModel:
        model = FakeCleaningModel(answer, error)
        monkeypatch.setattr(cleaning_module, "resolve_model",
                            lambda specifier, default: model)
        return model
    return install


@pytest.fixture
def plain_prompt(monkeypatch):
    """Replaces the prompt by its plain text. For pages with media references,
    which a real `Prompt` would try to resolve as ezMM items."""
    class PlainPrompt:
        def __init__(self, file_path, **kwargs):
            self.text = kwargs["page"]

        def __str__(self):
            return self.text

    monkeypatch.setattr(cleaning_module, "Prompt", PlainPrompt)


def long_page() -> str:
    content = "\n".join(make_page())
    assert len(content) >= MIN_CLEANING_LENGTH
    return content


@pytest.mark.asyncio
async def test_the_main_content_is_returned(cleaning_model):
    model = cleaning_model("`10-39`")
    content = long_page()

    result = await clean_source_content(content)

    assert result == "\n".join(content.splitlines()[10:40])
    assert "Menu item" not in result and "Comment" not in result
    assert "10: " + ARTICLE_LINE in model.prompts[0]
    assert model.kwargs[0]["resolve_media"] is False
    assert "reasoning_effort" in model.kwargs[0]


@pytest.mark.asyncio
async def test_media_references_survive_verbatim(cleaning_model, plain_prompt):
    lines = make_page()
    lines[10] = "<image:5>"
    lines[30] = "<video:6> Video: the signing ceremony"
    cleaning_model("`10-39`")

    result = await clean_source_content("\n".join(lines))

    assert "<image:5>" in result and "<video:6> Video: the signing ceremony" in result


@pytest.mark.asyncio
async def test_skipping_all_media_falls_back(cleaning_model, plain_prompt):
    lines = make_page()
    lines[20] = "<image:5>"
    cleaning_model("`10-19` `21-39`")
    assert await clean_source_content("\n".join(lines)) is None


@pytest.mark.asyncio
async def test_short_pages_are_not_cleaned(cleaning_model):
    model = cleaning_model("`0-0`")
    content = "A short post. " * 10
    assert len(content) < MIN_CLEANING_LENGTH

    assert await clean_source_content(content) == content
    assert model.prompts == []


@pytest.mark.asyncio
async def test_empty_content_yields_none(cleaning_model):
    model = cleaning_model("`0-0`")
    assert await clean_source_content("") is None
    assert model.prompts == []


@pytest.mark.asyncio
async def test_the_input_is_truncated(cleaning_model, monkeypatch):
    monkeypatch.setattr(cleaning_module, "max_cleaning_input_length", 2_000)
    model = cleaning_model("`10-39`")
    content = "\n".join(make_page(n_noise_bottom=1_000))
    assert len(content) > 2_000

    await clean_source_content(content)

    assert "Comment 999" not in model.prompts[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [None, "", "`none`", "I don't know."])
async def test_unusable_answers_yield_none(cleaning_model, answer):
    cleaning_model(answer)
    assert await clean_source_content(long_page()) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("error_cls", [QuotaExceededError, RateLimitError])
async def test_fatal_errors_propagate(cleaning_model, error_cls):
    cleaning_model(error=error_cls("quota"))
    with pytest.raises(error_cls):
        await clean_source_content(long_page())


@pytest.mark.asyncio
async def test_ordinary_failures_yield_none(cleaning_model):
    cleaning_model(error=RuntimeError("timeout"))
    assert await clean_source_content(long_page()) is None


# --- The prompt -------------------------------------------------------------

def test_the_prompt_renders():
    lines = make_page(n_article=3)
    text = str(Prompt(cleaning_module.CLEANING_PROMPT_PATH,
                      page=number_lines(lines), n_lines=len(lines)))

    assert "0: Menu item 0" in text
    assert f"({len(lines)} lines)" in text
    assert "`12-57`" in text          # the answer format the parser expects
    assert parse_spans("`12-57`", 100) == [(12, 57)]
    assert "{{" not in text and "{%" not in text
