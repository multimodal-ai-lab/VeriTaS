"""Source media in evidence propositions.

The faithfulness judge - the one call that sees a proposition next to a source -
names the source's media that show what the proposition states. Once the citation
is admissible, their references are put in front of the proposition.
"""

from datetime import datetime

import pytest

from tests.gold_evidence.conftest import make_citation, make_evidence, make_source
from veritas.common.annotation import PROPERTIES
from veritas.gold_evidence import filtering as filtering_module
from veritas.gold_evidence.models import Faithfulness, media_references, prepend_media

PROPOSITION = "A crowd gathered in front of the town hall on 3 May."
T_C = datetime(2024, 5, 1)
T_F = datetime(2024, 5, 21)


# --- Prepending ------------------------------------------------------------------

def test_media_references_go_in_front_of_the_proposition():
    assert prepend_media(PROPOSITION, ["<image:1>", "<video:2>"]) == \
           f"<image:1> <video:2> {PROPOSITION}"


def test_prepending_is_idempotent_and_skips_duplicates():
    once = prepend_media(PROPOSITION, ["<image:1>", "<image:1>"])
    assert once == f"<image:1> {PROPOSITION}"
    assert prepend_media(once, ["<image:1>", "<image:2>"]) == f"<image:2> <image:1> {PROPOSITION}"


def test_nothing_to_prepend_leaves_the_proposition_as_it_is():
    assert prepend_media(PROPOSITION, []) == PROPOSITION
    proposition = "The video <video:5> shows the square."
    assert prepend_media(proposition, ["<video:5>"]) == proposition


def test_media_references_are_found_in_order_and_once():
    text = "Intro <image:1> text <video:22> again <image:1> and <image:1234567890>."
    assert media_references(text) == ["<image:1>", "<video:22>"]
    assert media_references(None) == []


# --- Attaching them while judging ------------------------------------------------------

def judge_answering(media):
    async def fake_assess(proposition, source_str):
        return Faithfulness(assessment=1.0 if media is not None else -1.0, media=media or [])
    return fake_assess


@pytest.mark.asyncio
async def test_an_admissible_citation_puts_its_media_in_front_of_the_proposition(monkeypatch):
    monkeypatch.setattr(filtering_module, "assess_faithfulness",
                        judge_answering(["<image:12>"]))
    item = make_evidence(proposition=PROPOSITION, decided=False,
                         citations=[make_citation(filtered=False)])
    item.citations[0].source = make_source()

    await filtering_module.judge_citation(item.citations[0], evidence=item, t_c=T_C, t_f=T_F)

    assert item.citations[0].admissible is True
    assert item.proposition == f"<image:12> {PROPOSITION}"
    assert item.citations[0].media == ["<image:12>"]


@pytest.mark.asyncio
async def test_an_unfaithful_citation_contributes_no_media(monkeypatch):
    async def unfaithful(proposition, source_str):
        return Faithfulness(assessment=-1.0, media=["<image:12>"])

    monkeypatch.setattr(filtering_module, "assess_faithfulness", unfaithful)
    item = make_evidence(proposition=PROPOSITION, decided=False,
                         citations=[make_citation(filtered=False)])
    item.citations[0].source = make_source()

    await filtering_module.judge_citation(item.citations[0], evidence=item, t_c=T_C, t_f=T_F)

    assert item.citations[0].admissible is False
    assert item.proposition == PROPOSITION


@pytest.mark.asyncio
async def test_judging_again_does_not_repeat_the_media(monkeypatch):
    monkeypatch.setattr(filtering_module, "assess_faithfulness",
                        judge_answering(["<image:12>"]))
    item = make_evidence(proposition=PROPOSITION, decided=False,
                         citations=[make_citation(filtered=False)])
    item.citations[0].source = make_source()

    for _ in range(2):
        await filtering_module.judge_citation(item.citations[0], evidence=item, t_c=T_C, t_f=T_F)
    assert item.proposition == f"<image:12> {PROPOSITION}"


def test_the_sufficiency_prompt_shows_the_media_with_the_proposition():
    from jinja2 import Environment, FileSystemLoader

    item = make_evidence(proposition=f"<image:1> {PROPOSITION}")
    rendered = Environment(loader=FileSystemLoader("")).get_template(
        "veritas/gold_evidence/prompts/assess_from_evidence.md.j2").render(
        evidence=[item], rationales=[], claim="Someone claimed X.", claim_date=None,
        medium=None, subject="Claim", property=PROPERTIES["integrity"])
    assert f"<image:1> {PROPOSITION}" in rendered


def test_a_citation_without_a_judgement_has_no_media():
    assert make_citation(filtered=False).media == []


# --- Parsing the judge's answer -------------------------------------------------------

ANSWER = """The source shows the square.
`Entails` _Certain_
Medium: <image:12>
- Medium: <video:3> | trailing words are ignored
Medium: <image:99>
Medium: <image:12>
```
It says so.
```"""


def test_named_media_are_parsed_and_checked_against_the_source():
    media = filtering_module.parse_cited_media(
        ANSWER, available=["<image:12>", "<video:3>", "<image:4>"])
    assert media == ["<image:12>", "<video:3>"]  # <image:99> is not in the source


def test_the_number_of_media_per_citation_is_capped():
    assert filtering_module.parse_cited_media(
        ANSWER, available=["<image:12>", "<video:3>"], limit=1) == ["<image:12>"]


@pytest.mark.parametrize("answer", ["Medium: none", "`Unknown`\n```x```", ""])
def test_no_media_are_parsed_from_an_answer_naming_none(answer):
    assert filtering_module.parse_cited_media(answer, available=["<image:1>"]) == []


def test_the_media_lines_do_not_disturb_the_rating():
    score, explanation = filtering_module.parse_faithfulness_response(ANSWER)
    assert score == pytest.approx(1.0)
    assert explanation == "It says so."


class StubJudge:
    specifier = "stub:judge"

    def __init__(self, answer: str):
        self.answer = answer
        self.prompts: list[str] = []

    async def generate(self, prompt, **kwargs):
        self.prompts.append(str(prompt))
        return self.answer, None


class TextPrompt:
    """Stands in for `Prompt`, which would resolve the (made-up) media references
    of these tests against the ezMM store."""

    def __init__(self, path, **kwargs):
        self.kwargs = kwargs

    def __str__(self):
        return str(self.kwargs)


@pytest.mark.asyncio
async def test_the_faithfulness_judge_returns_the_media_it_names(monkeypatch):
    judge = StubJudge(ANSWER)
    monkeypatch.setattr(filtering_module, "_resolve_filtering_model", lambda: judge)
    monkeypatch.setattr(filtering_module, "max_media_per_citation", 4)
    monkeypatch.setattr(filtering_module, "Prompt", TextPrompt)

    faithfulness = await filtering_module.assess_faithfulness(
        PROPOSITION, "Photo: <image:12> A video: <video:3>")

    assert faithfulness.assessment == pytest.approx(1.0)
    assert faithfulness.media == ["<image:12>", "<video:3>"]
    assert "'has_media': True" in judge.prompts[0]


def render_faithfulness(source: str) -> str:
    from jinja2 import Environment, FileSystemLoader

    return Environment(loader=FileSystemLoader("")).get_template(
        "veritas/gold_evidence/prompts/assess_faithfulness.md.j2").render(
        proposition=PROPOSITION, source=source,
        has_media=bool(media_references(source)), max_media=4)


def test_the_faithfulness_prompt_asks_for_media_only_when_the_source_has_some():
    with_media = render_faithfulness("Photo: <image:12>")
    without_media = render_faithfulness("Just text.")
    assert "## 4. Media" in with_media and "## 5. Explanation" in with_media
    assert "Medium:" not in without_media and "## 4. Explanation" in without_media


def test_the_faithfulness_prompt_references_no_real_medium_of_its_own():
    """A concrete reference in the instructions would be resolved to a stored
    medium and sent along with every call."""
    rendered = render_faithfulness("Photo: <image:12>")
    assert media_references(rendered.replace("Photo: <image:12>", "")) == []


# --- Persistence ---------------------------------------------------------------------

def test_cited_media_survive_the_citation_roundtrip():
    from veritas.db.veritas_db import _citation_columns, row_to_citation

    citation = make_citation()
    citation.faithfulness = Faithfulness(assessment=1.0, media=["<image:1>"])
    row = dict(zip(*_citation_columns(citation, evidence_id=1, claim_id=1)))
    assert row_to_citation(row).media == ["<image:1>"]


def test_the_cleaned_content_is_stored_once_per_source():
    from veritas.db.veritas_db import _source_columns, row_to_source

    source = make_source(content="Raw <image:1> page.")
    source.cleaned_content = "Main <image:1>."
    source.cleaned_at = datetime(2026, 9, 30)
    row = dict(zip(*_source_columns(source)))

    assert row["cleaned_content"] == "Main <image:1>."
    assert "cleaned_content" not in row["full_source"]
    restored = row_to_source(row)
    assert restored.cleaned_content == "Main <image:1>."
    assert restored.cleaned_at == datetime(2026, 9, 30)
    assert restored.main_content == "Main <image:1>."


def test_the_raw_content_is_used_where_there_is_no_cleaned_one():
    assert make_source(content="Raw page.").main_content == "Raw page."
