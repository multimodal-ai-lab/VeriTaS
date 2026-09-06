"""Prompt templates render, and the validators cannot see what would make them
circular."""

import re
from datetime import datetime

import pytest

from tests.gold_evidence.conftest import make_evidence
from veritas.common import Prompt
from veritas.common.annotation import PROPERTIES
from veritas.gold_evidence.models import SourceKind

PROMPT_DIR = "veritas/gold_evidence/prompts"

GOLD_VERDICT_WORDS = ("compromised", "intact", "the verdict is", "gold verdict",
                      "fact-checker concluded", "rating:")


def render_extract_evidence() -> str:
    return str(Prompt(
        f"{PROMPT_DIR}/extract_evidence.md.j2",
        article="The post at https://x.com/a/status/1 said X.",
        claim="Someone claimed X.",
        claim_date="May 01, 2024",
        fact_check_date="May 21, 2024",
        publisher_name="Example FactCheck",
        source_kinds=[kind.value for kind in SourceKind],
    ))


def render_faithfulness() -> str:
    return str(Prompt(
        f"{PROMPT_DIR}/assess_faithfulness.md.j2",
        proposition="The mayor signed the decree on 3 May.",
        source="City register entry: decree signed 3 May by the mayor.",
    ))


def render_temporal(**overrides) -> str:
    kwargs = dict(
        claim="Someone claimed X.",
        claim_date="May 01, 2024",
        source="An article about X.",
        source_name="Example News",
        source_url="https://example.org/a",
        available_since="May 10, 2024",
        publisher_hint=None,
    )
    kwargs.update(overrides)
    return str(Prompt(f"{PROMPT_DIR}/validate_temporally.md.j2", **kwargs))


def render_dating() -> str:
    return str(Prompt(
        f"{PROMPT_DIR}/determine_publication_time.md.j2",
        url="https://example.org/a",
        content="Published on 10 May 2024. Lorem ipsum.",
    ))


def render_assessment(property_name: str = "integrity", with_evidence: bool = True) -> str:
    evidence = [make_evidence(available_since=datetime(2024, 4, 15))] if with_evidence else []
    return str(Prompt(
        f"{PROMPT_DIR}/assess_from_evidence.md.j2",
        evidence=evidence,
        claim="Someone claimed X.",
        claim_date="May 01, 2024",
        medium=None,
        subject="Claim",
        property=PROPERTIES[property_name],
    ))


# --- Rendering -------------------------------------------------------------

def test_all_templates_render():
    for rendered in (render_extract_evidence(), render_faithfulness(),
                     render_temporal(), render_dating(), render_assessment()):
        assert rendered.strip()
        assert "{{" not in rendered and "{%" not in rendered


@pytest.mark.parametrize("property_name", list(PROPERTIES))
def test_assessment_template_renders_for_every_property(property_name):
    rendered = render_assessment(property_name)
    prop = PROPERTIES[property_name]
    assert prop.name in rendered
    assert prop.positive_category in rendered
    assert prop.negative_category in rendered


def test_assessment_template_handles_an_empty_evidence_set():
    rendered = render_assessment(with_evidence=False)
    assert "No evidence is available" in rendered


def test_assessment_template_shows_source_metadata():
    rendered = render_assessment()
    assert "Example" in rendered            # source name
    assert "news_article" in rendered       # source kind
    assert "secondary" in rendered          # proximity
    assert "April 15, 2024" in rendered     # available_since


def test_extraction_template_lists_every_source_kind():
    rendered = render_extract_evidence()
    for kind in SourceKind:
        assert kind.value in rendered


def test_extraction_template_states_the_exclusion_rules():
    rendered = render_extract_evidence().lower()
    assert "do not extract" in rendered
    assert "original source" in rendered
    assert "independently locatable" in rendered
    assert "tool" in rendered


# --- Non-circularity -------------------------------------------------------

def test_faithfulness_prompt_contains_neither_claim_nor_verdict():
    """§3.2: the faithfulness validator sees only proposition + source."""
    rendered = render_faithfulness()
    assert "Someone claimed X." not in rendered
    lowered = rendered.lower()
    for word in GOLD_VERDICT_WORDS:
        assert word not in lowered, f"faithfulness prompt leaks '{word}'"


def test_faithfulness_prompt_forbids_outside_knowledge():
    lowered = render_faithfulness().lower()
    assert "do not judge whether the proposition is true in the world" in lowered


def test_temporal_prompt_sees_the_claim_but_asserts_no_verdict():
    """§3.3 needs the claim to judge concurrency and later events, but must not
    be told - or asked for - the verdict."""
    rendered = render_temporal()
    assert "Someone claimed X." in rendered
    lowered = rendered.lower()
    assert "you are **not** asked whether the claim is true" in lowered
    for word in ("compromised", "intact", "gold verdict"):
        assert word not in lowered


def test_temporal_prompt_includes_the_registry_hint_when_given():
    rendered = render_temporal(publisher_hint="'Snopes' is a known fact-checking organization.")
    assert "Snopes" in rendered
    assert "Registry note" in rendered
    assert "Registry note" not in render_temporal()


def test_temporal_prompt_asks_all_three_questions_in_one_call():
    rendered = render_temporal()
    assert re.search(r"^## A\.", rendered, re.MULTILINE)
    assert re.search(r"^## B\.", rendered, re.MULTILINE)
    assert re.search(r"^## C\.", rendered, re.MULTILINE)
    assert "professional_fact_check" in rendered
    assert "concurrent_fact_check" in rendered
    assert "later_event" in rendered


def test_assessment_prompt_forbids_recalling_the_fact_check():
    lowered = render_assessment().lower()
    assert "do not rely on your own recollection" in lowered
    assert "base your reasoning **only** on the evidence" in lowered


def test_dating_prompt_excludes_modification_times():
    lowered = render_dating().lower()
    assert "last updated" in lowered
    assert "do not guess" in lowered
