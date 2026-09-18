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
        proposition="The mayor signed the decree on 3 May.",
        sources=[{
            "name": "Example News",
            "kind": "news_article",
            "locator": "https://example.org/a",
            "available_since": "May 10, 2024",
            "excerpt": "An article about X.",
            "truncated": False,
        }],
    )
    kwargs.update(overrides)
    return str(Prompt(f"{PROMPT_DIR}/validate_temporally.md.j2", **kwargs))


def render_dating() -> str:
    return str(Prompt(
        f"{PROMPT_DIR}/determine_publication_time.md.j2",
        url="https://example.org/a",
        content="Published on 10 May 2024. Lorem ipsum.",
    ))


def render_assessment(property_name: str = "integrity", with_evidence: bool = True,
                      rationales=()) -> str:
    evidence = [make_evidence(available_since=datetime(2024, 4, 15))] if with_evidence else []
    return str(Prompt(
        f"{PROMPT_DIR}/assess_from_evidence.md.j2",
        evidence=evidence,
        rationales=list(rationales),
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


def test_assessment_template_lists_every_source_of_a_proposition():
    from tests.gold_evidence.conftest import make_evidence, make_source
    from veritas.common import Prompt
    from veritas.common.annotation import PROPERTIES

    evidence = [make_evidence(sources=[
        make_source(name="Reuters", locator="https://a/1"),
        make_source(name="City Register", locator="https://a/2"),
    ])]
    rendered = str(Prompt(f"{PROMPT_DIR}/assess_from_evidence.md.j2",
                          evidence=evidence, rationales=[], claim="Someone claimed X.",
                          claim_date=None, medium=None, subject="Claim",
                          property=PROPERTIES["integrity"]))
    assert "Reuters" in rendered
    assert "City Register" in rendered


def test_extraction_template_lists_every_source_kind():
    rendered = render_extract_evidence()
    for kind in SourceKind:
        assert kind.value in rendered


def test_extraction_template_asks_for_the_rationale_and_sources():
    rendered = render_extract_evidence()
    assert "verdict_rationale" in rendered
    assert '"sources"' in rendered
    # The narrowed definition of "essential" is what the roles are judged against.
    assert "the Verdict Rationale breaks without this proposition" in rendered


def test_extraction_template_forbids_a_verdict_in_the_rationale():
    lowered = render_extract_evidence().lower()
    assert "must **not** state or imply the verdict" in lowered
    assert "must **not** introduce a new externally verifiable factual premise" in lowered


def test_extraction_template_binds_the_rationale_to_essential_evidence():
    """The rationale may only build on what the instance is disqualified for
    losing, so it can never outlive the evidence it rests on."""
    lowered = render_extract_evidence().lower()
    assert "may build **only on propositions you marked `essential`**" in lowered


def test_assessment_template_shows_the_rationale_when_there_is_one():
    from tests.gold_evidence.conftest import make_rationale

    rendered = render_assessment(rationales=[make_rationale()])
    assert "Reasoning Aid" in rendered
    assert "3 May is before 5 May" in rendered
    assert "asserts **no facts of its own**" in rendered


def test_assessment_template_omits_the_section_without_a_rationale():
    assert "Reasoning Aid" not in render_assessment()


def test_assessment_template_handles_evidence_free_instances():
    from tests.gold_evidence.conftest import make_rationale

    rendered = render_assessment(with_evidence=False, rationales=[make_rationale()])
    assert "No evidence items are available" in rendered
    assert "Reasoning Aid" in rendered


def test_extraction_template_keeps_provenance_out_of_the_proposition():
    """The per-source faithfulness check compares one source against the whole
    proposition, so a proposition that quantifies over its sources ("multiple
    posts show X") is unsatisfiable for every one of them. Extraction has to write
    the bare fact instead and leave the cardinality to the `sources` list."""
    lowered = render_extract_evidence().lower()
    assert "write the fact, not the citation" in lowered
    assert "each source must carry the proposition alone" in lowered
    assert "interchangeable witnesses to the same fact" in lowered
    # Named attribution stays: it is content, not provenance.
    assert "named attribution is different" in lowered


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


def test_temporal_prompt_shows_the_evidence_it_judges():
    """Without it the model would be asked about an empty proposition."""
    rendered = render_temporal()
    assert "The mayor signed the decree on 3 May." in rendered
    assert re.search(r"^## The Evidence$", rendered, re.MULTILINE)


def test_temporal_prompt_lists_every_source_with_its_date():
    rendered = render_temporal(sources=[
        {"name": "Example News", "kind": "news_article", "locator": "https://example.org/a",
         "available_since": "May 10, 2024", "excerpt": "An article about X.",
         "truncated": False},
        {"name": "City Register", "kind": "government_record",
         "locator": "https://example.org/b", "available_since": "May 12, 2024",
         "excerpt": "A register entry.", "truncated": False},
    ])
    assert "Example News" in rendered and "City Register" in rendered
    assert "May 10, 2024" in rendered and "May 12, 2024" in rendered
    assert "A register entry." in rendered


def test_temporal_prompt_without_source_content_says_so():
    """Tools and offline evidence have no content to show; the model must know that
    it is judging the proposition rather than an empty page."""
    rendered = render_temporal(sources=[{
        "name": "Prof. Meier (phone interview)", "kind": "offline", "locator": None,
        "available_since": "May 10, 2024", "excerpt": "", "truncated": False,
    }])
    assert "could not be retrieved" in rendered
    assert "The mayor signed the decree on 3 May." in rendered


def test_temporal_prompt_asks_for_the_judgement_that_is_stored():
    """The model answers the field that is stored, so no answer gets inverted on
    the way into `LaterEventCheck`."""
    rendered = render_temporal()
    assert '"change_detected"' in rendered
    assert "true_at_t_c" not in rendered


def test_temporal_prompt_separates_reporting_from_a_changing_world():
    """The confusion this check exists to avoid: a correction issued after the
    claim is reporting, not a change of the matter it reports on."""
    lowered = render_temporal().lower()
    assert "is not itself a change in the relevant world state" in lowered
    assert "correction" in lowered
    assert "had this evidence been available at t_c" in lowered


def test_assessment_prompt_forbids_recalling_the_fact_check():
    lowered = render_assessment().lower()
    assert "do not rely on your own recollection" in lowered
    assert "base your reasoning **only** on the evidence" in lowered


def test_dating_prompt_excludes_modification_times():
    lowered = render_dating().lower()
    assert "last updated" in lowered
    assert "do not guess" in lowered
