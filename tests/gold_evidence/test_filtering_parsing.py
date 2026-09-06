"""Stage 2 response parsing (faithfulness and the single-call temporal verdict)."""

import pytest

from veritas.gold_evidence.filtering import (
    parse_faithfulness_response,
    parse_temporal_response,
)


# --- Faithfulness ----------------------------------------------------------

def faithfulness_response(category: str, certainty: str | None = "Certain",
                          explanation: str = "The source says so.") -> str:
    parts = [f"The source addresses the proposition.\n\n`{category}`"]
    if certainty:
        parts.append(f"_{certainty}_")
    parts.append(f"```\n{explanation}\n```")
    return "\n\n".join(parts)


@pytest.mark.parametrize("category, certainty, expected", [
    ("Entails", "Certain", 1.0),
    ("Entails", "Rather certain", 2 / 3),
    ("Entails", "Rather uncertain", 1 / 3),
    ("Contradicts", "Certain", -1.0),
    ("Contradicts", "Rather certain", -2 / 3),
    ("Contradicts", "Rather uncertain", -1 / 3),
])
def test_scores_span_the_full_scale(category, certainty, expected):
    score, explanation = parse_faithfulness_response(
        faithfulness_response(category, certainty))
    assert score == pytest.approx(expected)
    assert explanation == "The source says so."


def test_unknown_is_neutral_and_needs_no_certainty():
    score, _ = parse_faithfulness_response(faithfulness_response("Unknown", certainty=None))
    assert score == pytest.approx(0.0)


def test_missing_certainty_on_a_decided_category_is_unusable():
    score, _ = parse_faithfulness_response(faithfulness_response("Entails", certainty=None))
    assert score is None


def test_invalid_category_is_unusable():
    score, _ = parse_faithfulness_response(faithfulness_response("Maybe"))
    assert score is None


def test_missing_category_is_unusable():
    score, _ = parse_faithfulness_response("I am not sure what to answer here.")
    assert score is None


def test_category_is_case_insensitive():
    score, _ = parse_faithfulness_response(faithfulness_response("ENTAILS", "certain"))
    assert score == pytest.approx(1.0)


# --- Temporal validation ---------------------------------------------------

def temporal_response(professional=True, concurrent=None, later_event=False,
                      reasoning="Because.") -> str:
    return (
        "Reasoning about A, B and C.\n\n```json\n"
        f'{{"professional_fact_check": {str(professional).lower()}, '
        f'"concurrent_fact_check": {"null" if concurrent is None else str(concurrent).lower()}, '
        f'"later_event": {str(later_event).lower()}, '
        f'"reasoning": "{reasoning}"}}\n```'
    )


def test_parses_all_three_judgements_from_one_call():
    parsed = parse_temporal_response(
        temporal_response(professional=True, concurrent=True, later_event=False))
    assert parsed == {
        "professional_fact_check": True,
        "concurrent_fact_check": True,
        "later_event": False,
        "reasoning": "Because.",
    }


def test_concurrency_is_nulled_when_not_a_fact_check():
    """A non-fact-check cannot be a concurrent fact-check, whatever the model said."""
    parsed = parse_temporal_response(
        temporal_response(professional=False, concurrent=True))
    assert parsed["professional_fact_check"] is False
    assert parsed["concurrent_fact_check"] is None


def test_yes_no_strings_are_accepted():
    response = ('```json\n{"professional_fact_check": "yes", '
                '"concurrent_fact_check": "no", "later_event": "NO"}\n```')
    parsed = parse_temporal_response(response)
    assert parsed["professional_fact_check"] is True
    assert parsed["concurrent_fact_check"] is False
    assert parsed["later_event"] is False


def test_missing_keys_default_to_false():
    parsed = parse_temporal_response('```json\n{"later_event": true}\n```')
    assert parsed["professional_fact_check"] is False
    assert parsed["concurrent_fact_check"] is None
    assert parsed["later_event"] is True


def test_unparseable_response_returns_none():
    assert parse_temporal_response("I cannot answer that.") is None


def test_response_without_any_recognized_key_returns_none():
    assert parse_temporal_response('```json\n{"foo": 1}\n```') is None


def test_json_without_a_fence_is_accepted():
    parsed = parse_temporal_response('{"professional_fact_check": true, "later_event": false}')
    assert parsed["professional_fact_check"] is True
