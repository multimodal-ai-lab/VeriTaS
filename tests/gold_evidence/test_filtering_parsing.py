"""Stage 2 response parsing (faithfulness and the temporal verdict)."""

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

def temporal_response(later_event=False, justification="Because.") -> str:
    return (
        "Reasoning about the timing.\n\n```json\n"
        f'{{"later_event": {str(later_event).lower()}, '
        f'"justification": "{justification}"}}\n```'
    )


def test_parses_the_later_event_judgement():
    parsed = parse_temporal_response(temporal_response(later_event=True))
    assert parsed == {"later_event": True, "justification": "Because."}


def test_no_later_event_is_not_confused_with_a_missing_answer():
    """`false` is a judgement, not the absence of one. The prompt asks directly for
    `later_event`, so nothing on this path inverts the model's answer."""
    parsed = parse_temporal_response(temporal_response(later_event=False))
    assert parsed["later_event"] is False


def test_yes_no_strings_are_accepted():
    assert parse_temporal_response('```json\n{"later_event": "NO"}\n```')["later_event"] is False
    assert parse_temporal_response('```json\n{"later_event": "yes"}\n```')["later_event"] is True


def test_missing_justification_is_tolerated():
    parsed = parse_temporal_response('```json\n{"later_event": true}\n```')
    assert parsed["later_event"] is True
    assert parsed["justification"] is None


def test_unparseable_response_returns_none():
    assert parse_temporal_response("I cannot answer that.") is None


def test_response_without_the_judgement_returns_none():
    assert parse_temporal_response('```json\n{"foo": 1}\n```') is None


def test_json_without_a_fence_is_accepted():
    parsed = parse_temporal_response('{"later_event": false, "justification": "x"}')
    assert parsed["later_event"] is False
