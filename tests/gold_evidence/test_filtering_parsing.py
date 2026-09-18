"""Stage 2 response parsing (faithfulness and the temporal verdict)."""

import pytest

from veritas.gold_evidence.filtering import (
    parse_faithfulness_response,
    parse_later_event_response,
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


# --- The later-event judgement ---------------------------------------------

def later_event_response(change_detected=False, justification="Because.") -> str:
    return (
        "Reasoning about the timing.\n\n```json\n"
        f'{{"change_detected": {str(change_detected).lower()}, '
        f'"justification": "{justification}"}}\n```'
    )


def test_parses_the_later_event_judgement():
    parsed = parse_later_event_response(later_event_response(change_detected=True))
    assert parsed == {"change_detected": True, "justification": "Because."}


def test_no_change_is_not_confused_with_a_missing_answer():
    """`false` is a judgement, not the absence of one - and it is the answer for
    the common case: a source that merely reports on an earlier state."""
    parsed = parse_later_event_response(later_event_response(change_detected=False))
    assert parsed["change_detected"] is False


def test_yes_no_strings_are_accepted():
    assert parse_later_event_response(
        '```json\n{"change_detected": "NO"}\n```')["change_detected"] is False
    assert parse_later_event_response(
        '```json\n{"change_detected": "yes"}\n```')["change_detected"] is True


def test_missing_justification_is_tolerated():
    parsed = parse_later_event_response('```json\n{"change_detected": true}\n```')
    assert parsed["change_detected"] is True
    assert parsed["justification"] is None


def test_unparseable_response_returns_none():
    assert parse_later_event_response("I cannot answer that.") is None


def test_response_without_the_judgement_returns_none():
    assert parse_later_event_response('```json\n{"foo": 1}\n```') is None


def test_json_without_a_fence_is_accepted():
    parsed = parse_later_event_response('{"change_detected": false, "justification": "x"}')
    assert parsed["change_detected"] is False
