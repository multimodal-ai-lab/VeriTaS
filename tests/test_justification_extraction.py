"""Tests for parsing the model's JSON answer: the verdict (integrity + certainty) and the justification."""

import pytest

from eval.baselines.common.parsing import extract_answer, extract_justification, extract_verdict


def _answer(integrity, certainty, justification="Per [AP](https://apnews.com/x) it is.") -> str:
    return (
        "```json\n"
        "{\n"
        f'  "integrity": {integrity},\n'
        f'  "certainty": {certainty},\n'
        f'  "justification": "{justification}"\n'
        "}\n"
        "```"
    )


@pytest.mark.parametrize(
    "integrity, certainty, expected",
    [
        ('"intact"', '"certain"', "Intact (certain)"),
        ('"intact"', '"rather certain"', "Intact (rather certain)"),
        ('"intact"', '"rather uncertain"', "Intact (rather uncertain)"),
        ('"unknown"', "null", "Unknown"),
        ('"compromised"', '"rather uncertain"', "Compromised (rather uncertain)"),
        ('"compromised"', '"rather certain"', "Compromised (rather certain)"),
        ('"compromised"', '"certain"', "Compromised (certain)"),
        ('"Compromised"', '"Rather Certain"', "Compromised (rather certain)"),
        ('" intact "', '"certain "', "Intact (certain)"),
        # "unknown" takes no certainty, but one given anyway does no harm
        ('"unknown"', '"certain"', "Unknown"),
    ],
)
def test_verdict_from_integrity_and_certainty(integrity, certainty, expected):
    assert extract_verdict(_answer(integrity, certainty)) == expected


@pytest.mark.parametrize(
    "integrity, certainty",
    [
        ('"intact"', "null"),  # A tendency needs a certainty
        ('"compromised"', '"very certain"'),
        ('"misleading"', '"certain"'),
        ("null", "null"),
    ],
)
def test_invalid_answer_is_no_verdict(integrity, certainty):
    """Not "Unknown": the model did not answer as asked, and the claim is retried."""
    assert extract_verdict(_answer(integrity, certainty)) is None


def test_justification_keeps_urls_and_markdown_links():
    body = "Per [AP](https://apnews.com/article/a_(b),c) and [Reuters](https://reut.rs/x?y=1&z=2), it is authentic."
    assert extract_justification(_answer('"intact"', '"certain"', body)) == body


def test_answer_without_code_block():
    response = 'After checking: {"integrity": "intact", "certainty": "certain", "justification": "Confirmed."}'
    assert extract_verdict(response) == "Intact (certain)"
    assert extract_justification(response) == "Confirmed."


def test_last_answer_wins():
    draft = '{"integrity": "unknown", "certainty": null, "justification": "Draft."}'
    final = '{"integrity": "compromised", "certainty": "certain", "justification": "Final."}'
    response = f"First thought:\n{draft}\nAfter more research:\n```json\n{final}\n```"
    assert extract_verdict(response) == "Compromised (certain)"
    assert extract_justification(response) == "Final."


def test_braces_and_line_breaks_inside_the_justification():
    response = '{"integrity": "intact", "certainty": "certain", "justification": "The {quoted} part\nis real."}'
    assert extract_verdict(response) == "Intact (certain)"
    assert extract_justification(response) == "The {quoted} part\nis real."


def test_other_json_objects_are_ignored():
    response = (
        '```json\n{"integrity": "intact", "certainty": "rather certain", "justification": "Confirmed."}\n```\n'
        'Searched with {"query": "eiffel tower"}.'
    )
    assert extract_verdict(response) == "Intact (rather certain)"
    assert extract_answer(response)["justification"] == "Confirmed."


@pytest.mark.parametrize(
    "response",
    [
        "",
        "The claim is intact, I am certain.",
        "VERDICT: Intact (certain)",  # The old format
        '{"integrity": "intact", "certainty": "certain", "justification": "Trailing comma",}',
    ],
)
def test_missing_or_broken_answer_is_no_verdict(response):
    assert extract_answer(response) == {}
    assert extract_verdict(response) is None
    assert extract_justification(response) == ""
