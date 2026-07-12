"""Tests for parsing the JUSTIFICATION field out of two-step model responses."""

import pytest

from eval.baselines.common.types import get_label_scheme
from eval.baselines.providers.base import BaseFactChecker, extract_justification


class DummyFactChecker(BaseFactChecker):
    """Concrete BaseFactChecker so the extraction helpers can be exercised."""

    provider_name = "dummy"

    def check_claim(self, claim, image_paths=None, video_paths=None, claim_date=None):
        raise NotImplementedError


@pytest.fixture
def two_step_checker():
    return DummyFactChecker(
        model="dummy",
        label_scheme=get_label_scheme(7),
        seven_bin_prediction_mode="two_step",
    )


def test_extracts_from_exact_output_block():
    response = (
        "The photo was taken in 2019, not during the 2024 protest.\n\n"
        "  DIRECTION: COMPROMISED\n"
        "  CERTAINTY: rather certain\n"
        "  JUSTIFICATION: Reverse image search places the photo at a 2019 rally.\n"
        "  VERDICT: [COMPROMISED (RATHER CERTAIN)]\n"
    )
    assert extract_justification(response) == (
        "Reverse image search places the photo at a 2019 rally."
    )


def test_returns_empty_when_absent():
    assert extract_justification("Some analysis.\n\nVERDICT: [INTACT (CERTAIN)]") == ""


def test_returns_empty_for_empty_response():
    assert extract_justification("") == ""


@pytest.mark.parametrize(
    "line",
    [
        "**JUSTIFICATION:** The source confirms it.",
        "- JUSTIFICATION: The source confirms it.",
        "  justification : The source confirms it.",
    ],
)
def test_tolerates_markdown_and_bullet_variants(line):
    response = f"DIRECTION: INTACT\nCERTAINTY: certain\n{line}\nVERDICT: [INTACT (CERTAIN)]"
    assert extract_justification(response) == "The source confirms it."


def test_preserves_urls_and_markdown_links():
    body = (
        "Per [AP](https://apnews.com/article/a_(b),c) and "
        "[Reuters](https://reut.rs/x?y=1&z=2), the photo is authentic."
    )
    response = f"CERTAINTY: certain\nJUSTIFICATION: {body}\nVERDICT: [INTACT (CERTAIN)]"
    assert extract_justification(response) == body


def test_captures_trailing_justification_without_verdict_line():
    response = "DIRECTION: INTACT\nCERTAINTY: certain\nJUSTIFICATION: Confirmed by the registry."
    assert extract_justification(response) == "Confirmed by the registry."


def test_verdict_fallback_ignores_label_words_inside_justification(two_step_checker):
    """Two-step parsing fails here; the fallback must not read the justification."""
    response = (
        "Analysis of the image.\n"
        "JUSTIFICATION: Nothing suggests the photo is compromised or manipulated.\n"
        "INTACT (CERTAIN)"
    )
    assert two_step_checker._extract_verdict(response) == "Intact (certain)"


def test_two_step_verdict_still_parses_with_justification_present(two_step_checker):
    response = (
        "  DIRECTION: COMPROMISED\n"
        "  CERTAINTY: rather certain\n"
        "  JUSTIFICATION: The image predates the event.\n"
        "  VERDICT: [COMPROMISED (RATHER CERTAIN)]"
    )
    assert two_step_checker._extract_verdict(response) == "Compromised (rather certain)"
