"""Stage 6: label parsing, the assessment prompt, and the property cascade.

Pure tests: no database, no network, no LLM calls. The property assessments are
replaced by a stub that records what it was given.
"""

import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from veritas.common import Claim, Prompt, Verdict
from veritas.common.annotation import PROPERTIES, Rating
from veritas.common.annotation.rating import RatingAggregated
from veritas.common.verdict import MediumVerdict
from veritas.pipeline import stage_6
from veritas.pipeline.stage_6 import PriorAssessment, extract_label_from_response, medium_subject

RATERS = ["stub:a", "stub:b", "stub:c"]


def response(output: str) -> SimpleNamespace:
    return SimpleNamespace(output=output, model=SimpleNamespace(specifier="stub:a"))


def answer(reasoning: str = "Step-by-step reasoning with `quotes`, _emphasis_ and colons: here.",
           **fields) -> SimpleNamespace:
    """A response with free reasoning, concluded by the final JSON answer."""
    return response(f"{reasoning}\n\n```json\n{json.dumps(fields)}\n```")


def rating(score: float, raters=RATERS, explanation="merged") -> RatingAggregated:
    aggregated = RatingAggregated(
        individual_ratings=[Rating(score=score, rater=r, explanation="because") for r in raters],
        rater="ensemble")
    aggregated.explanation = explanation
    return aggregated


# --- Label parsing ---------------------------------------------------------

def test_parses_the_final_json_answer():
    r = extract_label_from_response(answer(category="False", certainty="Rather certain",
                                           explanation="The record contradicts the claim."),
                                    PROPERTIES["veracity"])
    assert r.score == pytest.approx(-2 / 3)
    assert r.explanation == "The record contradicts the claim."
    assert r.tags == []


def test_labels_in_the_reasoning_are_ignored():
    r = extract_label_from_response(answer("Maybe `True`, _certain_? No.", category="False",
                                           certainty="Certain", explanation="Contradicted."),
                                    PROPERTIES["veracity"])
    assert r.score == -1


def test_the_answer_is_case_insensitive():
    r = extract_label_from_response(answer(category="true", certainty="RATHER UNCERTAIN",
                                           explanation="Weak support."), PROPERTIES["veracity"])
    assert r.score == pytest.approx(1 / 3)


def test_tags_are_canonicalized_and_restricted_to_the_chosen_category():
    r = extract_label_from_response(answer(category="Fabricated", certainty="Certain",
                                           tags=["ai-generated", "Manipulated", "Pristine-ish"],
                                           explanation="Synthesized."), PROPERTIES["authenticity"])
    assert r.tags == ["AI-generated", "Manipulated"]

    r = extract_label_from_response(answer(category="Pristine", certainty="Certain", tags=["Forged"],
                                           explanation="A real photo."), PROPERTIES["authenticity"])
    assert r.tags == []


def test_unknown_needs_no_certainty():
    r = extract_label_from_response(answer(category="Unknown", certainty=None, explanation="No evidence."),
                                    PROPERTIES["veracity"])
    assert r.score == 0


def test_minor_json_syntax_errors_are_tolerated():
    r = extract_label_from_response(response(
        'Reasoning.\n```json\n{"category": "False", "certainty": "Certain", "explanation": "No.",}\n```'),
        PROPERTIES["veracity"])
    assert r.score == -1


@pytest.mark.parametrize("output", [
    "`False` _Certain_ ```Because.```",  # The old syntax, no JSON
    "Reasoning only.",  # No final answer at all
    '```json\n{"category": "False", "certainty": "Certain"}\n```',  # No explanation
    '```json\n{"category": "Maybe", "certainty": "Certain", "explanation": "x"}\n```',  # Invalid category
    '```json\n{"category": "False", "certainty": null, "explanation": "x"}\n```',  # No certainty
    '```json\n{"category": "False", "certainty": "Sure", "explanation": "x"}\n```',  # Invalid certainty
])
def test_invalid_responses_are_rejected(output):
    with pytest.raises(ValueError):
        extract_label_from_response(response(output), PROPERTIES["veracity"])


# --- Prompt ------------------------------------------------------------------

CLAIM = Claim(id=1, data="The mayor resigned.", date=datetime(2024, 5, 1),
              appearance_ids=set(), review_ids={1})
REVIEWS = [("The mayor did not resign.", SimpleNamespace(name="Checker"))]


def render(property_name: str, prior_assessments=(), has_media=False) -> str:
    prompt = Prompt("veritas/prompts/assess_property.md.j2", reviews=REVIEWS, claim=CLAIM,
                    medium=None, subject="Claim", property=PROPERTIES[property_name],
                    has_media=has_media, prior_assessments=list(prior_assessments))
    return str(prompt)


def test_the_prompt_omits_prior_assessments_without_any():
    assert "Prior Assessments" not in render("veracity")


def test_the_prompt_shows_the_prior_assessments():
    priors = [PriorAssessment(PROPERTIES["contextualization"], "Image", rating(1, explanation="Matches."))]
    text = render("veracity", priors, has_media=True)
    assert "# Prior Assessments" in text
    assert "**Contextualization of the Image**: Correct (certain). Matches." in text
    assert "Concentrate exclusively on the Veracity" in text


def test_the_veracity_media_hint_is_shown_only_with_media():
    hint = PROPERTIES["veracity"].media_hints
    assert hint and hint not in render("veracity")
    assert hint in render("veracity", has_media=True)


def test_the_prompt_asks_for_a_final_json_answer():
    text = render("authenticity")
    assert "```json" in text
    assert '"tags"' in text
    assert '"tags"' not in render("veracity")  # Veracity has no tags


def test_the_context_coverage_hint_does_not_presume_a_veracity():
    assert "evaluated as true" not in render("context_coverage")


def test_medium_subjects_are_numbered_only_for_multiple_media():
    image = SimpleNamespace(kind="image")
    assert medium_subject(image, 0, 1) == "Image"
    assert medium_subject(image, 1, 2).startswith("Image 2")


# --- Cascade -----------------------------------------------------------------

class FakeClaim:
    """Just enough of a Claim for `predict_verdict_single`."""

    def __init__(self, data="The mayor resigned.", verdict: Verdict | None = None):
        self.id = 1
        self.data = data
        self.review_ids = {1}
        self.verdict_ids = set()
        self.is_rectified = False
        self.is_original = True
        self.dismissed = False
        self.dismiss_reasons = []
        self._verdict = verdict
        self.review = SimpleNamespace(stage=None)

        async def set_stage(stage):
            self.review.stage = stage

        self.review.set_stage = set_stage

    @property
    async def current_verdict(self):
        return self._verdict

    @property
    async def reviews(self):
        return [self.review]

    async def save_to_db(self):
        pass

    async def dismiss(self, reason, also_dismiss_reviews=True):
        self.dismissed = True
        self.dismiss_reasons.append(reason)


@pytest.fixture
def cascade(monkeypatch):
    """Stubs the property assessments with configurable scores and records the
    calls. Media are stubbed so that no media store is needed."""
    state = {"scores": {}, "calls": []}

    async def fake_assess(property_name, claim, medium=None, incomplete_rating=None, prior_assessments=()):
        state["calls"].append({
            "property": property_name,
            "medium": medium.reference if medium else None,
            "incomplete": incomplete_rating,
            "priors": [(p.property.name, p.subject, p.rating.score) for p in prior_assessments],
        })
        if incomplete_rating is not None:
            return incomplete_rating
        return rating(state["scores"].get(property_name, 1))

    class FakeSequence:
        def __init__(self, data):
            self.data = data

        def unique_items(self):
            return [SimpleNamespace(reference=ref, kind="image") for ref in state.get("media", [])]

    async def fake_save(self):
        self.id = self.id or 99

    monkeypatch.setattr(stage_6, "assess_property", fake_assess)
    monkeypatch.setattr(stage_6, "MultimodalSequence", FakeSequence)
    monkeypatch.setattr(Verdict, "save_to_db", fake_save)
    monkeypatch.setattr(MediumVerdict, "medium",
                        property(lambda self: SimpleNamespace(reference=self.reference)))
    return state


def properties_called(state):
    return [call["property"] for call in state["calls"]]


@pytest.mark.asyncio
async def test_later_properties_see_the_prior_assessments(cascade):
    cascade["media"] = ["<image:1>", "<image:2>"]
    await stage_6.predict_verdict_single(FakeClaim())

    calls = {(c["property"], c["medium"]): c["priors"] for c in cascade["calls"]}
    assert calls[("authenticity", "<image:1>")] == []
    assert calls[("contextualization", "<image:2>")] == [("Authenticity", "Image", 1)]
    veracity_priors = calls[("veracity", None)]
    assert [(name, subject.split(" (")[0]) for name, subject, _ in veracity_priors] == [
        ("Authenticity", "Image 1"), ("Contextualization", "Image 1"),
        ("Authenticity", "Image 2"), ("Contextualization", "Image 2")]
    assert calls[("context_coverage", None)] == veracity_priors + [("Veracity", "Claim", 1)]


@pytest.mark.asyncio
async def test_fabricated_media_are_still_contextualized(cascade):
    cascade["media"] = ["<image:1>"]
    cascade["scores"] = {"authenticity": -1}
    await stage_6.predict_verdict_single(FakeClaim())
    assert properties_called(cascade) == ["authenticity", "contextualization", "veracity", "context_coverage"]


@pytest.mark.parametrize("veracity, assesses_context_coverage", [
    (-1, False), (-2 / 3, False), (-1 / 3, True), (0, True), (1 / 3, True), (1, True)])
@pytest.mark.asyncio
async def test_context_coverage_is_skipped_only_for_false_claims(cascade, veracity, assesses_context_coverage):
    cascade["scores"] = {"veracity": veracity}
    await stage_6.predict_verdict_single(FakeClaim())
    assert ("context_coverage" in properties_called(cascade)) == assesses_context_coverage


@pytest.mark.asyncio
async def test_completion_keeps_and_completes_the_authenticity(cascade):
    cascade["media"] = ["<image:1>"]
    authenticity = rating(1, raters=RATERS[:2])
    verdict = Verdict(id=5, claim_id=1, review_ids={1},
                      media_verdicts=[MediumVerdict(reference="<image:1>", authenticity=authenticity,
                                                    contextualization=rating(1))],
                      veracity=rating(1), context_coverage=rating(1))
    await stage_6.predict_verdict_single(FakeClaim(verdict=verdict))

    authenticity_call = next(c for c in cascade["calls"] if c["property"] == "authenticity")
    assert authenticity_call["incomplete"] is authenticity
    assert verdict.media_verdicts[0].authenticity is authenticity


@pytest.mark.asyncio
async def test_completion_stores_newly_assessed_properties(cascade):
    verdict = Verdict(id=5, claim_id=1, review_ids={1}, veracity=rating(1), context_coverage=None)
    await stage_6.predict_verdict_single(FakeClaim(verdict=verdict))
    assert verdict.context_coverage is not None


@pytest.mark.asyncio
async def test_completion_drops_ratings_no_longer_reached(cascade):
    """If the completed veracity turned false, the old context coverage is obsolete."""
    false_veracity = rating(-1)
    verdict = Verdict(id=5, claim_id=1, review_ids={1}, veracity=false_veracity, context_coverage=rating(1))
    await stage_6.predict_verdict_single(FakeClaim(verdict=verdict))
    assert verdict.veracity is false_veracity
    assert verdict.context_coverage is None
