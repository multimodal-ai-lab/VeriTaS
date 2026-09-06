"""Stage 3 - Evidence sufficiency validation (Spec §4).

An ensemble of strong (M)LLMs from different model families receives *only* the
claim and the retained multimodal evidence and predicts a VeriTaS verdict with
high reasoning effort. The instance is kept iff that prediction is sufficiently
close to the gold verdict.

The ensemble is a **sufficiency validator, not a mechanism for revising the gold
verdict**: nothing in this module writes to `verdicts` or changes a gold rating.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

from ezmm import Item, MultimodalSequence
from veritas.common import Claim, Prompt, Verdict
from veritas.common.annotation import PROPERTIES, Property, Rating
from veritas.common.annotation.rating import Category3Bin, RatingAggregated
from veritas.common.verdict import MediumVerdict
from veritas.ensemble import Ensemble, ModelResponse, ensemble as global_ensemble
from veritas.gold_evidence import (
    ensemble_mode as default_ensemble_mode,
    ensemble_models,
    min_ensemble_ratings,
    proximity_threshold as default_threshold,
    reasoning_effort_sufficiency,
)
from veritas.gold_evidence.llm import FATAL_ERRORS
from veritas.gold_evidence.closeness import (
    ENSEMBLE_MODES,
    MODE_FULL,
    MODE_INTEGRITY,
    PredictedVerdict,
    is_close,
    property_diffs,
)
from veritas.gold_evidence.models import Evidence
from veritas.pipeline.stage_6 import extract_label_from_response

logger = logging.getLogger("VeriTaS")

PROMPT_PATH = "veritas/gold_evidence/prompts/assess_from_evidence.md.j2"

#: Cap on the stored raw model output per member, in characters.
MAX_STORED_OUTPUT = 20_000

_ensemble: Ensemble | None = None


def get_ensemble() -> Ensemble:
    """The sufficiency ensemble. Uses the models configured under
    `gold_evidence.ensemble_models`, falling back to the global VeriTaS ensemble."""
    global _ensemble
    if _ensemble is None:
        _ensemble = Ensemble(ensemble_models) if ensemble_models else global_ensemble
    return _ensemble


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

@dataclass
class MemberResponse:
    """One ensemble member's answer, keeping its full reasoning for the record."""

    rating: Rating
    raw_output: str
    #: The model's reasoning trace as reported by the provider's API.
    reasoning: str | None = None

    def to_dict(self) -> dict:
        return {
            "model": self.rating.rater,
            "score": self.rating.score,
            "tags": list(self.rating.tags or []),
            # The justification the prompt asked the model to state ...
            "explanation": self.rating.explanation,
            # ... the reasoning the provider's API reported for the call ...
            "reasoning": self.reasoning[:MAX_STORED_OUTPUT] if self.reasoning else None,
            # ... and the complete answer text.
            "raw_output": self.raw_output[:MAX_STORED_OUTPUT],
        }


@dataclass
class SufficiencyResult:
    """Outcome of validating one evidence set against the gold verdict."""

    claim_id: int
    condition: str
    ensemble_mode: str
    n_evidence: int
    threshold: float
    predicted: PredictedVerdict | None = None
    member_responses: dict[str, list[dict]] = field(default_factory=dict)
    property_diffs: dict[str, float] = field(default_factory=dict)
    max_property_diff: float | None = None
    is_close: bool | None = None
    model_specifiers: list[str] = field(default_factory=list)
    error: str | None = None

    def to_db_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "condition": self.condition,
            "ensemble_mode": self.ensemble_mode,
            "n_evidence": self.n_evidence,
            "predicted_verdict": self.predicted.model_dump(mode="json") if self.predicted else None,
            "member_responses": self.member_responses,
            "property_diffs": self.property_diffs,
            "max_property_diff": self.max_property_diff,
            "is_close": self.is_close,
            "threshold": self.threshold,
            "model_specifiers": self.model_specifiers,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

async def validate_sufficiency(
        claim: Claim,
        evidence: Iterable[Evidence],
        gold: Verdict,
        *,
        condition: str,
        mode: str = None,
        threshold: float = None,
) -> SufficiencyResult:
    """Predicts a verdict from the evidence alone and compares it to the gold verdict."""
    if mode is None:
        mode = default_ensemble_mode
    if threshold is None:
        threshold = default_threshold
    assert mode in ENSEMBLE_MODES, f"Unknown ensemble mode: {mode}"

    evidence = list(evidence)
    result = SufficiencyResult(
        claim_id=claim.id,
        condition=condition,
        ensemble_mode=mode,
        n_evidence=len(evidence),
        threshold=threshold,
        model_specifiers=get_ensemble().model_names,
    )

    if not evidence:
        result.error = "No evidence in this condition."
        result.is_close = False
        return result

    try:
        if mode == MODE_INTEGRITY:
            predicted = await _predict_integrity(claim, evidence, result)
        else:
            predicted = await _predict_full(claim, evidence, result)
    except FATAL_ERRORS:
        # A degraded ensemble would silently reject the claim; abort the run instead.
        raise
    except Exception as e:
        logger.warning(f"Sufficiency validation failed for claim {claim.id} "
                       f"({condition}): {type(e).__name__}: {e}")
        result.error = f"{type(e).__name__}: {e}"
        return result

    if predicted is None or predicted.n_ratings < min_ensemble_ratings:
        result.error = (result.error or
                        f"Fewer than {min_ensemble_ratings} usable ensemble ratings.")
        return result

    result.predicted = predicted
    result.property_diffs = property_diffs(predicted, gold)
    result.max_property_diff = max(result.property_diffs.values()) if result.property_diffs else None
    result.is_close = is_close(predicted, gold, threshold)
    return result


async def _predict_integrity(claim: Claim, evidence: list[Evidence],
                             result: SufficiencyResult) -> PredictedVerdict | None:
    """One ensemble call predicting the claim's integrity directly."""
    rating, responses = await assess_property_from_evidence(
        "integrity", claim, evidence)
    result.member_responses["integrity"] = [r.to_dict() for r in responses]
    if rating is None:
        return None
    return PredictedVerdict(claim_id=claim.id, mode=MODE_INTEGRITY, integrity=rating)


async def _predict_full(claim: Claim, evidence: list[Evidence],
                        result: SufficiencyResult) -> PredictedVerdict | None:
    """The full stage-6 property cascade, driven by evidence instead of reviews."""
    media = MultimodalSequence(claim.data).unique_items()
    media_verdicts: list[MediumVerdict] = []
    incorrect_contextualization = False

    for medium in media:
        authenticity, responses = await assess_property_from_evidence(
            "authenticity", claim, evidence, medium=medium)
        result.member_responses[f"authenticity[{medium.reference}]"] = [
            r.to_dict() for r in responses]

        contextualization, responses = await assess_property_from_evidence(
            "contextualization", claim, evidence, medium=medium)
        result.member_responses[f"contextualization[{medium.reference}]"] = [
            r.to_dict() for r in responses]

        if authenticity is None or contextualization is None:
            result.error = f"Could not assess medium {medium.reference}."
            return None

        media_verdicts.append(MediumVerdict(reference=medium.reference,
                                            authenticity=authenticity,
                                            contextualization=contextualization))
        if contextualization.as_3_bin() == Category3Bin.NEGATIVE:
            incorrect_contextualization = True

    veracity = context_coverage = None
    if not incorrect_contextualization:
        veracity, responses = await assess_property_from_evidence("veracity", claim, evidence)
        result.member_responses["veracity"] = [r.to_dict() for r in responses]
        if veracity is not None and veracity.score > 0:
            context_coverage, responses = await assess_property_from_evidence(
                "context_coverage", claim, evidence)
            result.member_responses["context_coverage"] = [r.to_dict() for r in responses]

    predicted = PredictedVerdict(
        claim_id=claim.id,
        mode=MODE_FULL,
        veracity=veracity,
        context_coverage=context_coverage,
        media_verdicts=media_verdicts,
    )
    if predicted.effective_integrity is None:
        result.error = "No property could be assessed from the evidence."
        return None
    return predicted


async def assess_property_from_evidence(
        property_name: str,
        claim: Claim,
        evidence: list[Evidence],
        medium: Item | None = None,
) -> tuple[RatingAggregated | None, list[MemberResponse]]:
    """Queries the ensemble for one property, given only claim + evidence.

    Returns the aggregated rating and every member's full response, so the
    individual reasonings can be persisted alongside the result."""
    prop: Property = PROPERTIES[property_name]
    subject = "Claim" if medium is None else medium.kind.capitalize()

    prompt = build_prompt(claim, evidence, prop, subject=subject, medium=medium)

    def extract(response: ModelResponse) -> MemberResponse:
        rating = extract_label_from_response(response, prop)
        return MemberResponse(
            rating=rating,
            raw_output=str(response.output) if response.output is not None else "",
            reasoning=response.reasoning,
        )

    responses: list[MemberResponse] = await get_ensemble().generate(
        prompt,
        response_extraction_fn=extract,
        aggregation_fn=lambda rs: rs,
        reasoning_effort=reasoning_effort_sufficiency,
    )

    if not responses:
        return None, []
    rating = RatingAggregated(individual_ratings=[r.rating for r in responses],
                              rater="ensemble")
    return rating, responses


def build_prompt(claim: Claim, evidence: list[Evidence], prop: Property, *,
                 subject: str, medium: Item | None = None) -> Prompt:
    """Composes the evidence-only prompt. The claim is included for every property
    except authenticity, mirroring `stage_6.assess_property`."""
    return Prompt(
        PROMPT_PATH,
        evidence=evidence,
        claim=None if prop.name == "Authenticity" else claim.data,
        claim_date=claim.date_str or None,
        medium=medium,
        subject=subject,
        property=prop,
    )
