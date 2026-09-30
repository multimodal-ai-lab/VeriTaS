"""The sufficiency validator always asks the ensemble.

Neither a lack of key evidence nor an empty condition decides a condition in
advance: whether what is there suffices to recover the gold verdict is exactly the
question the ensemble answers. The key items a condition lacks are recorded for the
analysis only.
"""

from datetime import datetime

import pytest

from tests.gold_evidence.conftest import make_evidence, make_gold_verdict, make_rationale
from veritas.common import Claim
from veritas.common.annotation import Rating
from veritas.common.annotation.rating import RatingAggregated
from veritas.gold_evidence import CONDITION_CLAIM, CONDITION_FACT_CHECK
from veritas.gold_evidence import sufficiency as sufficiency_module

CLAIM = Claim(id=1, data="A claim", date=datetime(2024, 5, 1),
              appearance_ids=set(), review_ids={1})


class StubEnsemble:
    model_names = ["stub:a", "stub:b", "stub:c"]


@pytest.fixture
def ensemble(monkeypatch):
    """Records what the ensemble was shown and answers "false, certain"."""
    calls = []

    async def fake_assess(property_name, claim, evidence, medium=None, rationales=()):
        calls.append({"property": property_name, "n_evidence": len(evidence),
                      "n_rationales": len(list(rationales))})
        ratings = [Rating(score=-1.0, rater=name, explanation="because")
                   for name in StubEnsemble.model_names]
        return RatingAggregated(individual_ratings=ratings, rater="ensemble"), []

    monkeypatch.setattr(sufficiency_module, "assess_property_from_evidence", fake_assess)
    monkeypatch.setattr(sufficiency_module, "get_ensemble", lambda: StubEnsemble())
    return calls


@pytest.mark.asyncio
async def test_a_condition_lacking_key_evidence_is_still_judged(ensemble):
    result = await sufficiency_module.validate_sufficiency(
        CLAIM, [], make_gold_verdict(veracity=-1.0), condition=CONDITION_CLAIM,
        mode="integrity", rationales=[make_rationale()], n_key_missing=2)

    assert len(ensemble) == 1
    assert result.is_close is True
    assert result.error is None
    assert result.n_key_missing == 2
    assert result.to_db_dict()["n_key_missing"] == 2


@pytest.mark.asyncio
async def test_the_rationale_is_shown_even_when_key_evidence_is_missing(ensemble):
    """It carries reasoning only, no evidence, so it cannot leak what is missing."""
    result = await sufficiency_module.validate_sufficiency(
        CLAIM, [make_evidence()], make_gold_verdict(veracity=-1.0),
        condition=CONDITION_CLAIM, mode="integrity",
        rationales=[make_rationale()], n_key_missing=1)

    assert ensemble[0]["n_rationales"] == 1
    assert result.with_rationale is True


@pytest.mark.asyncio
async def test_without_evidence_and_rationale_the_claim_is_judged_alone(ensemble):
    result = await sufficiency_module.validate_sufficiency(
        CLAIM, [], make_gold_verdict(veracity=-1.0), condition=CONDITION_FACT_CHECK,
        mode="integrity")

    assert ensemble == [{"property": "integrity", "n_evidence": 0, "n_rationales": 0}]
    assert result.is_close is True
    assert result.with_rationale is False


@pytest.mark.asyncio
async def test_a_wrong_prediction_is_not_close(ensemble):
    result = await sufficiency_module.validate_sufficiency(
        CLAIM, [], make_gold_verdict(veracity=1.0), condition=CONDITION_FACT_CHECK,
        mode="integrity")
    assert result.is_close is False
