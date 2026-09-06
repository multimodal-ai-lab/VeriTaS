"""Orchestration logic of `reconstruct_claim`, with all I/O stubbed out.

The point of these tests is the contract that matters scientifically: the gold
verdict is never written, the claim is never dismissed, and acceptance is decided
by `E_factcheck` alone while `E_claim` is only recorded.
"""

from datetime import datetime

import pytest

from tests.gold_evidence.conftest import make_evidence, make_gold_verdict
from veritas.common import Claim
from veritas.gold_evidence import (
    CONDITION_CLAIM,
    CONDITION_FACT_CHECK,
    STATUS_ACCEPTED,
    STATUS_REJECTED,
)
from veritas.gold_evidence import pipeline as pipeline_module
from veritas.gold_evidence.admissibility import apply_admissibility
from veritas.gold_evidence.models import Faithfulness, TemporalValidation
from veritas.gold_evidence.pipeline import (
    REJECT_INSUFFICIENT_EVIDENCE,
    REJECT_NO_CLAIM_TIME,
    REJECT_NO_FACT_CHECK_TIME,
    REJECT_NO_ADMISSIBLE_EVIDENCE,
    REJECT_NO_EVIDENCE_EXTRACTED,
    REJECT_NO_GOLD_VERDICT,
    reconstruct_claim,
    summarize,
)
from veritas.gold_evidence.sufficiency import SufficiencyResult


class FakeDB:
    """Records every write so the tests can assert what was and was not touched."""

    def __init__(self, evidence=None):
        self.evidence = evidence or []
        self.status_writes = []
        self.saved_results = []

    async def get_evidence_for_claim(self, claim_id, admissible_only=False):
        return list(self.evidence)

    async def set_gold_evidence_status(self, claim_id, status, reason=None):
        self.status_writes.append((claim_id, status, reason))

    async def save_gold_evidence_result(self, result):
        self.saved_results.append(result)
        return len(self.saved_results)


def make_claim(claim_id: int = 1) -> Claim:
    return Claim(id=claim_id, data="A claim", date=datetime(2024, 5, 1),
                 appearance_ids=set(), review_ids={1})


@pytest.fixture
def wired(monkeypatch):
    """Installs the fakes and returns a small control surface for each test."""
    state = {
        "db": FakeDB(),
        "gold": make_gold_verdict(veracity=-1.0),
        "extracted": [],
        "extract_calls": [],
        "close": {CONDITION_CLAIM: True, CONDITION_FACT_CHECK: True},
        "sufficiency_calls": [],
        "t_c": datetime(2024, 5, 1),
        "t_f": datetime(2024, 5, 21),
    }

    monkeypatch.setattr(pipeline_module, "db", state["db"])

    async def fake_current_verdict(self):
        return state["gold"]

    monkeypatch.setattr(Claim, "current_verdict", property(fake_current_verdict))

    async def fake_extract(claim, **kwargs):
        state["extract_calls"].append(kwargs)
        state["db"].evidence = state["extracted"]
        return list(state["extracted"])

    async def fake_filter(claim, evidence):
        """Stands in for Stage 2: marks every item accessible, faithful and
        in-time unless the test already set those fields itself."""
        for item in evidence:
            if item.accessible is None:
                item.accessible = True
            if item.faithfulness is None:
                item.faithfulness = Faithfulness(assessment=1.0)
            if item.temporal_validation is None:
                item.temporal_validation = TemporalValidation(
                    before_fact_check=True, before_claim=True)
            apply_admissibility(item, undated_policy="permissive")
        return [item for item in evidence if item.admissible]

    async def fake_validate(claim, evidence, gold, *, condition, mode, threshold):
        state["sufficiency_calls"].append((condition, len(list(evidence))))
        return SufficiencyResult(
            claim_id=claim.id, condition=condition, ensemble_mode=mode,
            n_evidence=len(list(evidence)), threshold=threshold,
            is_close=state["close"][condition],
        )

    async def fake_reference_times(claim):
        return state["t_c"], state["t_f"]

    monkeypatch.setattr(pipeline_module, "extract_evidence", fake_extract)
    monkeypatch.setattr(pipeline_module, "filter_evidence", fake_filter)
    monkeypatch.setattr(pipeline_module, "validate_sufficiency", fake_validate)
    monkeypatch.setattr(pipeline_module, "get_reference_times", fake_reference_times)
    return state


# --- Happy path ------------------------------------------------------------

@pytest.mark.asyncio
async def test_accepted_when_e_factcheck_recovers_the_verdict(wired):
    wired["extracted"] = [make_evidence(filtered=False)]
    outcome = await reconstruct_claim(make_claim())

    assert outcome.status == STATUS_ACCEPTED
    assert outcome.reason is None
    assert outcome.n_candidates == 1
    assert outcome.n_admissible == 1
    assert outcome.recoverable(CONDITION_CLAIM) is True
    assert outcome.recoverable(CONDITION_FACT_CHECK) is True


@pytest.mark.asyncio
async def test_both_conditions_are_always_evaluated(wired):
    wired["extracted"] = [
        make_evidence(locator="https://a/1", filtered=False),
        make_evidence(locator="https://a/2", filtered=False),
    ]
    outcome = await reconstruct_claim(make_claim())

    conditions = [c for c, _ in wired["sufficiency_calls"]]
    assert set(conditions) == {CONDITION_CLAIM, CONDITION_FACT_CHECK}
    assert len(wired["db"].saved_results) == 2
    assert outcome.status == STATUS_ACCEPTED


@pytest.mark.asyncio
async def test_e_claim_is_never_larger_than_e_factcheck(wired, monkeypatch):
    wired["extracted"] = [
        make_evidence(locator=f"https://a/{i}", filtered=False) for i in range(3)
    ]

    async def fake_filter(claim, evidence):
        """Stage 2 marking the third item as post-claim but pre-fact-check."""
        for i, item in enumerate(evidence):
            item.accessible = True
            item.faithfulness = Faithfulness(assessment=1.0)
            item.temporal_validation = TemporalValidation(
                before_fact_check=True, before_claim=(i < 2))
            apply_admissibility(item, undated_policy="permissive")
        return evidence

    monkeypatch.setattr(pipeline_module, "filter_evidence", fake_filter)
    await reconstruct_claim(make_claim())

    sizes = dict(wired["sufficiency_calls"])
    assert sizes[CONDITION_CLAIM] == 2
    assert sizes[CONDITION_FACT_CHECK] == 3


# --- Rejection paths -------------------------------------------------------

@pytest.mark.asyncio
async def test_rejected_without_a_gold_verdict(wired):
    wired["gold"] = None
    outcome = await reconstruct_claim(make_claim())
    assert outcome.status == STATUS_REJECTED
    assert outcome.reason == REJECT_NO_GOLD_VERDICT


@pytest.mark.asyncio
async def test_rejected_without_a_claim_time(wired):
    """Without t_c, E_claim and E_factcheck collapse into the same set."""
    wired["t_c"] = None
    outcome = await reconstruct_claim(make_claim())
    assert outcome.status == STATUS_REJECTED
    assert outcome.reason == REJECT_NO_CLAIM_TIME
    assert outcome.n_candidates == 0  # rejected before Stage 1 spent anything


@pytest.mark.asyncio
async def test_rejected_without_a_fact_check_time(wired):
    """Without t_f there is no evidence cutoff at all."""
    wired["t_f"] = None
    outcome = await reconstruct_claim(make_claim())
    assert outcome.status == STATUS_REJECTED
    assert outcome.reason == REJECT_NO_FACT_CHECK_TIME
    assert wired["sufficiency_calls"] == []


@pytest.mark.asyncio
async def test_rejected_when_nothing_could_be_extracted(wired):
    wired["extracted"] = []
    outcome = await reconstruct_claim(make_claim())
    assert outcome.status == STATUS_REJECTED
    assert outcome.reason == REJECT_NO_EVIDENCE_EXTRACTED


@pytest.mark.asyncio
async def test_rejected_when_all_evidence_is_inadmissible(wired, monkeypatch):
    wired["extracted"] = [make_evidence(filtered=False)]

    async def fake_filter(claim, evidence):
        for item in evidence:
            item.accessible = False
            apply_admissibility(item)
        return []

    monkeypatch.setattr(pipeline_module, "filter_evidence", fake_filter)
    outcome = await reconstruct_claim(make_claim())
    assert outcome.status == STATUS_REJECTED
    assert outcome.reason == REJECT_NO_ADMISSIBLE_EVIDENCE
    assert outcome.n_admissible == 0
    assert wired["sufficiency_calls"] == []  # no ensemble calls were wasted


@pytest.mark.asyncio
async def test_rejected_when_e_factcheck_does_not_recover_the_verdict(wired):
    wired["extracted"] = [make_evidence(filtered=False)]
    wired["close"] = {CONDITION_CLAIM: True, CONDITION_FACT_CHECK: False}
    outcome = await reconstruct_claim(make_claim())

    assert outcome.status == STATUS_REJECTED
    assert outcome.reason == REJECT_INSUFFICIENT_EVIDENCE
    # E_claim's outcome is still recorded - it is the research variable.
    assert outcome.recoverable(CONDITION_CLAIM) is True


@pytest.mark.asyncio
async def test_e_claim_failing_alone_does_not_reject_the_instance(wired):
    """The interesting case: the fact-checking period was necessary."""
    wired["extracted"] = [make_evidence(filtered=False)]
    wired["close"] = {CONDITION_CLAIM: False, CONDITION_FACT_CHECK: True}
    outcome = await reconstruct_claim(make_claim())

    assert outcome.status == STATUS_ACCEPTED
    assert outcome.recoverable(CONDITION_CLAIM) is False
    assert outcome.recoverable(CONDITION_FACT_CHECK) is True


# --- Invariants ------------------------------------------------------------

@pytest.mark.asyncio
async def test_the_claim_is_never_dismissed(wired):
    wired["extracted"] = [make_evidence(filtered=False)]
    wired["close"] = {CONDITION_CLAIM: False, CONDITION_FACT_CHECK: False}
    claim = make_claim()
    await reconstruct_claim(claim)

    assert claim.dismissed is False
    assert claim.dismissed_reason is None
    # Only the additive gold-evidence status columns were written.
    assert all(status in (STATUS_ACCEPTED, STATUS_REJECTED, "extracted", "filtered")
               for _, status, _ in wired["db"].status_writes)


@pytest.mark.asyncio
async def test_stored_evidence_is_reused_without_re_extraction(wired, monkeypatch):
    stored = [make_evidence()]
    apply_admissibility(stored[0])
    wired["db"].evidence = stored

    called = []

    async def fake_extract(claim, **kwargs):
        called.append(claim.id)
        return []

    monkeypatch.setattr(pipeline_module, "extract_evidence", fake_extract)
    outcome = await reconstruct_claim(make_claim())

    assert called == []  # Stage 1 was skipped
    assert outcome.n_candidates == 1


@pytest.mark.asyncio
async def test_re_extraction_replaces_the_stored_evidence(wired):
    """Otherwise items the new run no longer produces survive as orphans."""
    wired["db"].evidence = [make_evidence()]
    wired["extracted"] = [make_evidence(filtered=False)]
    await reconstruct_claim(make_claim(), re_extract=True)

    assert wired["extract_calls"] == [{"replace": True}]


# --- Summary ---------------------------------------------------------------

def test_summarize_counts_statuses_and_reasons():
    from veritas.gold_evidence.pipeline import ClaimOutcome

    outcomes = [
        ClaimOutcome(claim_id=1, status=STATUS_ACCEPTED, n_candidates=5, n_admissible=4),
        ClaimOutcome(claim_id=2, status=STATUS_REJECTED,
                     reason=REJECT_INSUFFICIENT_EVIDENCE, n_candidates=3, n_admissible=2),
    ]
    summary = summarize(outcomes)
    assert summary["n_claims"] == 2
    assert summary["statuses"] == {STATUS_ACCEPTED: 1, STATUS_REJECTED: 1}
    assert summary["rejection_reasons"] == {REJECT_INSUFFICIENT_EVIDENCE: 1}
    assert summary["n_candidates"] == 8
    assert summary["n_admissible"] == 6
