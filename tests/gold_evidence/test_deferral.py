"""Rate-limited sources are deferred, not recorded as inaccessible.

Marking a throttled source permanently inaccessible would inflate the
`inaccessible` rejection rate - a reported statistic - with what is only a
temporary condition, and would reject the whole claim on incomplete evidence.
"""

from datetime import datetime, timedelta

import pytest

from tests.gold_evidence.conftest import make_evidence, make_source
from veritas.gold_evidence import filtering as filtering_module
from veritas.gold_evidence.retrieval import SourceRetrieval

T_C = datetime(2024, 5, 1)
T_F = datetime(2024, 5, 21)


@pytest.fixture
def retrieval(monkeypatch):
    """Lets a test dictate what the retrieval of a source returns."""
    state = {"result": SourceRetrieval(accessible=False, rate_limited=True,
                                       error="slow down"),
             "saves": []}

    async def fake_retrieve(locator, determine_time=True):
        return state["result"]

    async def fake_save(self):
        state["saves"].append(self)

    monkeypatch.setattr(filtering_module, "retrieve_source", fake_retrieve)
    monkeypatch.setattr(type(make_evidence()), "save_to_db", fake_save)
    return state


async def filter_one(**kwargs):
    """Filters a single source and returns it together with its evidence item."""
    evidence = make_evidence(decided=False, **kwargs)
    source = evidence.sources[0]
    await filtering_module.filter_source(source, evidence=evidence, claim=None,
                                         t_c=T_C, t_f=T_F)
    return evidence, source


@pytest.mark.asyncio
async def test_rate_limited_source_is_deferred_not_rejected(retrieval):
    evidence, source = await filter_one()

    assert source.deferred is True
    assert source.deferred_until > datetime.now()
    # Deliberately left unjudged, so a later run evaluates it properly.
    assert source.admissible is None
    assert source.accessible is None
    assert source.filtered is False
    # The item waits with it: it is not decided while a source is still pending.
    assert evidence.deferred is True
    assert evidence.admissible is None


@pytest.mark.asyncio
async def test_deferral_window_is_configurable(retrieval, monkeypatch):
    monkeypatch.setattr(filtering_module, "defer_hours", 2)
    _, source = await filter_one()

    assert source.deferred_until <= datetime.now() + timedelta(hours=2, minutes=1)


@pytest.mark.asyncio
async def test_a_plain_failure_is_still_recorded_as_inaccessible(retrieval):
    retrieval["result"] = SourceRetrieval(accessible=False, error="404 not found")
    evidence, source = await filter_one()

    assert source.deferred is False
    assert source.accessible is False
    assert source.admissible is False
    assert source.inadmissibility_reason == "inaccessible"
    assert evidence.admissible is False  # its only source is gone


@pytest.mark.asyncio
async def test_an_expired_deferral_no_longer_counts(retrieval):
    source = make_source(filtered=False)
    source.deferred_until = datetime.now() - timedelta(hours=1)
    assert source.deferred is False


@pytest.mark.asyncio
async def test_a_deferred_source_does_not_hold_up_a_surviving_item(retrieval):
    """Only the deferred source waits; the item is still carried by the other one."""
    evidence = make_evidence(sources=[make_source(locator="https://a/1"),
                                      make_source(locator="https://a/2", filtered=False)])
    evidence.sources[1].defer(hours=24)

    assert evidence.deferred is True          # a source is still pending ...
    assert evidence.admissible is True        # ... but the proposition already holds


# --- Pipeline level -----------------------------------------------------------

@pytest.mark.asyncio
async def test_a_claim_with_deferred_evidence_is_not_rejected(monkeypatch):
    from veritas.common import Claim
    from veritas.gold_evidence import STATUS_DEFERRED
    from veritas.gold_evidence import pipeline as pipeline_module
    from tests.gold_evidence.conftest import make_gold_verdict

    class FakeDB:
        def __init__(self):
            self.status_writes = []

        async def get_evidence_for_claim(self, claim_id, admissible_only=False):
            return []

        async def get_verdict_rationales_for_claim(self, claim_id):
            return []

        async def set_gold_evidence_status(self, claim_id, status, reason=None):
            self.status_writes.append((status, reason))

        async def save_gold_evidence_result(self, result):
            raise AssertionError("Stage 3 must not run on an incomplete evidence set")

    fake_db = FakeDB()
    deferred_item = make_evidence(decided=False)
    deferred_item.sources[0].deferred_until = datetime.now() + timedelta(hours=24)

    async def fake_extract(claim, **kwargs):
        from veritas.gold_evidence.extraction import Extraction

        return Extraction(evidence=[deferred_item])

    async def fake_filter(claim, evidence, **kwargs):
        return []

    async def fake_times(claim):
        return T_C, T_F

    async def fake_verdict(self):
        return make_gold_verdict(veracity=-1.0)

    monkeypatch.setattr(pipeline_module, "db", fake_db)
    monkeypatch.setattr(pipeline_module, "extract_evidence", fake_extract)
    monkeypatch.setattr(pipeline_module, "filter_evidence", fake_filter)
    monkeypatch.setattr(pipeline_module, "get_reference_times", fake_times)
    monkeypatch.setattr(Claim, "current_verdict", property(fake_verdict))

    claim = Claim(id=1, data="A claim", date=T_C, appearance_ids=set(), review_ids={1})
    outcome = await pipeline_module.reconstruct_claim(claim)

    assert outcome.status == STATUS_DEFERRED
    assert outcome.deferred is True
    assert outcome.n_deferred == 1
    assert outcome.reason is None  # not a rejection
    assert fake_db.status_writes[-1] == (STATUS_DEFERRED, None)


def test_deferred_claims_are_resumable():
    from veritas.gold_evidence import RESUMABLE_STATUSES, STATUS_DEFERRED

    assert STATUS_DEFERRED in RESUMABLE_STATUSES
