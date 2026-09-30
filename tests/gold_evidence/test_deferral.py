"""Rate-limited sources are deferred, not recorded as inaccessible - and every
source is retrieved once per URL, however many citations of however many claims
point at it.

Marking a throttled source permanently inaccessible would inflate the
`inaccessible` rejection rate - a reported statistic - with what is only a
temporary condition, and would reject the whole claim on incomplete evidence.
"""

import asyncio
from datetime import datetime, timedelta

import pytest

from tests.gold_evidence.conftest import make_citation, make_evidence, make_source
from veritas.gold_evidence import filtering as filtering_module
from veritas.gold_evidence.models import Source
from veritas.gold_evidence.retrieval import SourceRetrieval

T_C = datetime(2024, 5, 1)
T_F = datetime(2024, 5, 21)


class FakeSourceDB:
    """The global `sources` table: stores what Stage 2 decided about each URL."""

    def __init__(self):
        self.rows: dict[str, Source] = {}

    async def get_source_by_locator(self, locator):
        from veritas.gold_evidence.models import locator_key

        stored = self.rows.get(locator_key(locator))
        return stored.model_copy() if stored else None

    async def update_source(self, source):
        if source.id is None:
            source.id = len(self.rows) + 1
        self.rows[source.key] = source.model_copy()


@pytest.fixture
def retrieval(monkeypatch):
    """Lets a test dictate what the retrieval of a source returns, and counts the
    retrievals."""
    state = {"result": SourceRetrieval(accessible=False, rate_limited=True,
                                       error="slow down"),
             "calls": [], "db": FakeSourceDB()}

    async def fake_retrieve(locator, determine_time=True):
        state["calls"].append(locator)
        await asyncio.sleep(0)  # give concurrent claims a chance to interleave
        return state["result"]

    async def fake_save(self):
        pass

    async def fake_times(claim):
        return T_C, T_F

    async def not_a_fact_checker(locator):
        return False

    async def faithful(proposition, source_str):
        from veritas.gold_evidence.models import Faithfulness

        return Faithfulness(assessment=1.0)

    monkeypatch.setattr(filtering_module, "retrieve_source", fake_retrieve)
    monkeypatch.setattr(filtering_module, "db", state["db"])
    monkeypatch.setattr(filtering_module, "get_reference_times", fake_times)
    monkeypatch.setattr(filtering_module, "_is_fact_checking_org", not_a_fact_checker)
    monkeypatch.setattr(filtering_module, "assess_faithfulness", faithful)
    monkeypatch.setattr(filtering_module, "_retrieved_anew", set())
    monkeypatch.setattr(type(make_evidence()), "save_to_db", fake_save)
    return state


def accessible(**overrides) -> SourceRetrieval:
    kwargs = dict(accessible=True, content="The page says X.",
                  available_since=datetime(2024, 4, 1), dating_method="meta")
    kwargs.update(overrides)
    return SourceRetrieval(**kwargs)


async def filter_one(**kwargs):
    """Filters a single-citation item and returns it together with its citation."""
    evidence = make_evidence(decided=False, **kwargs)
    await filtering_module.filter_evidence(None, [evidence])
    return evidence, evidence.citations[0]


@pytest.mark.asyncio
async def test_rate_limited_source_is_deferred_not_rejected(retrieval):
    evidence, citation = await filter_one()
    source = citation.source

    assert source.deferred is True
    assert source.deferred_until > datetime.now()
    # Deliberately left unjudged, so a later run evaluates it properly.
    assert source.accessible is None
    assert citation.admissible is None
    assert citation.filtered is False
    # The item waits with it: it is not decided while a source is still pending.
    assert evidence.deferred is True
    assert evidence.admissible is None


@pytest.mark.asyncio
async def test_deferral_window_is_configurable(retrieval, monkeypatch):
    monkeypatch.setattr(filtering_module, "defer_hours", 2)
    _, citation = await filter_one()

    assert citation.source.deferred_until <= datetime.now() + timedelta(hours=2, minutes=1)


@pytest.mark.asyncio
async def test_archive_today_gate_defers_the_source_like_a_rate_limit(retrieval):
    """Archive.today's access check is not inaccessibility either: it is
    deferred the same way a rate limit is (see `retrieval.SourceRetrieval.gated`
    and `scripts/retry_deferred_archive_today.py`)."""
    retrieval["result"] = SourceRetrieval(accessible=False, gated=True, error="captcha")
    evidence, citation = await filter_one()

    assert citation.source.deferred is True
    assert citation.admissible is None
    assert evidence.deferred is True
    assert evidence.admissible is None


@pytest.mark.asyncio
async def test_a_plain_failure_is_still_recorded_as_inaccessible(retrieval):
    retrieval["result"] = SourceRetrieval(accessible=False, error="404 not found")
    evidence, citation = await filter_one()

    assert citation.source.deferred is False
    assert citation.source.accessible is False
    assert citation.source.retrieval_error == "404 not found"
    assert citation.admissible is False
    assert citation.inadmissibility_reason == "inaccessible"
    assert evidence.admissible is False  # its only source is gone


@pytest.mark.asyncio
async def test_an_accessible_source_is_dated_and_judged(retrieval):
    retrieval["result"] = accessible()
    evidence, citation = await filter_one()

    assert citation.source.accessible is True
    assert citation.source.available_since == datetime(2024, 4, 1)
    assert citation.source.dating_method == "meta"
    assert citation.temporal_validation.before_claim is True
    assert citation.admissible is True
    assert evidence.admissible is True


@pytest.mark.asyncio
async def test_an_expired_deferral_no_longer_counts(retrieval):
    source = make_source(retrieved=False)
    source.deferred_until = datetime.now() - timedelta(hours=1)
    assert source.deferred is False


@pytest.mark.asyncio
async def test_a_deferred_source_does_not_hold_up_a_surviving_item(retrieval):
    """Only the deferred source waits; the item is still carried by the other one."""
    evidence = make_evidence(citations=[make_citation(locator="https://a/1"),
                                        make_citation(locator="https://a/2", filtered=False)])
    evidence.citations[1].source.defer(hours=24)

    assert evidence.deferred is True          # a source is still pending ...
    assert evidence.admissible is True        # ... but the proposition already holds


# --- One retrieval per URL -----------------------------------------------------

@pytest.mark.asyncio
async def test_a_url_cited_by_several_items_is_retrieved_once(retrieval):
    from veritas.gold_evidence.extraction import share_sources

    retrieval["result"] = accessible()
    items = share_sources([
        make_evidence(proposition="p1", decided=False, locator="https://a/1"),
        make_evidence(proposition="p2", decided=False, locator="https://a/1/"),
        make_evidence(proposition="p3", decided=False, locator="https://a/2"),
    ])
    await filtering_module.filter_evidence(None, items)

    assert sorted(retrieval["calls"]) == ["https://a/1", "https://a/2"]
    # ... while faithfulness was judged for every citation.
    assert all(item.citations[0].faithfulness is not None for item in items)
    assert all(item.admissible for item in items)


@pytest.mark.asyncio
async def test_a_url_retrieved_for_one_claim_is_reused_by_the_next(retrieval):
    """Sources are global: another claim citing the same URL adopts the stored
    retrieval instead of fetching the page again."""
    retrieval["result"] = accessible()
    first = make_evidence(decided=False, locator="https://a/1")
    await filtering_module.filter_evidence(None, [first])

    second = make_evidence(decided=False, locator="https://a/1", claim_id=2)
    await filtering_module.filter_evidence(None, [second])

    assert retrieval["calls"] == ["https://a/1"]
    assert second.citations[0].source.available_since == datetime(2024, 4, 1)
    assert second.admissible is True


@pytest.mark.asyncio
async def test_concurrent_claims_citing_one_url_retrieve_it_once(retrieval):
    retrieval["result"] = accessible()
    claims = [[make_evidence(decided=False, locator="https://a/1", claim_id=i)]
              for i in range(4)]
    await asyncio.gather(*(filtering_module.filter_evidence(None, items) for items in claims))

    assert retrieval["calls"] == ["https://a/1"]
    assert all(items[0].admissible for items in claims)


@pytest.mark.asyncio
async def test_re_retrieve_fetches_a_stored_source_once_more(retrieval):
    retrieval["result"] = accessible()
    await filtering_module.filter_evidence(
        None, [make_evidence(decided=False, locator="https://a/1")])

    retrieval["result"] = accessible(available_since=datetime(2024, 3, 1))
    again = [make_evidence(decided=False, locator="https://a/1", claim_id=c) for c in (2, 3)]
    for items in again:
        await filtering_module.filter_evidence(None, [items], re_retrieve=True)

    # Once for the original run, once for the re-retrieval - shared by both claims.
    assert retrieval["calls"] == ["https://a/1", "https://a/1"]
    assert all(item.citations[0].source.available_since == datetime(2024, 3, 1)
               for item in again)


@pytest.mark.asyncio
async def test_a_tool_citation_does_not_trigger_a_retrieval(retrieval):
    from veritas.gold_evidence.models import SourceKind

    evidence = make_evidence(decided=False, locator="https://geotool.example.com",
                             kind=SourceKind.TOOL)
    await filtering_module.filter_evidence(None, [evidence])

    assert retrieval["calls"] == []
    assert evidence.admissible is True


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
    deferred_item.citations[0].source.deferred_until = datetime.now() + timedelta(hours=24)

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


# --- Manual retry (Archive.today) ---------------------------------------------

@pytest.mark.asyncio
async def test_retry_clears_archive_today_source_deferral_and_reconstructs(monkeypatch):
    """`retry_deferred_archive_today_sources` clears the deferral itself - the
    normal periodic run would otherwise wait out `defer_hours` before noticing
    that the source can be retried. The source is global, so it is cleared once
    and every claim citing it is reconstructed."""
    from veritas.common import Claim
    from veritas.gold_evidence import pipeline as pipeline_module

    source = make_source(locator="https://archive.ph/abcde", retrieved=False)
    source.id = 3
    source.deferred_until = datetime.now() + timedelta(hours=20)
    claims = [Claim(id=i, data="A claim", date=T_C, appearance_ids=set(), review_ids={1})
              for i in (1, 2)]

    class FakeDB:
        saved = []

        async def get_deferred_archive_today_sources(self):
            return [source]

        async def update_source(self, s):
            self.saved.append((s.id, s.deferred_until))

        async def get_claim_ids_citing_sources(self, source_ids):
            assert source_ids == [3]
            return [1, 2]

        async def get_claims_by_ids(self, claim_ids):
            assert claim_ids == [1, 2]
            return claims

    reconstructed = []

    async def fake_reconstruct(claims, **kwargs):
        reconstructed.extend(claims)
        return ["outcome"]

    fake_db = FakeDB()
    monkeypatch.setattr(pipeline_module, "db", fake_db)
    monkeypatch.setattr(pipeline_module, "reconstruct_claims", fake_reconstruct)

    outcomes = await pipeline_module.retry_deferred_archive_today_sources(mode="integrity")

    assert source.deferred_until is None
    assert fake_db.saved == [(3, None)]
    assert reconstructed == claims
    assert outcomes == ["outcome"]


@pytest.mark.asyncio
async def test_retry_without_gated_sources_does_nothing(monkeypatch):
    from veritas.gold_evidence import pipeline as pipeline_module

    class FakeDB:
        async def get_deferred_archive_today_sources(self):
            return []

    async def fake_reconstruct(claims, **kwargs):
        raise AssertionError("nothing to reconstruct")

    monkeypatch.setattr(pipeline_module, "db", FakeDB())
    monkeypatch.setattr(pipeline_module, "reconstruct_claims", fake_reconstruct)

    assert await pipeline_module.retry_deferred_archive_today_sources() == []
