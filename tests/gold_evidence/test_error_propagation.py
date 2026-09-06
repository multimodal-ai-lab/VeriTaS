"""Quota and rate-limit errors are run-level conditions, not verdicts on a claim.

If they are swallowed, an API outage silently rejects every claim it touches and
records that rejection in `claims.gold_evidence_status` - which is why every stage
must let them through.
"""

import pytest

from tests.gold_evidence.conftest import make_evidence
from veritas.gold_evidence import extraction as extraction_module
from veritas.gold_evidence import filtering as filtering_module
from veritas.gold_evidence import retrieval as retrieval_module
from veritas.gold_evidence import sufficiency as sufficiency_module
from veritas.models import QuotaExceededError, RateLimitError

FATAL = [QuotaExceededError, RateLimitError]


class ExplodingModel:
    specifier = "stub:model"

    def __init__(self, error):
        self.error = error

    async def generate(self, *a, **k):
        raise self.error


# --- Stage 2: faithfulness ---------------------------------------------------

@pytest.mark.parametrize("error_cls", FATAL)
@pytest.mark.asyncio
async def test_faithfulness_propagates_fatal_errors(monkeypatch, error_cls):
    monkeypatch.setattr(filtering_module, "_resolve_filtering_model",
                        lambda: ExplodingModel(error_cls("out of budget")))
    with pytest.raises(error_cls):
        await filtering_module.assess_faithfulness(make_evidence(), "source text")


@pytest.mark.asyncio
async def test_faithfulness_still_tolerates_ordinary_failures(monkeypatch):
    monkeypatch.setattr(filtering_module, "_resolve_filtering_model",
                        lambda: ExplodingModel(RuntimeError("bad request")))
    assert await filtering_module.assess_faithfulness(make_evidence(), "source") is None


# --- Stage 2: temporal validation --------------------------------------------

@pytest.mark.parametrize("error_cls", FATAL)
@pytest.mark.asyncio
async def test_temporal_validation_propagates_fatal_errors(monkeypatch, error_cls):
    from datetime import datetime

    from veritas.common import Claim

    monkeypatch.setattr(filtering_module, "_resolve_filtering_model",
                        lambda: ExplodingModel(error_cls("out of budget")))
    claim = Claim(id=1, data="A claim", date=datetime(2024, 5, 1),
                  appearance_ids=set(), review_ids={1})

    # Only items inside the studied interval reach the model at all.
    evidence = make_evidence(available_since=datetime(2024, 5, 10))
    with pytest.raises(error_cls):
        await filtering_module.validate_temporally(
            evidence, claim=claim, source_str="source",
            t_c=datetime(2024, 5, 1), t_f=datetime(2024, 5, 21))


# --- Stage 2: the item-level wrapper -----------------------------------------

@pytest.mark.parametrize("error_cls", FATAL)
@pytest.mark.asyncio
async def test_filter_single_propagates_fatal_errors(monkeypatch, error_cls):
    from datetime import datetime

    async def exploding_retrieve(*a, **k):
        raise error_cls("out of budget")

    monkeypatch.setattr(filtering_module, "retrieve_source", exploding_retrieve)
    evidence = make_evidence(filtered=False)

    with pytest.raises(error_cls):
        await filtering_module.filter_single(
            evidence, claim=None, t_c=datetime(2024, 5, 1), t_f=datetime(2024, 5, 21))

    # Nothing was recorded about the item, so a later run still evaluates it.
    assert evidence.admissible is None
    assert evidence.filtered is False


# --- Stage 1 ------------------------------------------------------------------

@pytest.mark.parametrize("error_cls", FATAL)
@pytest.mark.asyncio
async def test_extraction_propagates_fatal_errors(monkeypatch, error_cls):
    from datetime import datetime

    from veritas.common import Claim

    class FakeArticle:
        id = 3
        dismissed = False
        content = "An article mentioning https://a.example/1."

    class FakeReview:
        id = 7
        url = "https://factchecker.example/1"
        published = datetime(2024, 5, 21)
        raw_publisher_name = "Example FactCheck"

        @property
        async def publisher(self):
            return None

    monkeypatch.setattr(extraction_module, "_resolve_model",
                        lambda prompt: ExplodingModel(error_cls("out of budget")))
    claim = Claim(id=1, data="A claim", date=datetime(2024, 5, 1),
                  appearance_ids=set(), review_ids={7})

    with pytest.raises(error_cls):
        await extraction_module.extract_from_article(claim, FakeReview(), FakeArticle())


# --- Stage 3 ------------------------------------------------------------------

@pytest.mark.parametrize("error_cls", FATAL)
@pytest.mark.asyncio
async def test_sufficiency_propagates_fatal_errors(monkeypatch, error_cls):
    from datetime import datetime

    from tests.gold_evidence.conftest import make_gold_verdict
    from veritas.common import Claim
    from veritas.gold_evidence import CONDITION_FACT_CHECK

    async def exploding(*a, **k):
        raise error_cls("out of budget")

    monkeypatch.setattr(sufficiency_module, "assess_property_from_evidence", exploding)
    claim = Claim(id=1, data="A claim", date=datetime(2024, 5, 1),
                  appearance_ids=set(), review_ids={1})

    with pytest.raises(error_cls):
        await sufficiency_module.validate_sufficiency(
            claim, [make_evidence()], make_gold_verdict(veracity=-1.0),
            condition=CONDITION_FACT_CHECK, mode="integrity")


@pytest.mark.asyncio
async def test_sufficiency_still_records_ordinary_failures(monkeypatch):
    from datetime import datetime

    from tests.gold_evidence.conftest import make_gold_verdict
    from veritas.common import Claim
    from veritas.gold_evidence import CONDITION_FACT_CHECK

    async def exploding(*a, **k):
        raise RuntimeError("all models refused")

    monkeypatch.setattr(sufficiency_module, "assess_property_from_evidence", exploding)
    claim = Claim(id=1, data="A claim", date=datetime(2024, 5, 1),
                  appearance_ids=set(), review_ids={1})

    result = await sufficiency_module.validate_sufficiency(
        claim, [make_evidence()], make_gold_verdict(veracity=-1.0),
        condition=CONDITION_FACT_CHECK, mode="integrity")
    assert result.is_close is None
    assert "RuntimeError" in result.error


# --- Retrieval: scrapeMM quota becomes a fatal error --------------------------

@pytest.mark.asyncio
async def test_scrapemm_quota_becomes_a_fatal_error(monkeypatch):
    from scrapemm.common.exceptions import QuotaExceededError as ScrapeQuota

    async def exploding(*a, **k):
        raise ScrapeQuota("no credits left")

    monkeypatch.setattr(retrieval_module, "retrieve", exploding)
    with pytest.raises(QuotaExceededError):
        await retrieval_module.retrieve_source("https://example.org/a")


@pytest.mark.asyncio
async def test_scrapemm_rate_limit_stays_a_per_source_outcome(monkeypatch):
    from scrapemm.common.exceptions import RateLimitError as ScrapeRateLimit

    async def exploding(*a, **k):
        raise ScrapeRateLimit("slow down")

    monkeypatch.setattr(retrieval_module, "retrieve", exploding)
    result = await retrieval_module.retrieve_source("https://example.org/a")
    assert result.rate_limited is True
    assert result.accessible is False
