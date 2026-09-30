"""A scrapeMM server that cannot be reached is a run-level condition (like an
exhausted quota), not a reason to register a degraded publisher, dismiss a
review, or fail an article scrape permanently. `ServerError` is converted to
`veritas.models.QuotaExceededError` at every `retrieve()` call site in the
main pipeline so `Stage.run()` aborts the stage cleanly (see
`veritas/pipeline/util/stage.py`) instead of recording a false failure.
"""

import pytest
from scrapemm.common.exceptions import ServerError

from veritas.common import Review
from veritas.models import QuotaExceededError
from veritas.pipeline import stage_2 as stage_2_module
from veritas.pipeline import stage_3 as stage_3_module


def make_review(*, id: int = 1) -> Review:
    review = Review(url="https://factchecker.example/article", raw_claim="Some claim",
                    raw_publisher_url="https://publisher.example")
    review.id = id
    review.stage = 1
    return review


# --- Stage 2: publisher identification ------------------------------------------

@pytest.mark.asyncio
async def test_register_new_publisher_propagates_on_server_error(monkeypatch):
    async def fake_retrieve(url, **kwargs):
        raise ServerError("Could not reach the scrapeMM server at http://localhost:8080: timeout")

    monkeypatch.setattr(stage_2_module, "retrieve", fake_retrieve)

    with pytest.raises(QuotaExceededError):
        await stage_2_module.register_new_publisher("publisher.example")


@pytest.mark.asyncio
async def test_identify_publishers_does_not_dismiss_on_server_error(monkeypatch):
    """`review.publisher_id` is unset, so `register_new_publisher` runs and
    (via the fake below) reports the outage; the review must be left alone,
    not dismissed as 'could not identify publisher'."""
    review = make_review()
    saved = []

    async def fake_save(self):
        saved.append(self)

    class FakeDB:
        async def get_publisher_by_url(self, domain):
            return None

    async def fake_register(domain, name=None):
        raise QuotaExceededError("scrapeMM server unreachable: boom")

    monkeypatch.setattr(stage_2_module, "db", FakeDB())
    monkeypatch.setattr(Review, "save_to_db", fake_save)
    monkeypatch.setattr(stage_2_module, "register_new_publisher", fake_register)

    with pytest.raises(QuotaExceededError):
        await stage_2_module.identify_publishers([review])

    assert review.dismissed is False
    assert saved == []


# --- Stage 3: article scraping ---------------------------------------------------

@pytest.mark.asyncio
async def test_process_articles_propagates_on_server_error(monkeypatch):
    async def fake_retrieve(urls, **kwargs):
        raise ServerError("Could not reach the scrapeMM server at http://localhost:8080: timeout")

    async def no_article(self):
        return False

    monkeypatch.setattr(stage_3_module, "retrieve", fake_retrieve)
    monkeypatch.setattr(Review, "has_article", no_article)

    with pytest.raises(QuotaExceededError):
        await stage_3_module.process_articles([make_review()])
