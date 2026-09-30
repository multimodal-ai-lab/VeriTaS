"""Archive.today's access check is a deferral, not a permanent failure.

Like a rate limit, but resolved only once a human passes the CAPTCHA - possibly
much later than any fixed cooldown. scrapeMM buffers the gated URL and answers
it from its persistent page cache once that happens, so retrying later (or on
demand via `scripts/retry_deferred_archive_today.py`) succeeds without
retrieving anything twice.

A scrapeMM server that cannot be reached at all is a different, more severe
condition: it says nothing about any particular appearance, so it must abort
the stage (via `veritas.models.QuotaExceededError`) rather than being recorded
as a per-appearance failure - see the "ServerError" tests below.
"""

import asyncio
from datetime import datetime, timedelta

import pytest
from scrapemm.common import ScrapingResponse
from scrapemm.common.exceptions import CaptchaEncounteredError, RateLimitError, ServerError

from veritas.common import Appearance, Review
from veritas.models import QuotaExceededError
from veritas.pipeline import stage_4 as stage_4_module


def failed_response(error: Exception, url: str = "https://example.org/a",
                    method: str = "firecrawl") -> ScrapingResponse:
    return ScrapingResponse(url=url, content=None, errors={method: error}, method=method)


def make_appearance(*, url: str | None = None, archive_url: str | None = None,
                    id: int = 1) -> Appearance:
    app = Appearance(url=url, archive_url=archive_url)
    app.id = id
    return app


def make_review(*, id: int = 1, stage: int = 3) -> Review:
    review = Review(url="https://factchecker.example/article", raw_claim="Some claim")
    review.id = id
    review.stage = stage
    return review


@pytest.fixture(autouse=True)
def no_op_save(monkeypatch):
    """Every model here saves through the generic dispatcher, which needs a DB;
    these tests are pure, so saving is just recorded instead."""
    saved = []

    async def fake_save(self):
        saved.append(self)

    monkeypatch.setattr(Appearance, "save_to_db", fake_save)
    monkeypatch.setattr(Review, "save_to_db", fake_save)
    return saved


@pytest.fixture(autouse=True)
def no_review_cascade(monkeypatch):
    """`Appearance.defer()` cascades to the reviews already linking it, looked
    up via the DB (`self.reviews`); these tests have none linked yet."""
    async def no_reviews(self):
        return []

    monkeypatch.setattr(Appearance, "reviews", property(no_reviews))


# --- _is_deferrable: which errors are worth waiting out ------------------------

def test_is_deferrable_helper():
    is_deferrable = stage_4_module._is_deferrable

    assert is_deferrable(RateLimitError("slow down"), "https://example.org/a") is True
    assert is_deferrable(CaptchaEncounteredError("captcha"), "https://archive.ph/a") is True
    assert is_deferrable(CaptchaEncounteredError("captcha"), "https://example.org/a") is False
    assert is_deferrable(Exception("boom"), "https://archive.ph/a") is False


# --- scrape_appearances: which errors get deferred -----------------------------

@pytest.mark.asyncio
async def test_archive_today_captcha_defers_the_appearance(monkeypatch):
    app = make_appearance(url="https://archive.ph/abcde")

    async def fake_retrieve(url, **kwargs):
        return failed_response(CaptchaEncounteredError("captcha"), url=url)

    monkeypatch.setattr(stage_4_module, "retrieve", fake_retrieve)
    await stage_4_module.scrape_appearances([app])

    assert app.deferred is True
    assert app.deferred_until > datetime.now()
    assert app.dismissed is False


@pytest.mark.asyncio
async def test_archive_today_captcha_on_the_archive_url_defers_too(monkeypatch):
    app = make_appearance(url="https://example.org/gone", archive_url="https://archive.ph/abcde")

    async def fake_retrieve(url, **kwargs):
        if "archive.ph" in url:
            return failed_response(CaptchaEncounteredError("captcha"), url=url)
        return failed_response(Exception("404"), url=url)

    monkeypatch.setattr(stage_4_module, "retrieve", fake_retrieve)
    await stage_4_module.scrape_appearances([app])

    assert app.deferred is True
    assert app.dismissed is False


@pytest.mark.asyncio
async def test_a_captcha_on_a_non_archive_today_domain_is_still_a_failure(monkeypatch):
    """Only Archive.today's access check is deferred: it is the one gate
    scrapeMM buffers and later answers from a persistent cache. A CAPTCHA
    anywhere else is recorded as a plain, permanent failure, as before."""
    app = make_appearance(url="https://example.org/gated")

    async def fake_retrieve(url, **kwargs):
        return failed_response(CaptchaEncounteredError("captcha"), url=url)

    monkeypatch.setattr(stage_4_module, "retrieve", fake_retrieve)
    await stage_4_module.scrape_appearances([app])

    assert app.deferred is False
    assert app.dismissed is True


@pytest.mark.asyncio
async def test_rate_limit_still_defers_as_before(monkeypatch):
    app = make_appearance(url="https://example.org/a")

    async def fake_retrieve(url, **kwargs):
        return failed_response(RateLimitError("slow down"), url=url)

    monkeypatch.setattr(stage_4_module, "retrieve", fake_retrieve)
    await stage_4_module.scrape_appearances([app])

    assert app.deferred is True
    assert app.dismissed is False


# --- process_appearances_single_review: the review must not be dismissed -------

@pytest.mark.asyncio
async def test_review_with_only_a_deferred_appearance_is_not_dismissed(monkeypatch):
    app = make_appearance(url="https://archive.ph/abcde")
    app.deferred_until = datetime.now() + timedelta(hours=24)
    review = make_review()

    async def fake_extract(r):
        return [app]

    async def fake_scrape(apps):
        pass  # already deferred; nothing to do

    monkeypatch.setattr(stage_4_module, "extract_appearances", fake_extract)
    monkeypatch.setattr(stage_4_module, "scrape_appearances", fake_scrape)

    await stage_4_module.process_appearances_single_review(review)

    assert review.dismissed is False
    assert review.deferred is True
    assert review.deferred_until == app.deferred_until
    assert review.stage == 3  # unchanged: not yet decided


@pytest.mark.asyncio
async def test_review_is_dismissed_once_every_appearance_permanently_failed(monkeypatch):
    app = make_appearance(url="https://example.org/gone")
    app.dismissed = True
    review = make_review()

    async def fake_extract(r):
        return [app]

    async def fake_scrape(apps):
        pass

    monkeypatch.setattr(stage_4_module, "extract_appearances", fake_extract)
    monkeypatch.setattr(stage_4_module, "scrape_appearances", fake_scrape)

    await stage_4_module.process_appearances_single_review(review)

    assert review.dismissed is True
    assert review.deferred is False


@pytest.mark.asyncio
async def test_review_is_promoted_once_an_appearance_succeeds(monkeypatch):
    app = make_appearance(url="https://example.org/ok")
    app.original_scrape_ok = True
    review = make_review()

    async def fake_extract(r):
        return [app]

    async def fake_scrape(apps):
        pass

    monkeypatch.setattr(stage_4_module, "extract_appearances", fake_extract)
    monkeypatch.setattr(stage_4_module, "scrape_appearances", fake_scrape)

    await stage_4_module.process_appearances_single_review(review)

    assert review.stage == 4
    assert review.dismissed is False
    assert review.deferred is False


# --- retry_deferred_archive_today_appearances -----------------------------------

@pytest.mark.asyncio
async def test_retry_clears_appearance_and_review_deferral_and_reprocesses(monkeypatch):
    """The manual retry (`scripts/retry_deferred_archive_today.py`) clears the
    deferral itself - the normal periodic scan would otherwise wait out
    `RETRY_DEFER_HOURS` before noticing the appearance can be retried."""
    import veritas.db as db_module

    app = make_appearance(url="https://archive.ph/abcde")
    app.deferred_until = datetime.now() + timedelta(hours=20)
    review = make_review()
    review.deferred_until = datetime.now() + timedelta(hours=20)

    processed = []

    async def fake_process(r):
        processed.append(r)

    async def fake_appearances(self):
        return [app]

    class FakeDB:
        async def get_reviews_with_deferred_archive_today_appearances(self):
            return [review]

    monkeypatch.setattr(db_module, "db", FakeDB())
    monkeypatch.setattr(Review, "appearances", property(fake_appearances))
    monkeypatch.setattr(stage_4_module, "process_appearances_single_review", fake_process)

    n = await stage_4_module.retry_deferred_archive_today_appearances()

    assert n == 1
    assert app.deferred_until is None
    assert review.deferred_until is None
    assert processed == [review]


@pytest.mark.asyncio
async def test_retry_leaves_an_unrelated_deferred_appearance_alone(monkeypatch):
    """A rate-limited (non-Archive.today) appearance in the same review is left
    waiting out its own cooldown."""
    import veritas.db as db_module

    gated = make_appearance(url="https://archive.ph/abcde", id=1)
    gated.deferred_until = datetime.now() + timedelta(hours=20)
    rate_limited = make_appearance(url="https://other.example/x", id=2)
    rate_limited.deferred_until = datetime.now() + timedelta(hours=20)
    review = make_review()

    async def fake_process(r):
        pass

    async def fake_appearances(self):
        return [gated, rate_limited]

    class FakeDB:
        async def get_reviews_with_deferred_archive_today_appearances(self):
            return [review]

    monkeypatch.setattr(db_module, "db", FakeDB())
    monkeypatch.setattr(Review, "appearances", property(fake_appearances))
    monkeypatch.setattr(stage_4_module, "process_appearances_single_review", fake_process)

    await stage_4_module.retry_deferred_archive_today_appearances()

    assert gated.deferred_until is None
    assert rate_limited.deferred_until is not None  # untouched


# --- ServerError: an unreachable scrapeMM server aborts, never dismisses -------

@pytest.mark.asyncio
async def test_server_unreachable_propagates_and_does_not_dismiss(monkeypatch):
    app = make_appearance(url="https://example.org/a")

    async def fake_retrieve(url, **kwargs):
        raise ServerError("Could not reach the scrapeMM server at http://localhost:8080: boom")

    monkeypatch.setattr(stage_4_module, "retrieve", fake_retrieve)

    with pytest.raises(QuotaExceededError):
        await stage_4_module.scrape_appearances([app])

    assert app.dismissed is False
    assert app.deferred is False


@pytest.mark.asyncio
async def test_server_unreachable_on_the_archive_url_also_propagates(monkeypatch):
    app = make_appearance(url="https://example.org/gone", archive_url="https://example.org/archived")

    async def fake_retrieve(url, **kwargs):
        if url == "https://example.org/gone":
            return failed_response(Exception("404"), url=url)
        raise ServerError("Could not reach the scrapeMM server at http://localhost:8080: boom")

    monkeypatch.setattr(stage_4_module, "retrieve", fake_retrieve)

    with pytest.raises(QuotaExceededError):
        await stage_4_module.scrape_appearances([app])

    assert app.dismissed is False


@pytest.mark.asyncio
async def test_extract_appearances_from_article_lets_the_fatal_error_through(monkeypatch):
    """A bare `except Exception: pass` around each candidate URL would
    otherwise turn a scrapeMM outage into "this review has zero appearances",
    dismissing the review outright."""
    async def fake_extract_urls(review):
        return ["https://example.org/a", "https://example.org/b"]

    async def fake_appearance_from_url(url):
        raise QuotaExceededError("scrapeMM server unreachable: boom")

    monkeypatch.setattr(stage_4_module, "extract_appearance_urls", fake_extract_urls)
    monkeypatch.setattr(stage_4_module, "appearance_from_url", fake_appearance_from_url)

    with pytest.raises(QuotaExceededError):
        await stage_4_module.extract_appearances_from_article(make_review())


# --- Worker pool: a fatal error must not deadlock queue.join() -----------------

@pytest.mark.asyncio
async def test_worker_stops_processing_after_a_fatal_error(monkeypatch):
    """With a single worker, everything queued after the fatal review is
    skipped deterministically instead of being attempted (and failing the
    same way) or hanging - proving the drain-without-processing path works."""
    reviews = [make_review(id=i) for i in range(1, 6)]
    processed = []

    async def fake_process(review):
        if review.id == 2:
            raise QuotaExceededError("scrapeMM server unreachable: boom")
        processed.append(review.id)

    monkeypatch.setattr(stage_4_module, "process_appearances_single_review", fake_process)

    stage = stage_4_module.Stage4()
    stage.queue = asyncio.Queue()
    stage._fatal_error = None
    stage._workers = [asyncio.create_task(stage_4_module.worker(0, stage.queue, stage))]
    try:
        await asyncio.wait_for(stage_4_module.process_appearances(reviews, stage.queue), timeout=5)
    finally:
        for w in stage._workers:
            w.cancel()
        await asyncio.gather(*stage._workers, return_exceptions=True)

    assert processed == [1]  # review 2 raised; 3, 4, 5 were left untouched
    assert isinstance(stage._fatal_error, QuotaExceededError)


@pytest.mark.asyncio
async def test_worker_pool_does_not_deadlock_with_several_workers(monkeypatch):
    """Regression guard: every enqueued item must get `task_done()` called
    exactly once, whether processed, skipped, or the one that raised -
    otherwise `queue.join()` (awaited by `process_appearances`) hangs forever
    once every worker has hit the fatal condition."""
    reviews = [make_review(id=i) for i in range(1, 21)]
    processed = []

    async def fake_process(review):
        if review.id == 5:
            raise QuotaExceededError("scrapeMM server unreachable: boom")
        await asyncio.sleep(0)
        processed.append(review.id)

    monkeypatch.setattr(stage_4_module, "process_appearances_single_review", fake_process)

    stage = stage_4_module.Stage4()
    stage.queue = asyncio.Queue()
    stage._fatal_error = None
    stage._workers = [
        asyncio.create_task(stage_4_module.worker(i, stage.queue, stage)) for i in range(5)
    ]
    try:
        # Fails with a TimeoutError (not a hang) if draining is broken.
        await asyncio.wait_for(stage_4_module.process_appearances(reviews, stage.queue), timeout=5)
    finally:
        for w in stage._workers:
            w.cancel()
        await asyncio.gather(*stage._workers, return_exceptions=True)

    assert isinstance(stage._fatal_error, QuotaExceededError)
    assert 5 not in processed
    assert len(processed) < len(reviews)  # at least review 5 never completed


@pytest.mark.asyncio
async def test_step_reraises_the_fatal_error_and_clears_it(monkeypatch):
    stage = stage_4_module.Stage4()
    stage.queue = asyncio.Queue()
    stage._fatal_error = QuotaExceededError("scrapeMM server unreachable: boom")

    async def fake_get_queued_items(**kwargs):
        return []

    monkeypatch.setattr(stage, "get_queued_items", fake_get_queued_items)

    with pytest.raises(QuotaExceededError):
        await stage.step()

    assert stage._fatal_error is None
