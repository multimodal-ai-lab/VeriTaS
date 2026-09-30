import asyncio
import logging
import traceback

from ezmm.common import item_registry
from pydantic import BaseModel, Field, ValidationError
from scrapemm import retrieve
from scrapemm.common import ScrapingResponse
from scrapemm.common.exceptions import CaptchaEncounteredError, RateLimitError, ServerError

from veritas.common import Appearance, Prompt, Review
from veritas.common.appearance import appearance_from_url, is_sufficient_content
from veritas.models import QuotaExceededError, gpt_cheap
from veritas.pipeline import max_video_size, max_appearances_per_claim
from veritas.pipeline.util.stage import Stage
from veritas.util import is_archive_today_url

logger = logging.getLogger("VeriTaS")

#: How long a deferred appearance/review waits before being retried. Reused both
#: for a plain rate limit and for Archive.today's access check - the latter can
#: in fact take much longer to clear (it waits on a human), but the periodic
#: retry is cheap and `scripts/retry_deferred_archive_today.py` retries on
#: demand right after the check has been passed, without waiting this out.
RETRY_DEFER_HOURS = 24


class Stage4(Stage):
    """Extracts and scrapes appearances for reviews at stage 3.

    Owns a fixed-size pool of worker tasks reading from an `asyncio.Queue`,
    so per-review scraping runs concurrently across the whole loop, not just
    within a single batch.
    """

    id = 4
    name = "Appearance Retrieval"
    num_workers = 40

    async def setup(self, **kwargs) -> None:
        await super().setup()
        self.queue: asyncio.Queue[Review] = asyncio.Queue()
        #: Set by a worker that hits a run-level condition (e.g. scrapeMM
        #: unreachable), so `step()` can re-raise it once the batch has drained.
        self._fatal_error: Exception | None = None
        self._workers = [
            asyncio.create_task(worker(i, self.queue, self))
            for i in range(self.num_workers)
        ]

    async def step(self, *,
                   start: tuple[int, int] = (2016, 1),
                   end: tuple[int, int] | None = None,
                   batch_size: int = 100) -> None:
        reviews = await self.get_queued_items(start=start, end=end, batch_size=batch_size)
        if reviews:
            await process_appearances(reviews, self.queue)
        if self._fatal_error is not None:
            error, self._fatal_error = self._fatal_error, None
            raise error

    async def get_queued_items(self, *,
                               start: tuple[int, int],
                               end: tuple[int, int] | None,
                               batch_size: int) -> list[Review]:
        """Fetches a batch of reviews at stage 3 within the given date range."""
        from veritas.util.util import get_quarter_date_range
        from veritas.db import db
        start_date, _ignored = get_quarter_date_range(*start)
        end_date = None
        if end:
            _ignored, end_date = get_quarter_date_range(*end)
        return await db.get_reviews(stage=3, limit=batch_size,
                                    start_date=start_date, end_date=end_date)

    async def teardown(self) -> None:
        for w in getattr(self, "_workers", ()):
            w.cancel()
        if getattr(self, "_workers", None):
            await asyncio.gather(*self._workers, return_exceptions=True)


async def worker(worker_id: int, queue: asyncio.Queue, stage: Stage4):
    while True:
        review: Review = await queue.get()
        if stage._fatal_error is not None:
            # A fatal error already stopped this batch: drain without processing,
            # so `queue.join()` returns instead of waiting on work that will
            # never happen. The review is left untouched for a later run.
            queue.task_done()
            continue
        try:
            await process_appearances_single_review(review)
        except QuotaExceededError as e:
            # Run-level condition (e.g. scrapeMM unreachable): every other
            # in-flight retrieval would fail the same way. Record it instead of
            # letting it vanish - a worker crashing silently would eventually
            # starve `queue.join()` of any worker left to drain the backlog.
            stage._fatal_error = e
            logger.error(f"Worker {worker_id} hit a fatal error: {e}")
        except Exception as e:
            logger.debug(f"Worker {worker_id} failed processing appearances: {e}", exc_info=True)
        finally:
            queue.task_done()


async def process_appearances(reviews: list[Review], queue: asyncio.Queue):
    """Scrapes each appearance occurring in the review or
    in the *already scraped* article, saving them into the DB."""
    logger.info(f"Queueing {len(reviews)} reviews for appearance processing...")
    for review in reviews:
        queue.put_nowait(review)
    await queue.join()

    # Free up memory
    item_registry.cache.clear()

    logger.info(f"Appearance queue successfully completed.")


async def process_appearances_single_review(review: Review):
    # Use LLM to extract appearances from the scraped article
    appearances = await extract_appearances(review)

    if appearances:
        # Scrape all appearances that haven't been scraped yet
        await scrape_appearances(appearances)
        # Assign appearances to review
        review.appearance_ids = {app.id for app in appearances}

    # Upgrade stage tracker
    if any(not app.dismissed and app.scrape_ok for app in appearances):
        await review.set_stage(4)
        return

    # Nothing succeeded (yet). An appearance still waiting out a rate limit or
    # Archive.today's access check says nothing about the review, so it is left
    # pending - not dismissed - until the appearance is retried and decided.
    still_deferred = [app for app in appearances if app.deferred]
    if still_deferred:
        await review.defer(until=max(app.deferred_until for app in still_deferred))
    else:
        await review.dismiss("Could not retrieve any appearances.")


async def extract_appearances(review: Review) -> list[Appearance]:
    """Returns the appearances occurring in the review. If no appearances
    were provided in the ClaimReview meta, attempts to extract them from the
    article directly."""
    # Get appearances (either from review markup or from web article directly)
    if not (appearances := await review.appearances):
        appearances = await extract_appearances_from_article(review)
    return appearances


def _is_deferrable(error: Exception, url: str) -> bool:
    """Whether the error is worth waiting out rather than treating as a permanent
    failure: a rate limit (any domain), or Archive.today's access check.

    The latter is not a plain rate limit: scrapeMM does not solve it, and it is
    resolved only once a human passes it - possibly hours later, well past any
    fixed cooldown. scrapeMM buffers the gated URL and answers it from its
    persistent page cache once that happens, so retrying later (here, or right
    away via `scripts/retry_deferred_archive_today.py`) succeeds without
    retrieving anything twice."""
    if isinstance(error, RateLimitError):
        return True
    return isinstance(error, CaptchaEncounteredError) and is_archive_today_url(url)


async def scrape_appearances(appearances: list[Appearance]):
    """Downloads the contents from the linked appearances and saves them to the DB.
    Tries the original URL first (if available). If that does not yield sufficient content,
    falls back to the archived URL (if available). Sets success flags independently.
    Downloads only as many appearances as needed.
    """
    # Only take appearances that are not deferred and are missing original content (regardless of archive)
    # Implicitly ignores the dismissed status
    unscraped_appearances = [
        app for app in appearances
        if not app.original_scraped_content and not app.deferred
    ]
    if not unscraped_appearances:
        return

    logger.debug(f"Scraping {len(unscraped_appearances)} appearance(s)...")

    n_successes = 0
    for app in unscraped_appearances:
        # Try original URL first if available
        last_error: str | None = None
        try:
            if app.url and not app.original_scrape_ok:
                url = prepare_url(str(app.url))
                response = await retrieve(url, show_progress=False, max_video_size=max_video_size)
                if isinstance(response, ScrapingResponse):
                    app.scrape_method = response.method
                    if response.success:
                        app.original_scraped_content = response.content.multimodal
                        if is_sufficient_content(app.original_scraped_content):
                            app.original_scrape_ok = True
                            await app.save_to_db()
                            n_successes += 1
                            continue  # success, no need to try archive
                    elif response.errors:
                        error = list(response.errors.values())[0]
                        if _is_deferrable(error, url):
                            await app.defer(hours=RETRY_DEFER_HOURS)
                            await app.save_to_db()
                            # skip trying archive when deferred
                            continue
                        last_error = str(error)
                        # Not sufficient -> try archive below if available
                else:
                    last_error = "scrapeMM did not return a ScrapingResponse."

            # Try archived URL only if original is absent or insufficient
            if app.archive_url and not app.archived_scrape_ok:
                url = prepare_url(str(app.archive_url))
                response = await retrieve(url, show_progress=False, max_video_size=max_video_size)
                if isinstance(response, ScrapingResponse):
                    app.scrape_method = response.method
                    if response.success:
                        app.archived_scraped_content = response.content.multimodal
                        if is_sufficient_content(app.archived_scraped_content):
                            app.archived_scrape_ok = True
                            n_successes += 1
                        else:
                            last_error = "Scraped content too short."
                    elif response.errors:
                        error = list(response.errors.values())[0]
                        if _is_deferrable(error, url):
                            await app.defer(hours=RETRY_DEFER_HOURS)
                            await app.save_to_db()
                            continue
                        last_error = str(error)
                else:
                    last_error = "scrapeMM did not return a ScrapingResponse."

            # Update dismissal status
            if not (app.original_scrape_ok or app.archived_scrape_ok):
                # Neither original nor archive produced sufficient content, hence, dismiss
                await app.dismiss(last_error or "Could not retrieve sufficient content from either source.")
            elif app.dismissed:
                # The appearance might have been dismissed once before but retrieval was successful this time
                await app.take_back_dismissal()

        except ServerError as e:
            # Run-level condition: every remaining appearance would fail the
            # same way, so this propagates (via `worker()`) instead of being
            # recorded as this appearance's permanent failure.
            raise QuotaExceededError(f"scrapeMM server unreachable: {e}") from e

        except AssertionError:
            # scrapeMM cannot download that type of URL
            await app.dismiss("scrapeMM cannot download that type of URL")

        except Exception as e:
            logger.error(f"Error encountered when scraping appearance at {url}.", exc_info=True)
            await app.dismiss(f"Uncaught error encountered when trying to scrape this "
                              f"appearance. {type(e).__name__}: {e}")

        finally:
            await app.save_to_db()

        # Only scrape as much as needed
        if n_successes >= max_appearances_per_claim:
            return


async def extract_appearances_from_article(review: Review) -> list[Appearance]:
    """Uses the LLM to extract all appearance URLs from the scraped article."""
    urls = await extract_appearance_urls(review)
    appearances = []
    for url in urls:
        try:
            appearance = await appearance_from_url(url)
            if appearance:
                appearances.append(appearance)
        except QuotaExceededError:
            # Run-level condition (e.g. scrapeMM unreachable): let it propagate
            # instead of silently discarding it, which would otherwise leave
            # this review with zero appearances and get it dismissed outright.
            raise
        except Exception:
            pass
    return appearances


async def extract_appearance_urls(review: Review) -> list[str]:
    """Uses the LLM to extract all unique appearance URLs from the scraped article belonging to this review.
    Caution: It cannot extract URLs from embedded social media posts since they are not scraped by scrapemm."""
    article = await review.article
    scraped = str(article.content)[:100_000]  # Truncate to avoid context overflow

    urls = []
    try:
        class ExtractedAppearances(BaseModel):
            appearances: list[str] = Field(
                description="The HTTP URL(s) of the appearances of the Claim, including "
                            "archiving URLs (if available). Leave empty if no appearance URLs are presented in the Article."
            )

        prompt = Prompt(
            "veritas/prompts/find_appearances.md",
            article=scraped,
            date=review.raw_claim_date,
            claimant=review.raw_claimant_name,
            claim=review.raw_claim,
        )
        extracted: ExtractedAppearances = await gpt_cheap.generate(prompt, response_format=ExtractedAppearances)
        if extracted:
            urls = list(set(extracted.appearances))
    except ValidationError as e:
        # Sometimes the LLM fails to format its response correctly. Just discard such instances
        logger.debug(f"Could not extract appearances because LLM response was not formatted correctly. {e}")
    except Exception as e:
        logger.warning(f"Could not extract appearances: {e}\n{traceback.format_exc()}")

    # Ensure that no URL is hallucinated
    urls = [url for url in urls if url in scraped]
    return urls


async def retry_deferred_archive_today_appearances() -> int:
    """Clears the deferral of every appearance (and its review) still waiting
    behind Archive.today's access check, and retries it right away instead of
    waiting out its cooldown. Returns how many reviews were retried.

    Meant to be run once a human has passed the check (e.g. via scrapeMM's
    `configure_archive_today_session()`), so that a successful retry does not
    sit unused until the appearance's `deferred_until` happens to pass. See
    `veritas.gold_evidence.pipeline.retry_deferred_archive_today_sources` for
    the Gold Evidence counterpart, and `scripts/retry_deferred_archive_today.py`
    for the entry point that runs both."""
    from veritas.db import db

    reviews = await db.get_reviews_with_deferred_archive_today_appearances()
    for review in reviews:
        appearances = await review.appearances
        for app in appearances:
            if app.deferred and (is_archive_today_url(app.url) or is_archive_today_url(app.archive_url)):
                app.deferred_until = None
                await app.save_to_db()

        # Clearing the review's own deferral unconditionally is safe: if another,
        # still-deferred appearance of this review needs more time, the retry
        # below re-defers the review from its (now current) appearance states.
        if review.deferred_until is not None:
            review.deferred_until = None
            await review.save_to_db()

        await process_appearances_single_review(review)

    return len(reviews)


def prepare_url(url: str) -> str:
    if is_archive_today_url(url):
        if url.endswith("/image"):  # Remove /image to download actual record instead of page screenshot
            return url[:-6]
    return url
