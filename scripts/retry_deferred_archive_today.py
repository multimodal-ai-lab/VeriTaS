"""Retries every appearance and evidence source VeriTaS deferred behind
Archive.today's access check.

Archive.today gates a snapshot's replay page with a CAPTCHA. scrapeMM does not
solve it: a gated retrieval is buffered on scrapeMM's side and reported back to
VeriTaS as a failure, which the main pipeline (Stage 4) and Gold Evidence
Retrieval turn into a deferral instead of a permanent failure - see
`veritas.pipeline.stage_4._is_deferrable` and
`veritas.gold_evidence.retrieval.SourceRetrieval.gated`. Once a human has
passed the check - ``python -m scripts.configure_archive_today`` - every URL
that was buffered while it was up is retrieved and cached on scrapeMM's side,
so asking scrapeMM for the same URL again now succeeds.

The ordinary pipeline runs pick deferred items up again on their own once their
`deferred_until` window passes, but that window can be far shorter than the
wait actually was (a captcha is solved by a human, not on a schedule). This
script clears the deferral of every appearance/evidence source pointing at
Archive.today and retries it immediately, rather than waiting for the periodic
re-scan.

Usage: python -m scripts.retry_deferred_archive_today
"""

import asyncio
import json

from veritas import log_to_console, logger
from veritas.db import db
from veritas.gold_evidence import ensemble_mode, proximity_threshold
from veritas.gold_evidence.pipeline import retry_deferred_archive_today_sources, summarize
from veritas.models import QuotaExceededError
from veritas.pipeline.stage_4 import retry_deferred_archive_today_appearances


async def main() -> None:
    log_to_console("INFO")
    await db.connect_maybe_initialize()
    try:
        n_reviews = await retry_deferred_archive_today_appearances()
        logger.info(f"Retried {n_reviews} review(s) with a deferred Archive.today appearance.")

        outcomes = await retry_deferred_archive_today_sources(
            mode=ensemble_mode, threshold=proximity_threshold)
        if outcomes:
            logger.info(f"Retried {len(outcomes)} claim(s) with a deferred Archive.today "
                        f"evidence source.")
            print(json.dumps(summarize(outcomes), indent=2))

        try:
            from scrapemm import get_archive_today_buffer
            remaining = get_archive_today_buffer()
        except ImportError:
            remaining = []
        if remaining:
            logger.info(f"{len(remaining)} Archive.today URL(s) are still buffered in "
                        f"scrapeMM (still gated, or never requested through VeriTaS). "
                        f"Pass the access check again to clear them.")
    except QuotaExceededError as e:
        # e.g. scrapeMM is unreachable: whatever wasn't retried yet stays
        # untouched (still deferred, or still awaiting the earlier session's
        # retry) for the next run - nothing here needs to be undone.
        logger.error(f"Aborting: {e}")
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
