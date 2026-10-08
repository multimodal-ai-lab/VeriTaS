"""Backfills the original URL of appearances that only have an Archive.today
snapshot.

`Appearance.url` stays empty when the original URL could not be resolved at
extraction time - which `resolve_archive_today_url()` can currently only do
for the long URL form (the original URL is embedded in the path itself, no
network needed). Short-code URLs (https://archive.ph/uTVE4) have no client-side
resolution path since scrapeMM's client/server split moved snapshot lookup
server-side without exposing it over the client's HTTP API; those appearances
stay unresolved until scrapeMM offers one again.

Re-running this script is nonetheless useful whenever `resolve_archive_today_url`
gains new resolving power (as it has before), since it picks up every
appearance still missing its original URL without needing to touch the ones
already scraped.

Usage: python -m scripts.resolve_archive_today_original_urls
"""

import asyncio

from veritas import log_to_console, logger
from veritas.common.appearance import resolve_missing_archive_today_original_urls
from veritas.db import db


async def main() -> None:
    log_to_console("INFO")
    await db.connect_maybe_initialize()
    try:
        n_candidates, n_resolved = await resolve_missing_archive_today_original_urls()
        logger.info(f"Resolved {n_resolved}/{n_candidates} original URL(s).")
    finally:
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
