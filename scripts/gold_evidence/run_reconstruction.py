"""Runs the Gold Evidence Reconstruction (Stages 1-3) over a range of claims.

All parameters are read from the `gold_evidence.reconstruction` section of
`config.yaml` - there is no command-line interface. Every key has a default, so
an un-updated `config.yaml` still works. Example snippets:

    # Quarter range, released claims first
    reconstruction:
      start: [ 2024, 1 ]
      end: [ 2024, 4 ]
      limit: 200

    # Specific claims (ignores the date range)
    reconstruction:
      claim_ids: [ 1234, 5678 ]

    # See what would be processed, without spending any API calls
    reconstruction:
      start: [ 2024, 1 ]
      limit: 20
      dry_run: True

The sufficiency ensemble mode and proximity threshold are not duplicated here;
they come from `gold_evidence.ensemble_mode` / `gold_evidence.proximity_threshold`.

By default only *released* claims are processed; once none are left in the given
range, the remaining verdict-complete claims (reviews at stage 6, or 7 for
rectified claims) follow. The gold verdict is never modified.
"""

import asyncio
import json
from datetime import datetime

from veritas import globals, logger
from veritas.db import db
from veritas.gold_evidence import (
    RESUMABLE_STATUSES,
    ensemble_mode,
    proximity_threshold,
)
from veritas.gold_evidence.pipeline import reconstruct_claims, summarize
from veritas.models import QuotaExceededError, RateLimitError
from veritas.util.util import get_quarter_date_range

_cfg: dict = (globals.get("gold_evidence") or {}).get("reconstruction") or {}


def _get(key: str, default):
    value = _cfg.get(key, default)
    return default if value is None else value


#: First quarter of the claim date range, e.g. [2024, 1]. None -> no lower bound.
start: tuple[int, int] | None = tuple(_cfg["start"]) if _cfg.get("start") else None
#: Last quarter of the claim date range (inclusive). None -> no upper bound.
end: tuple[int, int] | None = tuple(_cfg["end"]) if _cfg.get("end") else None
limit: int = int(_get("limit", 100))
batch_size: int = int(_get("batch_size", 25))
#: Process exactly these claim IDs, ignoring the date range.
claim_ids: list[int] | None = _cfg.get("claim_ids") or None
include_unreleased: bool = bool(_get("include_unreleased", False))
redo: bool = bool(_get("redo", False))
re_extract: bool = bool(_get("re_extract", False))
re_filter: bool = bool(_get("re_filter", False))
dry_run: bool = bool(_get("dry_run", False))
log_level: str = _get("log_level", "INFO")


def date_range() -> tuple[datetime | None, datetime | None]:
    start_date = end_date = None
    if start:
        start_date, _ignored = get_quarter_date_range(*start)
    if end:
        _ignored, end_date = get_quarter_date_range(*end)
    return start_date, end_date


async def main() -> None:
    logger.setLevel(log_level)
    await db.connect_maybe_initialize(max_connections=4)

    start_date, end_date = date_range()
    statuses = None if redo else RESUMABLE_STATUSES

    if dry_run:
        claims = await db.get_claims_for_gold_evidence(
            limit=limit, start_date=start_date, end_date=end_date,
            statuses=statuses, released_first=not include_unreleased,
            claim_ids=claim_ids)
        print(f"{len(claims)} claim(s) would be processed:")
        for claim in claims:
            print(f"  {claim.id:>8}  {claim.date_str:<24} "
                  f"{'released' if claim.released else 'unreleased':<11} "
                  f"{claim.gold_evidence_status or '-'}")
        await db.close()
        return

    processed = 0
    all_outcomes = []
    # Claims stay selectable while their status is resumable, and a claim that ends
    # up deferred or errored keeps such a status. Excluding what this run already
    # touched keeps the batches moving forward instead of re-serving the same rows.
    handled: list[int] = []
    try:
        while processed < limit:
            this_batch_size = min(batch_size, limit - processed)
            claims = await db.get_claims_for_gold_evidence(
                limit=this_batch_size, start_date=start_date, end_date=end_date,
                statuses=statuses, released_first=not include_unreleased,
                claim_ids=claim_ids, exclude_claim_ids=handled)
            if not claims:
                logger.info("No further claims to process.")
                break
            handled.extend(claim.id for claim in claims)

            outcomes = await reconstruct_claims(
                claims,
                mode=ensemble_mode,
                threshold=proximity_threshold,
                re_extract=re_extract,
                re_filter=re_filter,
            )
            all_outcomes.extend(outcomes)
            processed += len(claims)
            logger.info(f"Processed {processed}/{limit} claims.")

    except (QuotaExceededError, RateLimitError) as e:
        logger.error(f"Aborting run: {e}")

    finally:
        if all_outcomes:
            print(json.dumps(summarize(all_outcomes), indent=2))
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
