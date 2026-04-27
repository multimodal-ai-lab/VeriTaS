"""Compute and print statistics about reviews in a given date range.

Configuration:
    Set the START, END, and NO_PLOT variables below to control behavior.
    The date range is inclusive of START and exclusive of END.

Printed statistics (within the given date range):
1. Per-stage (1..6) pipeline stats:
   - Number of input instances
   - Number of output instances
   - Number of dismissals and the top 5 dismissal reasons at that stage
2. Stage ≥ 3 overview (unchanged from before):
   - Total number of reviews
   - Number and share of dismissed reviews
   - Top 5 dismissal reasons and their count
   - Shares of reviews with an IFCN/EFCSN signatory publisher, a scraped article, and an identified claimant

If the date range spans at least one year, a stacked bar chart is shown with the
number of dismissed and non-dismissed reviews per quarter (based on stage ≥ 3, unless NO_PLOT is True).
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime

from veritas.db import db


@dataclass
class ReviewRow:
    id: int
    published: datetime | None
    stage: int
    dismissed: bool
    dismissed_reason: str | None
    has_claimant: bool
    is_signatory: bool
    has_scraped_article: bool


# --- Static configuration (edit these) ---
# Specify the start and end of the date range. Strings in YYYY-MM-DD or date objects are accepted.
START: str | date = "2016-01-01"
END: str | date = "2026-03-31"


def _to_date(d: str | date) -> date:
    if isinstance(d, date):
        return d
    return datetime.strptime(d, "%Y-%m-%d").date()


async def _fetch_rows(start: date, end: date) -> list[ReviewRow]:
    await db.connect_maybe_initialize()

    # Compose SQL to get everything we need in one go
    query = """
            SELECT r.id,
                   r.published,
                   r.stage,
                   r.dismissed,
                   r.dismissed_reason,
                   (r.raw_claimant_name IS NOT NULL)                                  AS has_claimant,
                   (CASE
                        WHEN (p.ifcn_status IN ('active', 'in_renewal') OR p.efcsn_status = 'active')
                            THEN TRUE
                        ELSE FALSE END)                                               AS is_signatory,
                   (CASE WHEN a.scraped_page IS NOT NULL THEN TRUE ELSE FALSE END) AS has_scraped_article
            FROM reviews r
                     LEFT JOIN publishers p ON r.publisher_id = p.id
                     LEFT JOIN articles a ON RTRIM(a.url, '/') = RTRIM(r.url, '/')
            WHERE r.published >= $1
              AND r.published < $2
            ORDER BY r.published; \
            """

    rows = await db._fetch(query, datetime.combine(start, datetime.min.time()),
                           datetime.combine(end, datetime.min.time()))

    result: list[ReviewRow] = []
    for row in rows:
        result.append(
            ReviewRow(
                id=int(row["id"]),
                published=row["published"],
                stage=int(row["stage"]) if row["stage"] is not None else 0,
                dismissed=bool(row["dismissed"]),
                dismissed_reason=row["dismissed_reason"],
                has_claimant=bool(row["has_claimant"]),
                is_signatory=bool(row["is_signatory"]),
                has_scraped_article=bool(row["has_scraped_article"]),
            )
        )
    return result


def _print_stats(rows: list[ReviewRow]) -> None:
    n_total = len(rows)
    n_dismissed = sum(1 for r in rows if r.dismissed)
    share = (n_dismissed / n_total * 100.0) if n_total else 0.0

    print("=== Review statistics ===")
    print(f"Total reviews: {n_total}")
    print(f"Dismissed: {n_dismissed} ({share:.1f}%)\n")


def _print_per_stage_flow(rows: list[ReviewRow]) -> None:
    """Prints, for each stage 1..6, input/output counts and dismissals with top reasons.

    Interpretation:
    - Input to stage s: reviews with stage >= s-1
    - Output of stage s: reviews with stage >= s
    - Dismissed at stage s: reviews with dismissed=True and stage == s-1
    """
    print("=== Pipeline per-stage statistics ===")
    for s in range(1, 7):
        inputs = [r for r in rows if (r.stage or 0) >= (s - 1)]
        outputs = [r for r in rows if (r.stage or 0) >= s]
        dismissed = [r for r in rows if r.dismissed and (r.stage or 0) == (s - 1)]

        n_in = len(inputs)
        n_out = len(outputs)
        n_dis = len(dismissed)
        n_queued = n_in - n_out - n_dis
        n_processed = n_out + n_dis

        print(f"Stage {s}:")
        print(f"  Inputs:     {n_in}")
        print(f"  Outputs:    {n_out}")
        print(f"  Queued:     {n_queued}")
        print(f"  Dismissals: {n_dis}")

        if n_dis:
            reasons = Counter(r.dismissed_reason or "(unspecified)" for r in dismissed)
            print("  Top 5 dismissal reasons:")
            for reason, count in reasons.most_common(5):
                print(f"      {count:>5} ({count / n_processed:.1%})  {reason.split('\n')[0]}")

        print()


def _print_dismissal_summary(rows: list[ReviewRow]) -> None:
    """Prints a summary of dismissal reasons that occur at least twice across all stages."""
    n_total = len(rows)
    if not n_total:
        return

    dismissed_rows = [r for r in rows if r.dismissed]
    reasons = Counter(r.dismissed_reason or "(unspecified)" for r in dismissed_rows)

    # Filter reasons that occur at least twice
    frequent_reasons = [(reason, count) for reason, count in reasons.items() if count >= 2]
    # Sort by count descending
    frequent_reasons.sort(key=lambda x: x[1], reverse=True)

    print("=== Dismissal reasons (across all stages, min. 2 occurrences) ===")
    if not frequent_reasons:
        print("None found.\n")
        return

    for reason, count in frequent_reasons:
        share = (count / n_total * 100.0)
        # Only take the first line of the reason if it's multi-line
        reason_display = reason.split('\n')[0]
        print(f"  {count:>5} ({share:>5.1f}%)  {reason_display}")
    print()


async def main_async(start: str | date, end: str | date) -> None:
    start = _to_date(start)
    end = _to_date(end)

    rows = await _fetch_rows(start, end)
    _print_stats(rows)
    _print_per_stage_flow(rows)
    _print_dismissal_summary(rows)


def main() -> None:
    # Use the static configuration above. Adjust START, END, NO_PLOT as needed.
    asyncio.run(main_async(START, END))


if __name__ == "__main__":
    main()
