"""Compute and print statistics about claims in a given date range.

Configuration:
    Set the START and END variables below to control behavior.
    The date range is inclusive of START and exclusive of END.

Printed statistics (within the given date range):
1. Overview:
   - Total number of claims
   - Number and share of dismissed claims
2. Dismissal reasons:
   - Summary of dismissal reasons that occur at least twice across all claims.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime

from veritas.db import db


@dataclass
class ClaimRow:
    id: int
    date: datetime | None
    dismissed: bool
    dismissed_reason: str | None


# --- Static configuration (edit these) ---
# Specify the start and end of the date range. Strings in YYYY-MM-DD or date objects are accepted.
START: str | date = "2020-01-01"
END: str | date = "2026-03-31"

# Filter for rectified status:
#   - "all": include everything
#   - "non-rectified": only claims where is_rectified is False
#   - "rectified": only claims where is_rectified is True
RECTIFIED_FILTER: str = "rectified"


def _to_date(d: str | date) -> date:
    if isinstance(d, date):
        return d
    return datetime.strptime(d, "%Y-%m-%d").date()


async def _fetch_rows(start: date, end: date, rectified_filter: str = "all") -> list[ClaimRow]:
    await db.connect_maybe_initialize()

    # Determine rectification filter clause
    rect_clause = ""
    if rectified_filter == "non-rectified":
        rect_clause = "AND is_rectified IS FALSE"
    elif rectified_filter == "rectified":
        rect_clause = "AND is_rectified IS TRUE"

    # Query claims within the date range
    query = f"""
            SELECT id,
                   date,
                   dismissed,
                   dismissed_reason
            FROM claims
            WHERE date >= $1
              AND date < $2
              {rect_clause}
            ORDER BY date;
            """

    rows = await db._fetch(query, datetime.combine(start, datetime.min.time()),
                           datetime.combine(end, datetime.min.time()))

    result: list[ClaimRow] = []
    for row in rows:
        result.append(
            ClaimRow(
                id=int(row["id"]),
                date=row["date"],
                dismissed=bool(row["dismissed"]),
                dismissed_reason=row["dismissed_reason"],
            )
        )
    return result


def _print_stats(rows: list[ClaimRow]) -> None:
    n_total = len(rows)
    n_dismissed = sum(1 for r in rows if r.dismissed)
    share = (n_dismissed / n_total * 100.0) if n_total else 0.0

    print("=== Claim statistics ===")
    print(f"Total claims: {n_total}")
    print(f"Dismissed:    {n_dismissed} ({share:.1f}%)\n")


def _print_dismissal_summary(rows: list[ClaimRow]) -> None:
    """Prints a summary of dismissal reasons that occur at least twice."""
    n_total = len(rows)
    if not n_total:
        return

    dismissed_rows = [r for r in rows if r.dismissed]
    reasons = Counter(r.dismissed_reason or "(unspecified)" for r in dismissed_rows)

    # Filter reasons that occur at least twice
    frequent_reasons = [(reason, count) for reason, count in reasons.items() if count >= 2]
    # Sort by count descending
    frequent_reasons.sort(key=lambda x: x[1], reverse=True)

    print("=== Dismissal reasons (across all claims, min. 2 occurrences) ===")
    if not frequent_reasons:
        print("None found.\n")
        return

    for reason, count in frequent_reasons:
        share = (count / n_total * 100.0)
        # Only take the first line of the reason if it's multi-line
        reason_display = reason.split('\n')[0]
        print(f"  {count:>5} ({share:>5.1f}%)  {reason_display}")
    print()


async def main_async(start: str | date, end: str | date, rectified_filter: str = "all") -> None:
    start = _to_date(start)
    end = _to_date(end)

    rows = await _fetch_rows(start, end, rectified_filter)
    _print_stats(rows)
    _print_dismissal_summary(rows)


def main() -> None:
    # Use the static configuration above. Adjust START, END as needed.
    asyncio.run(main_async(START, END, RECTIFIED_FILTER))


if __name__ == "__main__":
    main()
