from __future__ import annotations

import asyncio
from collections import Counter
from datetime import date, datetime
from typing import Any

import matplotlib.pyplot as plt

from scripts.stats.common import TITLE_APPEND, COLORS
from scripts.stats.quarters.common import build_single_chart, generate_quarter_keys, quarter_key, parse_date
from veritas.db import db


def _to_date(d: str | date | None) -> date | None:
    if d is None:
        return None
    if isinstance(d, date):
        return d
    return datetime.strptime(d, "%Y-%m-%d").date()


async def fetch_claim_progress_by_quarter(
        start: date | None,
        end: date | None,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], Counter]]:
    await db.connect_maybe_initialize()

    where = ["c.date IS NOT NULL"]
    params: list[Any] = []

    if start is not None:
        where.append("c.date >= $%d" % (len(params) + 1))
        params.append(datetime.combine(start, datetime.min.time()))
    if end is not None:
        where.append("c.date < $%d" % (len(params) + 1))
        params.append(datetime.combine(end, datetime.min.time()))

    query = f"""
        SELECT c.id,
               c.date,
               c.dismissed,
               c.is_rectified,
               c.variant_id,
               v.integrity
        FROM claims c
        LEFT JOIN verdicts v
               ON v.claim_id = c.id
              AND v.is_current = TRUE
        WHERE {' AND '.join(where)}
        ORDER BY c.date, c.id
    """
    rows = await db._fetch(query, *params)

    per_quarter: dict[tuple[int, int], Counter] = {}

    for r in rows:
        qk = quarter_key(r["date"])
        if qk not in per_quarter:
            per_quarter[qk] = Counter()

        dismissed = bool(r["dismissed"])
        is_rectified = bool(r["is_rectified"])
        integrity = r["integrity"]

        if dismissed:
            label = "Dismissed"
        elif integrity is None:
            label = "Queued"
        else:
            if integrity < -1 / 3:
                label = "Compromised"
            elif integrity > 1 / 3:
                if is_rectified:
                    label = "Intact (rectified)"
                else:
                    label = "Intact (original)"
            else:
                label = "Unknown"

        per_quarter[qk][label] += 1

    keys = sorted(per_quarter.keys())
    return keys, per_quarter


async def main_async(start: str | date | None = None, end: str | date | None = None, save: str | None = None) -> None:
    start_d = _to_date(start) or date(2016, 1, 1)
    end_d = _to_date(end) or date.today()

    q_keys, q_counts = await fetch_claim_progress_by_quarter(start_d, end_d)
    if not q_keys:
        print("No claims found for the given filters.")
        return

    # Ensure all quarters are present on the axis, even if empty
    all_q_keys = generate_quarter_keys(start_d, end_d)
    q_counts = {k: q_counts.get(k, Counter()) for k in all_q_keys}

    categories = [
        "Dismissed",
        "Queued",
        "Compromised",
        "Unknown",
        "Intact (rectified)",
        "Intact (original)",
    ]

    colors = {
        "Dismissed": "#333333",
        "Queued": "#CCCCCC",
        "Compromised": COLORS["negative"],
        "Unknown": COLORS["neutral"],
        "Intact (rectified)": COLORS["positive"],
        "Intact (original)": COLORS["blue"],
    }

    suffix, fig = build_single_chart(
        "claim_progress",
        all_q_keys,
        q_counts,
        categories,
        "Available claims per quarter" + TITLE_APPEND["all"],
        "Number of claims",
        category_colors=colors,
        reverse_categories=True,  # first category is topmost in the chart
    )

    if save:
        out_dir = f"{save}/quarter_stats/claims"
        import os
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/{suffix}.pdf"
        fig.savefig(out_path, dpi=300)
        print(f"Saved figure to {out_path}")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    asyncio.run(
        main_async(
            start="2020-01-01",
            end="2026-03-31",
            save="plots/",
        )
    )
