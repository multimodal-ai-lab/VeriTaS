from __future__ import annotations
from collections import Counter
from datetime import date
from typing import Any

from veritas.db import db
from scripts.stats.common import COLORS, TITLE_APPEND
from scripts.stats.quarters.common import quarter_key, build_single_chart

async def fetch_appearance_retrieval_by_quarter(start: date | None, end: date | None, mode: str = "all") -> tuple[list[tuple[int, int]], dict[tuple[int, int], Counter]]:
    await db.connect_maybe_initialize()

    # Appearances are usually tied to reviews, and we want to aggregate by the quarter they were published
    # or the claim date depending on the mode. Consistent with platforms.py:
    date_reference = "c.date" if mode == "release" else "r.published"

    where = [f"{date_reference} IS NOT NULL"]
    params: list[object] = []
    if start is not None:
        where.append(f"{date_reference} >= $%d" % (len(params) + 1))
        params.append(start)
    if end is not None:
        where.append(f"{date_reference} <= $%d" % (len(params) + 1))
        params.append(end)

    if mode == "release":
        where.append("(c.released_quarter = TRUE OR c.released_longitudinal = TRUE)")
    elif mode == "natural":
        where.append("NOT c.dismissed AND NOT c.is_rectified")
    else:
        where.append("NOT c.dismissed")

    query = f"""
        SELECT {date_reference} as date, 
               (COALESCE(a.original_scrape_ok, FALSE) OR COALESCE(a.archived_scrape_ok, FALSE)) AS has_scrape,
               COALESCE(a.dismissed, FALSE) AS is_dismissed
        FROM appearances a
        JOIN reviews r ON a.id = ANY(r.appearance_ids)
        LEFT JOIN claims c ON c.id = r.claim_id
        WHERE {' AND '.join(where)}
        ORDER BY {date_reference}
    """
    rows = await db._fetch(query, *params)

    per_quarter: dict[tuple[int, int], Counter] = {}
    for r in rows:
        qk = quarter_key(r["date"])
        if qk not in per_quarter:
            per_quarter[qk] = Counter()
        
        has_scrape = bool(r["has_scrape"])
        is_dismissed = bool(r["is_dismissed"])
        
        if has_scrape and not is_dismissed:
            label = "Success"
        elif (not has_scrape) and is_dismissed:
            label = "Failed"
        elif (not has_scrape) and (not is_dismissed):
            label = "Queued"
        else:
            # Fallback as in plot_appearance_stats.py
            if is_dismissed:
                label = "Failed"
            else:
                label = "Queued"
        
        per_quarter[qk][label] += 1

    keys = sorted(per_quarter.keys())
    return keys, per_quarter

async def plot_appearance_retrieval(start_d: date | None, end_d: date | None, q_keys: list[tuple[int, int]], mode: str = "all") -> list[tuple[str, Any]]:
    _q_keys_ret, q_counts_ret = await fetch_appearance_retrieval_by_quarter(start_d, end_d, mode=mode)
    
    # align
    q_counts_ret = {k: q_counts_ret.get(k, Counter()) for k in q_keys}
    
    categories = ["Success", "Failed", "Queued"]
    category_colors = {
        "Success": COLORS["positive"],
        "Failed": COLORS["negative"],
        "Queued": COLORS["neutral"],
    }

    return [
        build_single_chart(
            "appearance_retrieval",
            q_keys,
            q_counts_ret,
            categories,
            "Appearance retrieval shares per quarter" + TITLE_APPEND[mode],
            "% of appearances",
            category_colors=category_colors,
            include_ring=True,
            as_shares=True,
        )
    ]
