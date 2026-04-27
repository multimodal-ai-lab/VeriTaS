from __future__ import annotations
from collections import Counter
from datetime import date
from typing import Any

from ezmm import MultimodalSequence
from veritas.db import db
from scripts.stats.common import COLORS, TITLE_APPEND
from scripts.stats.quarters.common import aggregate_by_quarter, to_percentage_shares, build_single_chart, Row, quarter_key

async def fetch_media_by_quarter(start: date | None, end: date | None, mode: str = "all") -> tuple[list[tuple[int, int]], dict[tuple[int, int], Counter]]:
    await db.connect_maybe_initialize()

    where = ["c.date IS NOT NULL"]
    params: list[object] = []
    if start is not None:
        where.append("c.date >= $%d" % (len(params) + 1))
        params.append(start)
    if end is not None:
        where.append("c.date <= $%d" % (len(params) + 1))
        params.append(end)

    if mode == "release":
        where.append("(c.released_quarter = TRUE OR c.released_longitudinal = TRUE)")
    elif mode == "natural":
        where.append("NOT c.dismissed AND NOT c.is_rectified")
    else:
        where.append("NOT c.dismissed")

    query = f"""
        SELECT c.date, c.data
        FROM claims c
        LEFT JOIN reviews r ON r.id = c.review_ids[1]
        WHERE {' AND '.join(where)}
        ORDER BY c.date
    """
    rows = await db._fetch(query, *params)

    per_quarter: dict[tuple[int, int], Counter] = {}
    for r in rows:
        d = r["date"]
        claim_data = r["data"]

        qk = quarter_key(d)
        if qk not in per_quarter:
            per_quarter[qk] = Counter()
        
        mm = MultimodalSequence(claim_data)
        has_img = mm.has_images()
        has_vid = mm.has_videos()
        
        if has_img and has_vid:
            label = "Both"
        elif has_img:
            label = "Images"
        elif has_vid:
            label = "Videos"
        else:
            label = "No media"
            
        per_quarter[qk][label] += 1

    keys = sorted(per_quarter.keys())
    return keys, per_quarter

async def plot_media(start_d: date | None, end_d: date | None, q_keys: list[tuple[int, int]], mode: str = "all") -> list[tuple[str, Any]]:
    _q_keys_media, q_counts_media = await fetch_media_by_quarter(start_d, end_d, mode=mode)

    # align
    q_counts_media = {k: q_counts_media.get(k, Counter()) for k in q_keys}
    media_categories = ["Images", "Both", "Videos", "No media"]

    return [
        build_single_chart(
            "media",
            q_keys,
            q_counts_media,
            media_categories,
            "Claim media shares per quarter" + TITLE_APPEND[mode],
            "% of claims",
            category_colors={
                "Images": COLORS["orange"],
                "Both": COLORS["light_orange"],
                "Videos": COLORS["negative"],
                "No media": COLORS["neutral"],
            },
            include_ring=True,
            as_shares=True,
        )
    ]
