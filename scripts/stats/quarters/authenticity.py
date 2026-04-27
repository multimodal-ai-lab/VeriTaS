from __future__ import annotations

from collections import Counter
from datetime import date
from typing import Any

from scripts.stats.common import COLORS, TITLE_APPEND
from scripts.stats.quarters.common import quarter_key, build_single_chart
from veritas.common.verdict import MediumVerdict
from veritas.db import db


async def fetch_authenticity_by_quarter(start: date | None, end: date | None, mode: str = "all") -> tuple[list[tuple[int, int]], dict[tuple[int, int], Counter]]:
    await db.connect_maybe_initialize()

    where = ["c.date IS NOT NULL", "v.is_current IS TRUE"]
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
        SELECT 
            c.date, 
            v.media
        FROM claims c
        JOIN verdicts v ON v.claim_id = c.id
        LEFT JOIN reviews r ON r.id = c.review_ids[1]
        WHERE {' AND '.join(where)}
        ORDER BY c.date
    """
    rows = await db._fetch(query, *params)

    per_quarter: dict[tuple[int, int], Counter] = {}
    for r in rows:
        d = r["date"]
        media_verdicts = r["media"]  # This is a list of dicts (MediumVerdict)
        if not media_verdicts:
            continue

        qk = quarter_key(d)
        if qk not in per_quarter:
            per_quarter[qk] = Counter()

        for mv in media_verdicts:
            mv = MediumVerdict.model_validate(mv)
            
            score = mv.authenticity.score
            tags = mv.authenticity.tags

            if score is None:
                label = "Unknown"
            elif score > 1/3:
                label = "Pristine"
            elif score < -1/3:
                if "AI-generated" in tags:
                    label = "AI-generated"
                else:
                    label = "Other Fabricated"
            else:
                label = "Unknown"
            
            per_quarter[qk][label] += 1

    keys = sorted(per_quarter.keys())
    return keys, per_quarter

async def plot_authenticity(start_d: date | None, end_d: date | None, q_keys: list[tuple[int, int]], mode: str = "all") -> list[tuple[str, Any]]:
    _q_keys_auth, q_counts_auth = await fetch_authenticity_by_quarter(start_d, end_d, mode=mode)

    # align
    q_counts_auth = {k: q_counts_auth.get(k, Counter()) for k in q_keys}
    
    auth_categories = ["Pristine", "Unknown", "Other Fabricated", "AI-generated"]
    
    # AI-generated: shade of red, Other Fabricated: another shade of red
    auth_colors = {
        "Pristine": COLORS["positive"],
        "Unknown": COLORS["neutral"],
        "Other Fabricated": COLORS["orange"],
        "AI-generated": COLORS["negative"],
    }

    return [
        build_single_chart(
            "authenticity",
            q_keys,
            q_counts_auth,
            auth_categories,
            "Media authenticity shares per quarter" + TITLE_APPEND[mode],
            "% of media items",
            category_colors=auth_colors,
            include_ring=True,
            as_shares=True,
        )
    ]
