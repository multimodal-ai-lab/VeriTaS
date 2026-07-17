from __future__ import annotations
from collections import Counter
from datetime import date
from typing import Any

from pydantic import ValidationError

from veritas.common import Claim
from veritas.common.annotation.rating import Category3Bin
from veritas.db import db
from scripts.stats.common import COLORS, TITLE_APPEND
from scripts.stats.quarters.common import quarter_key, build_single_chart, select_top_categories, remap_to_top, \
    fetch_review_rows, to_percentage_shares


async def fetch_claims_by_quarter(start: date | None, end: date | None, mode: str = "all") -> list[Claim]:
    await db.connect_maybe_initialize()

    where = ["c.date IS NOT NULL", "c.verdict_ids != '{}'"]
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
        SELECT c.*
        FROM claims c
        LEFT JOIN reviews r ON r.id = c.review_ids[1]
        WHERE {' AND '.join(where)}
        ORDER BY c.date
    """
    rows = await db._fetch(query, *params)
    return [Claim.model_validate(dict(r)) for r in rows]

async def fetch_claim_integrity_by_quarter(start: date | None, end: date | None, mode: str = "all") -> tuple[list[tuple[int, int]], dict[tuple[int, int], Counter]]:
    claims = await fetch_claims_by_quarter(start, end, mode=mode)

    per_quarter: dict[tuple[int, int], Counter] = {}
    for claim in claims:
        try:
            verdict = await claim.current_verdict
        except (AssertionError, ValidationError):
            # Older verdict version that doesn't have individual ratings
            continue

        label = "Unset"
        match verdict.integrity.as_3_bin():
            case Category3Bin.POSITIVE:
                label = "Intact"
            case Category3Bin.NEGATIVE:
                label = "Compromised"
            case Category3Bin.NEUTRAL:
                label = "NEI"

        qk = quarter_key(claim.date)
        if qk not in per_quarter:
            per_quarter[qk] = Counter()
        per_quarter[qk][label] += 1

    keys = sorted(per_quarter.keys())
    return keys, per_quarter

async def plot_claim_integrity(start_d: date | None, end_d: date | None, q_keys: list[tuple[int, int]], mode: str = "all") -> list[tuple[str, Any]]:
    _q_keys_claims, q_counts_claims = await fetch_claim_integrity_by_quarter(start_d, end_d, mode=mode)

    # align
    q_counts_claims = {k: q_counts_claims.get(k, Counter()) for k in q_keys}
    
    claim_categories = ["Intact", "NEI", "Compromised"]
    
    claim_colors = {
        "Intact": COLORS["positive"],
        "NEI": COLORS["neutral"],
        "Compromised": COLORS["negative"],
    }

    return [
        build_single_chart(
            "integrity",
            q_keys,
            q_counts_claims,
            claim_categories,
            "Claim integrity shares per quarter" + TITLE_APPEND[mode],
            "% of claims",
            category_colors=claim_colors,
            # hline=[500, 1000],
            include_ring=True,
            as_shares=True,
        )
    ]

async def plot_claim_languages(start_d: date | None, end_d: date | None, q_keys: list[tuple[int, int]], top_languages: int, mode: str = "all") -> list[tuple[str, Any]]:
    claims = await fetch_claims_by_quarter(start_d, end_d, mode=mode)

    q_counts_lang: dict[tuple[int, int], Counter] = {k: Counter() for k in q_keys}
    for claim in claims:
        qk = quarter_key(claim.date)
        if qk in q_counts_lang:
            lang = (claim.language_name or "(unknown)")
            q_counts_lang[qk][lang] += 1

    lang_top = select_top_categories(q_counts_lang, top_languages, other_label="Other")
    q_counts_lang_top = remap_to_top(q_counts_lang, lang_top, other_label="Other")

    return [
        build_single_chart(
            "claim_languages",
            q_keys,
            q_counts_lang_top,
            lang_top,
            "Claim language shares per quarter" + TITLE_APPEND[mode],
            "% of claims",
            include_ring=True,
            as_shares=True,
        )
    ]

async def plot_claim_publishers(start_d: date | None, end_d: date | None, q_keys: list[tuple[int, int]], top_publishers: int, mode: str = "all") -> list[tuple[str, Any]]:
    await db.connect_maybe_initialize()

    where = ["c.date IS NOT NULL", "c.verdict_ids != '{}'"]
    params: list[object] = []
    if start_d is not None:
        where.append("c.date >= $%d" % (len(params) + 1))
        params.append(start_d)
    if end_d is not None:
        where.append("c.date <= $%d" % (len(params) + 1))
        params.append(end_d)

    if mode == "release":
        where.append("(c.released_quarter = TRUE OR c.released_longitudinal = TRUE)")
    elif mode == "natural":
        where.append("NOT c.dismissed AND NOT c.is_rectified")
    else:
        where.append("NOT c.dismissed")

    query = f"""
        SELECT DISTINCT c.id, p.id, c.date, p.name AS publisher_name
        FROM claims c
        LEFT JOIN reviews r ON r.id = c.review_ids[1]
        LEFT JOIN publishers p ON p.id = r.publisher_id
        WHERE {' AND '.join(where)}
    """
    rows = await db._fetch(query, *params)

    q_counts_pub: dict[tuple[int, int], Counter] = {k: Counter() for k in q_keys}
    for r in rows:
        qk = quarter_key(r["date"])
        if qk in q_counts_pub:
            pub = r["publisher_name"] or "(unknown)"
            q_counts_pub[qk][pub] += 1

    pub_top = select_top_categories(q_counts_pub, top_publishers, other_label="Other")
    q_counts_pub_top = remap_to_top(q_counts_pub, pub_top, other_label="Other")

    return [
        build_single_chart(
            "claim_publishers",
            q_keys,
            q_counts_pub_top,
            pub_top,
            "Claim publisher shares per quarter" + TITLE_APPEND[mode],
            "% of claims",
            include_ring=True,
            as_shares=True,
        )
    ]
