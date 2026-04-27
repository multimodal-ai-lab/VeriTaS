from __future__ import annotations
from collections import Counter
from datetime import date
from typing import Any

from veritas.db import db
from veritas.common.platforms import PLATFORMS
from veritas.util import get_domain
from scripts.stats.common import COLORS, TITLE_APPEND
from scripts.stats.plot_appearance_stats import _PLATFORM_DOMAIN_MAP
from scripts.stats.quarters.common import quarter_key, remap_to_top, to_percentage_shares, quarter_label, build_single_chart

async def fetch_appearance_platforms_by_quarter(start: date | None, end: date | None, mode: str = "all") -> tuple[list[tuple[int, int]], dict[tuple[int, int], Counter]]:
    await db.connect_maybe_initialize()

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
        SELECT {date_reference} as date, a.url
        FROM appearances a
        JOIN reviews r ON a.id = ANY(r.appearance_ids)
        LEFT JOIN claims c ON c.id = r.claim_id
        WHERE {' AND '.join(where)}
        ORDER BY {date_reference}
    """
    rows = await db._fetch(query, *params)

    per_quarter: dict[tuple[int, int], Counter] = {}
    for r in rows:
        published = r["date"]
        url = r["url"]
        if not url:
            continue

        domain = get_domain(url)
        platform = PLATFORMS.get(domain, domain)

        label = platform if platform else domain
        if not label or label == "(unknown)":
            continue
        qk = quarter_key(published)
        if qk not in per_quarter:
            per_quarter[qk] = Counter()
        per_quarter[qk][label] += 1

    keys = sorted(per_quarter.keys())
    return keys, per_quarter

def _invert_platform_map_to_canonical_domain() -> dict[str, str]:
    label_to_domain: dict[str, str] = {}
    for domain, label in _PLATFORM_DOMAIN_MAP.items():
        if label not in label_to_domain:
            label_to_domain[label] = domain
    return label_to_domain

def _compute_domain_shares_for_summary(
    quarter_keys: list[tuple[int, int]],
    raw_counts_by_q: dict[tuple[int, int], Counter],
) -> dict[tuple[int, int], Counter]:
    label_to_domain = _invert_platform_map_to_canonical_domain()
    out: dict[tuple[int, int], Counter] = {}
    for qk in quarter_keys:
        cnt = raw_counts_by_q.get(qk, Counter())
        total = sum(cnt.values())
        if total <= 0:
            out[qk] = Counter()
            continue
        domain_counts: Counter = Counter()
        for label, v in cnt.items():
            if not label or label in {"Other", "(unknown)"}:
                continue
            domain = label_to_domain.get(label, label)
            domain_counts[domain] += v
        out[qk] = Counter({d: (n * 100.0) / total for d, n in domain_counts.items()})
    return out

def _print_top_domains_summary(
    quarter_keys: list[tuple[int, int]],
    domain_shares_by_q: dict[tuple[int, int], Counter],
    *,
    top_n: int = 5,
) -> None:
    for qk in quarter_keys:
        shares = domain_shares_by_q.get(qk, Counter())
        if not shares:
            continue
        top = sorted(shares.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
        print(f"{quarter_label(qk)}:")
        for i, (domain, pct) in enumerate(top, start=1):
            print(f"  {i}) {domain} — {pct:.1f}%")
        print()

async def plot_platforms(start_d: date | None, end_d: date | None, q_keys: list[tuple[int, int]], mode: str = "all") -> list[tuple[str, Any]]:
    q_keys_plat, q_counts_plat = await fetch_appearance_platforms_by_quarter(start_d, end_d, mode=mode)
    q_counts_plat_raw: dict[tuple[int, int], Counter] = {k: Counter(v) for k, v in q_counts_plat.items()}

    # align
    all_q2 = sorted(set(q_keys).union(q_keys_plat))
    q_keys = all_q2
    q_counts_plat = {k: q_counts_plat.get(k, Counter()) for k in q_keys}

    known_platforms = list(dict.fromkeys(_PLATFORM_DOMAIN_MAP.values()))
    platform_categories = known_platforms + ["Other"]
    q_counts_plat = remap_to_top(q_counts_plat, platform_categories, other_label="Other")

    platform_colors = {
        "X": "#000000",
        "Facebook": "#1877F2",
        "YouTube": "#FF0000",
        "Instagram": "#f68846",
        "TikTok": "#f12a54",
        "Reddit": "#f54100",
        "Telegram": "#27a4e4",
        "LinkedIn": "#0A66C2",
        "Threads": "#444444",
        "Other": COLORS["neutral"],
    }

    domain_shares_by_q = _compute_domain_shares_for_summary(q_keys, {k: q_counts_plat_raw.get(k, Counter()) for k in q_keys})
    _print_top_domains_summary(q_keys, domain_shares_by_q, top_n=5)

    figs = [
        build_single_chart(
            "platforms",
            q_keys,
            q_counts_plat,
            platform_categories,
            "Appearance platform shares per quarter" + TITLE_APPEND[mode],
            "% of appearances",
            category_colors=platform_colors,
            include_ring=True,
            as_shares=True,
        )
    ]
    return figs
