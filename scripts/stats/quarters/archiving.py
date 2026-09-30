from __future__ import annotations
from collections import Counter
from datetime import date
from typing import Any

from veritas.db import db
from veritas.util import get_domain
from veritas.util.url import ARCHIVING_SERVICE_NAMES, normalize_archiver_label
from scripts.stats.common import COLORS, TITLE_APPEND
from scripts.stats.quarters.common import quarter_key, remap_to_top, build_single_chart


async def fetch_appearance_archivers_by_quarter(start: date | None, end: date | None, mode: str = "all") -> tuple[list[tuple[int, int]], dict[tuple[int, int], Counter]]:
    await db.connect_maybe_initialize()

    date_reference = "c.date" if mode == "release" else "r.published"

    where = [f"{date_reference} IS NOT NULL", "a.archive_url IS NOT NULL"]
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
        SELECT {date_reference} as date, a.archive_url
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
        archive_url = r["archive_url"]
        if not archive_url:
            continue

        domain = get_domain(archive_url)
        label = normalize_archiver_label(domain)
        if not label or label == "(unknown)":
            continue

        qk = quarter_key(published)
        if qk not in per_quarter:
            per_quarter[qk] = Counter()
        per_quarter[qk][label] += 1

    keys = sorted(per_quarter.keys())
    return keys, per_quarter


async def plot_archiving_services(start_d: date | None, end_d: date | None, q_keys: list[tuple[int, int]], mode: str = "all") -> list[tuple[str, Any]]:
    q_keys_arch, q_counts_arch = await fetch_appearance_archivers_by_quarter(start_d, end_d, mode=mode)

    # align
    all_q = sorted(set(q_keys).union(q_keys_arch))
    q_keys = all_q
    q_counts_arch = {k: q_counts_arch.get(k, Counter()) for k in q_keys}

    known_services = list(dict.fromkeys(ARCHIVING_SERVICE_NAMES.values()))
    service_categories = known_services + ["Other"]
    q_counts_arch = remap_to_top(q_counts_arch, service_categories, other_label="Other")

    service_colors = {
        "Archive.today": COLORS["orange"],
        "Internet Archive": COLORS["darkblue"],
        "Perma.cc": COLORS["blue"],
        "Ghostarchive": COLORS["light_orange"],
        "Awesome Screenshot": COLORS["soft_orange"],
        "MediaVault": COLORS["soft_blue"],
        "Archive.st": COLORS["soft_light_orange"],
        "ArchivePort": COLORS["positive"],
        "Other": COLORS["neutral"],
    }

    return [
        build_single_chart(
            "archiving_services",
            q_keys,
            q_counts_arch,
            service_categories,
            "Archiving service shares per quarter" + TITLE_APPEND[mode],
            "% of appearances",
            category_colors=service_colors,
            include_ring=True,
            as_shares=True,
        )
    ]
