from __future__ import annotations

import asyncio
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from scripts.stats.common import COLORS
from scripts.stats.quarters.common import parse_date
from scripts.stats.plot_appearance_stats import _PLATFORM_DOMAIN_MAP, _normalize_platform_or_domain
from veritas.db import db
from veritas.util.url import get_domain

STATUSES = ["Success", "Failed", "Queued"]

STATUS_COLORS = {
    "Success": COLORS["positive"],
    "Failed": COLORS["negative"],
    "Queued": COLORS["neutral"],
}

# Ordered platform labels for the x-axis (derived from the shared domain map)
PLATFORM_ORDER: list[str] = list(dict.fromkeys(_PLATFORM_DOMAIN_MAP.values())) + ["Other"]


async def fetch_platform_scrape_status(
        start: date | None,
        end: date | None,
) -> dict[str, Counter]:
    """Fetch scrape status counts per platform, filtered by review published date.

    Each appearance is counted once (DISTINCT ON a.id) across all reviews
    whose published date falls within [start, end).

    Status definitions (matching Appearance.scrape_ok):
    - Success: (original_scrape_ok OR archived_scrape_ok) AND dismissed = FALSE
    - Failed:  (NOT original_scrape_ok AND NOT archived_scrape_ok) AND dismissed = TRUE
    - Queued:  (NOT original_scrape_ok AND NOT archived_scrape_ok) AND dismissed = FALSE
    """
    await db.connect_maybe_initialize()

    where = ["a.url IS NOT NULL", "r.published IS NOT NULL"]
    params: list[object] = []
    if start is not None:
        where.append("r.published >= $%d" % (len(params) + 1))
        params.append(datetime.combine(start, datetime.min.time()))
    if end is not None:
        where.append("r.published < $%d" % (len(params) + 1))
        params.append(datetime.combine(end, datetime.min.time()))

    query = f"""
        SELECT DISTINCT ON (a.id)
               a.url,
               (COALESCE(a.original_scrape_ok, FALSE) OR COALESCE(a.archived_scrape_ok, FALSE)) AS has_scrape,
               COALESCE(a.dismissed, FALSE)            AS is_dismissed
        FROM appearances a
        JOIN reviews r ON a.id = ANY(r.appearance_ids)
        WHERE {' AND '.join(where)}
        ORDER BY a.id
    """
    rows = await db._fetch(query, *params)
    print(f"Fetched {len(rows)} appearance rows.")

    known_platforms = set(PLATFORM_ORDER) - {"Other"}
    per_platform: dict[str, Counter] = {p: Counter() for p in PLATFORM_ORDER}

    for r in rows:
        url = r["url"]
        domain = get_domain(url)
        label = _normalize_platform_or_domain(domain)
        platform = label if label in known_platforms else "Other"

        has_scrape = bool(r["has_scrape"])
        is_dismissed = bool(r["is_dismissed"])

        if has_scrape and not is_dismissed:
            per_platform[platform]["Success"] += 1
        elif (not has_scrape) and is_dismissed:
            per_platform[platform]["Failed"] += 1
        else:
            per_platform[platform]["Queued"] += 1

    # Drop platforms with no data
    return {p: c for p, c in per_platform.items() if sum(c.values()) > 0}


def plot_scrape_success_by_platform(
        platform_counts: dict[str, Counter],
        start: date | None,
        end: date | None,
) -> list[tuple[str, Any]]:
    """Return a list of (suffix, figure) pairs for saving/displaying."""
    platforms = [p for p in PLATFORM_ORDER if p in platform_counts]
    n = len(platforms)

    figsize = (max(6, n * 1.1 + 1.5), 5)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    x = list(range(n))
    bottoms = [0.0] * n

    for status in STATUSES:
        heights = []
        for p in platforms:
            cnt = platform_counts[p]
            total = sum(cnt.values())
            heights.append(cnt.get(status, 0) * 100.0 / total if total > 0 else 0.0)
        ax.bar(x, heights, bottom=bottoms, label=status, color=STATUS_COLORS[status], edgecolor="white")
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    # Annotate total count above each bar
    for i, p in enumerate(platforms):
        total = sum(platform_counts[p].values())
        ax.text(i, 101.5, f"n={total:,}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(platforms, rotation=15, ha="right")
    ax.set_ylabel("% of appearances")
    ax.set_ylim(0, 115)
    ax.set_xlim(-0.5, n - 0.5)
    ax.margins(x=0)
    ax.grid(axis="y", linestyle=":", alpha=0.5)

    title = "Scrape success by platform"
    if start or end:
        parts = []
        if start:
            parts.append(f"from {start}")
        if end:
            parts.append(f"to {end}")
        title += f" ({', '.join(parts)})"
    ax.set_title(title)
    ax.legend(loc="upper right")

    fig.tight_layout()
    return [("platform_scrape_success", fig)]


async def main_async(
        start: str | None,
        end: str | None,
        save: str | None,
) -> None:
    print("Plotting platform scrape success rates...")
    start_d = parse_date(start)
    end_d = parse_date(end)

    platform_counts = await fetch_platform_scrape_status(start_d, end_d)
    for plat, cnt in platform_counts.items():
        total = sum(cnt.values())
        print(f"  {plat:12s}  success={cnt['Success']:5d}  failed={cnt['Failed']:5d}  queued={cnt['Queued']:5d}  (n={total})")

    if not platform_counts:
        print("No data found for the given range.")
        return

    figs = plot_scrape_success_by_platform(platform_counts, start_d, end_d)

    if save:
        base = Path(save) / "platform_scrape_success"
        base.mkdir(exist_ok=True, parents=True)
        for suffix, f in figs:
            out_path = base / f"{suffix}.pdf"
            f.savefig(out_path, dpi=300)
            print(f"Saved figure to {out_path}")

    plt.show()
    plt.close("all")


def main():
    asyncio.run(
        main_async(
            start="2020-01-01",
            end="2026-06-30",
            save="plots/",
        )
    )


if __name__ == "__main__":
    main()
