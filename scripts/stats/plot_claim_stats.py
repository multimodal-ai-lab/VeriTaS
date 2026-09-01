from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

from scripts.stats.common import COLORS
from veritas.db import db


def _ensure_plots_dir(base: str | None) -> Path:
    if base:
        p = Path(base)
        if p.is_dir():
            return p
        return p.parent
    return Path("plots")


def _bar_chart(ax, items: Iterable[tuple[str, int]], title: str, *, color: str, xlabel: str):
    data = list(items)
    if not data:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    total_all = sum(v for _, v in data)
    if total_all <= 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    # Filter out items that contribute less than 0.1% of the total
    data = [(lbl, v) for lbl, v in data if (v / total_all) * 100 >= 0.1]
    if not data:
        ax.set_title(title)
        ax.text(0.5, 0.5, ">= 0.1% only — no data", ha="center", va="center")
        ax.axis("off")
        return
    labels, values = zip(*data)
    y_pos = list(range(len(values)))
    ax.barh(y_pos, values, color=color, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    vmax = max(values) if values else 0
    for i, v in enumerate(values):
        pct = (v / total_all * 100.0)
        label = f"{v} ({pct:.1f}%)"
        ax.text(v + (vmax * 0.01 if vmax else 0.1), i, label, va="center")


def _ring_chart(ax, counts: dict[str, int], title: str):
    labels = ["Success", "Failed", "Queued"]
    values = [counts.get("success", 0), counts.get("failed", 0), counts.get("queued", 0)]
    colors = [COLORS["positive"], COLORS["negative"], COLORS["neutral"]]

    def _autopct(pct):
        return ("%1.0f%%" % pct) if pct >= 1 else ""

    wedges, text_labels, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.4, edgecolor="white"),
        autopct=_autopct,
        pctdistance=0.75,
        labeldistance=1.06,
        textprops=dict(fontsize=12),
    )
    for t in autotexts:
        t.set_color("black")
        t.set_fontsize(12)
        t.set_fontweight("bold")
    ax.set_title(title)
    ax.axis("equal")


# ------------------- Data fetchers -------------------
async def _fetch_top_languages(min_count: int = 50) -> list[tuple[str, int]]:
    """Top languages by count."""
    await db.connect_maybe_initialize()
    query = """
            SELECT COUNT(*) AS n, language
            FROM claims
            WHERE dismissed = FALSE
            GROUP BY language
            HAVING COUNT(*) >= $1
            ORDER BY n DESC;
            """
    rows = await db._fetch(query, min_count)
    return [(str(r["language"]), int(r["n"])) for r in rows]


async def _fetch_appearances_per_claim_distribution() -> list[tuple[str, int]]:
    """Distribution of number of appearances per claim.

    Buckets: 0..9, 10+
    """
    await db.connect_maybe_initialize()
    query = """
            SELECT COALESCE(array_length(appearance_ids, 1), 0) AS n
            FROM claims
            """
    rows = await db._fetch(query)
    cnt: dict[int, int] = defaultdict(int)
    for r in rows:
        cnt[int(r["n"] or 0)] += 1
    items: list[tuple[str, int]] = []
    for k in range(0, 10):
        items.append((f"{k}", cnt.get(k, 0)))
    ten_plus = sum(v for k, v in cnt.items() if k >= 10)
    items.append(("10+", ten_plus))
    return items


async def _fetch_successful_appearances_per_claim_distribution() -> list[tuple[str, int]]:
    """Distribution of number of successfully scraped appearances per claim.

    Success definition (consistent with appearance stats):
    - (original_scrape_ok OR archived_scrape_ok) AND dismissed = FALSE

    Buckets: 0..9, 10+
    """
    await db.connect_maybe_initialize()
    query = """
            SELECT c.id AS claim_id, COALESCE(COUNT(a.id), 0) AS n
            FROM claims c
            LEFT JOIN LATERAL unnest(c.appearance_ids) AS aid(aid) ON TRUE
            LEFT JOIN appearances a
              ON a.id = aid
             AND (COALESCE(a.original_scrape_ok, FALSE) OR COALESCE(a.archived_scrape_ok, FALSE))
             AND COALESCE(a.dismissed, FALSE) = FALSE
            GROUP BY c.id
            """
    rows = await db._fetch(query)
    cnt: dict[int, int] = defaultdict(int)
    for r in rows:
        cnt[int(r["n"] or 0)] += 1
    items: list[tuple[str, int]] = []
    for k in range(0, 10):
        items.append((f"{k}", cnt.get(k, 0)))
    ten_plus = sum(v for k, v in cnt.items() if k >= 10)
    items.append(("10+", ten_plus))
    return items


async def _fetch_claim_validation_counts() -> dict[str, int]:
    """Counts of claim validation results: success, failed, queued.

    Definitions:
    - success: check_completed = TRUE and none of the problem flags are TRUE
    - failed: check_completed = TRUE and any problem flag is TRUE, or dismissed = TRUE
    - queued: check_completed is not TRUE (NULL or FALSE) and dismissed = FALSE
    """
    await db.connect_maybe_initialize()
    query = """
            SELECT
                COALESCE(check_completed, FALSE) AS done,
                COALESCE(is_ambiguous, FALSE) AS amb,
                COALESCE(media_expose_verdict, FALSE) AS mev,
                COALESCE(text_exposes_verdict, FALSE) AS tev,
                COALESCE(missing_referenced_media, FALSE) AS mrm,
                COALESCE(dismissed, FALSE) AS dism
            FROM claims
            """
    rows = await db._fetch(query)
    counts = dict(success=0, failed=0, queued=0)
    for r in rows:
        done = bool(r["done"])  # type: ignore[index]
        dism = bool(r["dism"])  # type: ignore[index]
        any_problem = bool(r["amb"]) or bool(r["mev"]) or bool(r["tev"]) or bool(r["mrm"])  # type: ignore[index]
        if dism:
            counts["failed"] += 1
        elif done and not any_problem:
            counts["success"] += 1
        elif done and any_problem:
            counts["failed"] += 1
        else:
            counts["queued"] += 1
    return counts


async def main_async(save: str | None = None, show: bool = True):
    plots_dir = _ensure_plots_dir(save)
    plots_dir.mkdir(parents=True, exist_ok=True)

    languages, appearances_dist, success_app_dist, validation_counts = await asyncio.gather(
        _fetch_top_languages(50),
        _fetch_appearances_per_claim_distribution(),
        _fetch_successful_appearances_per_claim_distribution(),
        _fetch_claim_validation_counts(),
    )

    # 1) Top 15 languages
    fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    _bar_chart(
        ax1,
        languages[:15],
        "Top 15 languages",
        color=COLORS["blue"],
        xlabel="# Items",
    )
    fig1.tight_layout()
    fig1.savefig(plots_dir / "claims_languages_top15.png")

    # 2) Distribution of number of appearances per claim
    fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=300)
    ordered = [(label, count) for label, count in appearances_dist if label != "10+"]
    ordered.sort(key=lambda x: int(x[0]))
    ordered.append(("10+", next((c for l, c in appearances_dist if l == "10+"), 0)))
    _bar_chart(
        ax2,
        ordered,
        "Number of appearances per claim",
        color=COLORS["darkblue"],
        xlabel="# Claims",
    )
    fig2.tight_layout()
    fig2.savefig(plots_dir / "claims_appearances_distribution.png")

    # 3) Distribution of number of successfully scraped appearances per claim
    fig3b, ax3b = plt.subplots(figsize=(10, 6), dpi=300)
    ordered_success = [(label, count) for label, count in success_app_dist if label != "10+"]
    ordered_success.sort(key=lambda x: int(x[0]))
    ordered_success.append(("10+", next((c for l, c in success_app_dist if l == "10+"), 0)))
    _bar_chart(
        ax3b,
        ordered_success,
        "Number of successfully scraped appearances per claim",
        color=COLORS["light_orange"],
        xlabel="# Claims",
    )
    fig3b.tight_layout()
    fig3b.savefig(plots_dir / "claims_successful_appearances_distribution.png")

    # 4) Ring chart for validation outcome
    fig3, ax3 = plt.subplots(figsize=(7, 7), dpi=300)
    _ring_chart(ax3, validation_counts, "Claim validation status")
    fig3.tight_layout()
    fig3.savefig(plots_dir / "claims_validation_status.png")

    if show:
        plt.show()
    else:
        plt.close("all")


def main():
    parser = argparse.ArgumentParser(description="Plot claim-related statistics")
    parser.add_argument("--save", type=str, default=None, help="Directory or file path to save plots")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot window")
    args = parser.parse_args()

    asyncio.run(main_async(save=args.save, show=not args.no_show))


if __name__ == "__main__":
    main()
