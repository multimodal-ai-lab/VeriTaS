from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

from scripts.stats.common import COLORS
from veritas.db import db
from veritas.util.url import get_domain, normalize_archiver_label

# Known social media platforms can use multiple domains. We aggregate these
# domains under a single platform label for appearance domain stats. Unknown
# domains remain as-is.
_PLATFORM_DOMAIN_MAP: dict[str, str] = {
    # X (Twitter)
    "twitter.com": "X",
    "x.com": "X",
    "t.co": "X",
    # Facebook
    "facebook.com": "Facebook",
    "fb.com": "Facebook",
    "m.me": "Facebook",
    "fb.watch": "Facebook",
    # YouTube
    "youtube.com": "YouTube",
    "youtu.be": "YouTube",
    # Instagram
    "instagram.com": "Instagram",
    # TikTok
    "tiktok.com": "TikTok",
    # Reddit
    # "reddit.com": "Reddit",
    # Telegram
    "t.me": "Telegram",
    "telegram.me": "Telegram",
    # LinkedIn
    # "linkedin.com": "LinkedIn",
    # "lnkd.in": "LinkedIn",
    # Threads
    "threads.net": "Threads",
}


def _normalize_platform_or_domain(domain: str | None) -> str:
    """Map known social domains to a unified platform label; otherwise keep domain.

    The input should already be normalized to a registrable domain
    (i.e., without subdomain), as returned by get_domain().
    """
    if not domain:
        return "(unknown)"
    d = domain.lower().strip()
    return _PLATFORM_DOMAIN_MAP.get(d, d)


def _ensure_plots_dir(base: str | None) -> Path:
    if base:
        p = Path(base)
        if p.is_dir():
            return p
        # If a file path (not a dir) was passed, use its parent
        return p.parent
    return Path("plots")


async def _fetch_platform_counts() -> Counter:
    """Top scraping methods (from appearances.scrape_method). Excludes unknown/NULL."""
    await db.connect_maybe_initialize()
    query = """
            SELECT scrape_method, COUNT(*) AS n
            FROM appearances
            WHERE scrape_method IS NOT NULL AND TRIM(scrape_method) <> ''
            GROUP BY scrape_method
            ORDER BY n DESC
            """
    rows = await db._fetch(query)
    c = Counter()
    for r in rows:
        label = r["scrape_method"]
        c[label] = int(r["n"])
    return c


async def _fetch_scrape_method_success_fail() -> dict[str, dict[str, int]]:
    """Return per-scrape_method counts split into success vs fail.

    Definitions:
    - success: (original_scrape_ok OR archived_scrape_ok) AND dismissed = FALSE
    - fail: (NOT original_scrape_ok AND NOT archived_scrape_ok) AND dismissed = TRUE
    Queued (no scrape and not dismissed) is excluded from this chart.
    Unknown/blank scrape_method rows are excluded.
    """
    await db.connect_maybe_initialize()
    query = """
            SELECT scrape_method,
                   (COALESCE(original_scrape_ok, FALSE) OR COALESCE(archived_scrape_ok, FALSE)) AS has_scrape,
                   COALESCE(dismissed, FALSE)  AS is_dismissed
            FROM appearances
            WHERE scrape_method IS NOT NULL AND TRIM(scrape_method) <> ''
            """
    rows = await db._fetch(query)
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        method = r["scrape_method"]
        has_scrape = bool(r["has_scrape"])  # type: ignore[index]
        is_dismissed = bool(r["is_dismissed"])  # type: ignore[index]
        bucket: dict[str, int] = out.setdefault(method, {"success": 0, "fail": 0})
        if has_scrape and not is_dismissed:
            bucket["success"] += 1
        elif (not has_scrape) and is_dismissed:
            bucket["fail"] += 1
        # else: queued or rare combinations -> not counted here
    return out


async def _fetch_archiver_counts() -> Counter:
    """Top archiving services by domain(archive_url)."""
    await db.connect_maybe_initialize()
    query = """
            SELECT archive_url
            FROM appearances
            WHERE archive_url IS NOT NULL
            """
    rows = await db._fetch(query)
    c = Counter()
    for r in rows:
        url = r["archive_url"]
        if not url:
            continue
        domain = get_domain(url)
        label = normalize_archiver_label(domain)
        c[label] += 1
    return c


async def _fetch_appearance_domain_counts() -> Counter:
    """Top domains of appearance URLs (from appearances.url)."""
    await db.connect_maybe_initialize()
    query = """
            SELECT url
            FROM appearances
            WHERE url IS NOT NULL
            """
    rows = await db._fetch(query)
    c = Counter()
    for r in rows:
        url = r["url"]
        if not url:
            continue
        domain = get_domain(url)
        label = _normalize_platform_or_domain(domain)
        c[label] += 1
    return c


async def _fetch_status_counts() -> dict[str, int]:
    """Return counts for success/failed/queued appearances.

    - success: (original_scrape_ok OR archived_scrape_ok) AND dismissed = FALSE
    - failed: (NOT original_scrape_ok AND NOT archived_scrape_ok) AND dismissed = TRUE
    - queued: (NOT original_scrape_ok AND NOT archived_scrape_ok) AND dismissed = FALSE
    Other combinations (rare) are folded into the closest category logically.
    """
    await db.connect_maybe_initialize()
    query = """
            SELECT (COALESCE(original_scrape_ok, FALSE) OR COALESCE(archived_scrape_ok, FALSE)) AS has_scrape,
                   COALESCE(dismissed, FALSE) AS is_dismissed
            FROM appearances
            """
    rows = await db._fetch(query)
    counts = dict(success=0, failed=0, queued=0)
    for r in rows:
        has_scrape = bool(r["has_scrape"])  # type: ignore[index]
        is_dismissed = bool(r["is_dismissed"])  # type: ignore[index]
        if has_scrape and not is_dismissed:
            counts["success"] += 1
        elif (not has_scrape) and is_dismissed:
            counts["failed"] += 1
        elif (not has_scrape) and (not is_dismissed):
            counts["queued"] += 1
        else:
            # Fallback: count dismissed with scrape as failed; others as queued
            if is_dismissed:
                counts["failed"] += 1
            else:
                counts["queued"] += 1
    return counts


async def _fetch_url_archive_presence_counts() -> dict[str, int]:
    """Return counts for appearances grouped by presence of URL vs. archive URL.

    Buckets:
    - only_url:        url present, archive_url missing
    - only_archive:    archive_url present, url missing
    - both:            both present
    Empty strings are treated as missing.
    """
    await db.connect_maybe_initialize()
    query = (
        """
        SELECT
            SUM(CASE WHEN url IS NOT NULL AND TRIM(url) <> ''
                      AND (archive_url IS NULL OR TRIM(archive_url) = '') THEN 1 ELSE 0 END) AS only_url,
            SUM(CASE WHEN archive_url IS NOT NULL AND TRIM(archive_url) <> ''
                      AND (url IS NULL OR TRIM(url) = '') THEN 1 ELSE 0 END) AS only_archive,
            SUM(CASE WHEN url IS NOT NULL AND TRIM(url) <> ''
                      AND archive_url IS NOT NULL AND TRIM(archive_url) <> '' THEN 1 ELSE 0 END) AS both
        FROM appearances
        """
    )
    row = await db._fetchrow(query)
    return {
        "only_url": int(row["only_url"]) if row and row["only_url"] is not None else 0,  # type: ignore[index]
        "only_archive": int(row["only_archive"]) if row and row["only_archive"] is not None else 0,  # type: ignore[index]
        "both": int(row["both"]) if row and row["both"] is not None else 0,  # type: ignore[index]
    }


async def _fetch_dismissed_reasons(limit: int = 10) -> list[tuple[str, int]]:
    await db.connect_maybe_initialize()
    query = """
            SELECT dismissed_reason, COUNT(*) AS n
            FROM appearances
            WHERE dismissed = TRUE
            GROUP BY dismissed_reason
            ORDER BY n DESC
            """
    rows = await db._fetch(query)
    pairs: list[tuple[str, int]] = []
    for r in rows:
        reason = r["dismissed_reason"] or "(unspecified)"
        pairs.append((str(reason), int(r["n"])) )
    return pairs[:limit]


def _bar_chart(ax, items: Iterable[tuple[str, int]], title: str, *, color: str, edgecolor: str | None = None):
    data = list(items)
    if not data:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    # Compute total share before filtering
    total_all = sum(v for _, v in data)
    if total_all <= 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    # Filter out items with < 0.1% share of the total
    data = [(lbl, v) for lbl, v in data if (v / total_all) * 100 >= 0.1]
    if not data:
        ax.set_title(title)
        ax.text(0.5, 0.5, ">= 0.1% only — no data", ha="center", va="center")
        ax.axis("off")
        return
    labels, values = zip(*data)
    y_pos = list(range(len(values)))
    ax.barh(y_pos, values, color=color, edgecolor=edgecolor or "white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # largest on top
    ax.set_title(title)
    ax.set_xlabel("# Appearances")
    maxv = max(values) if values else 0
    offset = (0.01 * maxv) if maxv else 0.1
    for i, v in enumerate(values):
        pct = (v / total_all) * 100
        label = f"{v} ({pct:.1f}%)"
        ax.text(v + offset, i, label, va="center")


def _ring_chart(ax, counts: dict[str, int]):
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
    # Make percentages readable
    for t in autotexts:
        t.set_color("black")
        t.set_fontsize(12)
        t.set_fontweight("bold")
    ax.set_title("Appearance scraping status")
    ax.axis('equal')


def _ring_chart_url_archive(ax, counts: dict[str, int]):
    labels = ["Only URL", "Only Archive URL", "Both"]
    values = [counts.get("only_url", 0), counts.get("only_archive", 0), counts.get("both", 0)]
    total = sum(values)
    if total <= 0:
        ax.set_title("Appearance URL vs. Archive URL")
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return

    colors = [COLORS["orange"], COLORS["darkblue"], COLORS["neutral"]]

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
    ax.set_title("Appearance URL vs. Archive URL")
    ax.axis("equal")


def _stacked_bar_success_fail(ax, method_counts: dict[str, dict[str, int]], title: str):
    # Compute grand total across all methods first
    grand_total_all = sum((v.get("success", 0) + v.get("fail", 0)) for v in method_counts.values())
    if grand_total_all <= 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return
    # Filter out methods contributing < 0.1% of total
    filtered = [
        (m, v.get("success", 0), v.get("fail", 0))
        for m, v in method_counts.items()
        if ((v.get("success", 0) + v.get("fail", 0)) / grand_total_all) * 100 >= 0.1
    ]
    if not filtered:
        ax.set_title(title)
        ax.text(0.5, 0.5, ">= 0.1% only — no data", ha="center", va="center")
        ax.axis("off")
        return
    # Determine top 10 by total (success + fail)
    items = sorted(
        filtered,
        key=lambda x: (x[1] + x[2]),
        reverse=True,
    )[:10]
    labels = [m for m, _, _ in items]
    success_vals = [s for _, s, _ in items]
    fail_vals = [f for _, _, f in items]
    y_pos = list(range(len(labels)))

    # Plot as stacked horizontal bars: fail on the left, success on the right
    ax.barh(y_pos, fail_vals, color=COLORS["negative"], edgecolor="white", label="Fail")
    ax.barh(y_pos, success_vals, left=fail_vals, color=COLORS["positive"], edgecolor="white", label="Success")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("# Appearances")
    ax.legend(loc="lower right")
    # Add value labels at the end of the stacked bars with percentages of the overall grand total
    totals = [f + s for f, s in zip(fail_vals, success_vals)]
    maxv = max(totals) if totals else 0
    for i, total in enumerate(totals):
        pct = (total / grand_total_all) * 100
        label = f"{total} ({pct:.1f}%)"
        ax.text(total + (0.01 * maxv if maxv else 0.1), i, label, va="center")


async def main_async(save: str | None):
    platforms_c, archivers_c, status_counts, reasons, app_domains_c, method_sf, url_arch_presence = await asyncio.gather(
        _fetch_platform_counts(),
        _fetch_archiver_counts(),
        _fetch_status_counts(),
        _fetch_dismissed_reasons(10),
        _fetch_appearance_domain_counts(),
        _fetch_scrape_method_success_fail(),
        _fetch_url_archive_presence_counts(),
    )

    # Prepare top 10 lists
    platforms_top = platforms_c.most_common(10)
    archivers_top = archivers_c.most_common(10)
    app_domains_top = app_domains_c.most_common(10)
    # Helper for saving/showing individual figures
    def _finalize(fig, filename: str | None):
        fig.tight_layout()
        if save:
            out_dir = _ensure_plots_dir(save)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / filename  # type: ignore[operator]
            fig.savefig(out_path, dpi=300)
            print(f"Saved figure to {out_path}")
        plt.show()
        plt.close(fig)

    # Plot 1: Top scraping methods (stacked success vs. fail)
    fig1, ax1 = plt.subplots(figsize=(9, 6), constrained_layout=True, dpi=300)
    _stacked_bar_success_fail(
        ax1,
        method_sf,
        "Top scraping methods (success vs. fail)",
    )
    _finalize(fig1, "appearance_scraping_methods.pdf" if save else None)

    # Plot 2: Top archiving services
    fig2, ax2 = plt.subplots(figsize=(8, 6), constrained_layout=True, dpi=300)
    _bar_chart(
        ax2,
        archivers_top,
        "Archiving services",
        color=COLORS["darkblue"],
        edgecolor="#ffffff",
    )
    _finalize(fig2, "appearance_archivers.pdf" if save else None)

    # Plot 3: Appearance scraping status (ring)
    fig3, ax3 = plt.subplots(figsize=(6, 6), constrained_layout=True, dpi=300)
    _ring_chart(ax3, status_counts)
    _finalize(fig3, "appearance_status.pdf" if save else None)

    # Plot 4: Top appearance URL domains
    fig4, ax4 = plt.subplots(figsize=(8, 6), constrained_layout=True, dpi=300)
    _bar_chart(
        ax4,
        app_domains_top,
        "Top appearance URL domains",
        color=COLORS["orange"],
        edgecolor="#ffffff",
    )
    _finalize(fig4, "appearance_domains.pdf" if save else None)

    # Plot 5: URL vs Archive URL presence (ring)
    fig5, ax5 = plt.subplots(figsize=(6, 6), constrained_layout=True, dpi=300)
    _ring_chart_url_archive(ax5, url_arch_presence)
    _finalize(fig5, "appearance_url_vs_archive.pdf" if save else None)

    # Console output of top dismissal reasons
    if reasons:
        print("Top dismissal reasons (appearances):")
        width = max(len(r[0]) for r in reasons)
        for reason, n in reasons:
            print(f"  {reason.ljust(width)}  : {n}")
    else:
        print("No dismissed appearances found.")

    # No combined figure anymore.


def main():
    asyncio.run(main_async(save="plots/"))


if __name__ == "__main__":
    main()
