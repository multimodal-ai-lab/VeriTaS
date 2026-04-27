from __future__ import annotations

import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, date
from typing import Iterable, Any
from pathlib import Path

from scripts.stats.common import COLORS
from veritas.db import db


@dataclass
class Row:
    published: datetime
    stage: int | None
    language: str | None
    publisher_name: str | None
    ifcn_status: str | None = None
    efcsn_status: str | None = None
    dismissed: bool | None = None
    claim_data: str | None = None


def parse_date(s: str | None) -> date | None:
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()


def quarter_key(dt: datetime | date) -> tuple[int, int]:
    q = (dt.month - 1) // 3 + 1
    return dt.year, q


def generate_quarter_keys(start: date, end: date) -> list[tuple[int, int]]:
    """Generate all quarter keys between start and end (inclusive of quarters)."""
    keys = []
    curr_y, curr_q = quarter_key(start)
    end_y, end_q = quarter_key(end)

    while (curr_y < end_y) or (curr_y == end_y and curr_q <= end_q):
        keys.append((curr_y, curr_q))
        curr_q += 1
        if curr_q > 4:
            curr_q = 1
            curr_y += 1
    return keys


def quarter_label(key: tuple[int, int]) -> str:
    """Legacy label (not used for ticks anymore)."""
    y, q = key
    return f"{y}-Q{q}"


def quarter_tick_labels(quarter_keys: list[tuple[int, int]]) -> list[str]:
    """Create tick labels like Q1, Q2, Q3, Q4 with the year only on Q1 (newline)."""
    labels: list[str] = []
    for (year, q) in quarter_keys:
        if q == 1:
            labels.append(f"Q1\n{year}")
        else:
            labels.append(f"Q{q}")
    return labels


async def fetch_review_rows(start: date | None, end: date | None) -> list[Row]:
    """Fetch minimal fields for plotting from DB."""
    await db.connect_maybe_initialize()

    where = [f"r.published IS NOT NULL"]
    params: list[object] = []
    if start is not None:
        where.append(f"r.published >= $%d" % (len(params) + 1))
        params.append(datetime.combine(start, datetime.min.time()))
    if end is not None:
        where.append(f"r.published < $%d" % (len(params) + 1))
        params.append(datetime.combine(end, datetime.min.time()))

    query = f"""
        SELECT r.published as published, r.stage, r.language, r.dismissed,
               p.name AS publisher_name,
               p.ifcn_status AS ifcn_status,
               p.efcsn_status AS efcsn_status
        FROM reviews r
        LEFT JOIN publishers p ON p.id = r.publisher_id
        WHERE {' AND '.join(where)}
        ORDER BY r.published
    """

    rows = await db._fetch(query, *params)
    out: list[Row] = []
    for r in rows:
        out.append(
            Row(
                published=r["published"],
                stage=(int(r["stage"]) if r["stage"] is not None else None),
                language=r["language"],
                publisher_name=r["publisher_name"],
                ifcn_status=r["ifcn_status"],
                efcsn_status=r["efcsn_status"],
            )
        )
    return out


def aggregate_by_quarter(rows: Iterable[Row], key_getter) -> tuple[
    list[tuple[int, int]], dict[tuple[int, int], Counter]]:
    per_quarter: dict[tuple[int, int], Counter] = defaultdict(Counter)
    for r in rows:
        qk = quarter_key(r.published)
        label = key_getter(r)
        per_quarter[qk][label] += 1
    keys = sorted(per_quarter.keys())
    return keys, per_quarter


def stacked_bar(
        ax: plt.Axes,
        quarter_keys: list[tuple[int, int]],
        counts_by_q: dict[tuple[int, int], Counter],
        categories: list[str],
        title: str,
        ylabel: str,
        category_colors: dict[str, Any] | None = None,
        *,
        reverse_categories: bool = False,
        y_max: int | float | None = None,
        hline: int | float | Iterable[int | float] | None = None,
        show_legend: bool = True,
) -> None:
    x_labels = quarter_tick_labels(quarter_keys)
    x = list(range(len(quarter_keys)))
    bottoms = [0] * len(quarter_keys)

    cmap = plt.get_cmap('tab20')
    default_colors = [cmap(i % 20) for i in range(len(categories))]
    cat_colors: dict[str, Any] = {}
    category_colors = category_colors or {}
    for idx, cat in enumerate(categories):
        if cat in category_colors:
            cat_colors[cat] = category_colors[cat]
        elif cat == "Other" or cat == "Unknown":
            cat_colors[cat] = COLORS["neutral"]
        elif cat == "No Signatory" or cat == "Neither":
            cat_colors[cat] = COLORS["negative"]
        else:
            cat_colors[cat] = default_colors[idx]

    plot_categories = list(reversed(categories)) if reverse_categories else list(categories)

    for idx, cat in enumerate(plot_categories):
        heights = [counts_by_q.get(k, Counter()).get(cat, 0) for k in quarter_keys]
        ax.bar(x, heights, bottom=bottoms, label=cat, color=cat_colors[cat], edgecolor='white')
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=0, ha='center')
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if show_legend:
        ax.legend(loc='upper left', ncol=1, reverse=True)
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    if y_max is not None:
        ax.set_ylim(0, y_max)
    if hline is not None:
        if not isinstance(hline, Iterable):
            hline = (hline,)
        for hline in sorted(set(hline)):
            ax.axhline(hline, color='#FFFFFF', linestyle='--', linewidth=2.0)

    n = len(quarter_keys)
    if n > 0:
        ax.set_xlim(-0.5, n - 0.5)
        ax.margins(x=0)

    q1_indices = [i for i, (_y, q) in enumerate(quarter_keys) if q == 1]
    for xi in q1_indices:
        ax.axvline(x=xi, color='0.8', linestyle='--', linewidth=0.8, alpha=0.8, zorder=0)


def compute_figsize(n_quarters: int) -> tuple[float, float]:
    return (max(8, n_quarters * 0.4), 5)


def build_single_chart(
        suffix: str,
        q_keys: list[tuple[int, int]],
        counts_by_q: dict[tuple[int, int], Counter],
        categories: list[str],
        title: str,
        ylabel: str = "# Reviews",
        category_colors: dict[str, Any] | None = None,
        *,
        reverse_categories: bool = False,
        y_max: int | float | None = None,
        hline: int | float | Iterable[int | float] | None = None,
        include_ring: bool = False,
        ring_counts_by_q: dict[tuple[int, int], Counter] | None = None,
        as_shares: bool = False,
) -> tuple[str, plt.Figure]:
    if as_shares:
        # Convert to percentage shares for the bar chart
        shares_dict = to_percentage_shares(counts_by_q)
        plot_counts = {k: Counter(v) for k, v in shares_dict.items()}
        if ylabel == "# Reviews" or ylabel == "Number of claims": # Default y-labels
             ylabel = "% of total"
        y_max = 100 if y_max is None else y_max
    else:
        plot_counts = counts_by_q

    if include_ring:
        figsize = compute_figsize(len(q_keys))
        # Increase width for the ring chart
        # We want the ring chart to have a constant width of 4.2 units.
        # figsize[0] is the width of the bar chart.
        bar_width = figsize[0]
        ring_width = 4.2
        total_width = bar_width + ring_width
        fig, (ax_bar, ax_ring) = plt.subplots(
            1, 2, figsize=(total_width, figsize[1]), dpi=300,
            gridspec_kw={'width_ratios': [bar_width, ring_width]}
        )
    else:
        fig, ax_bar = plt.subplots(1, 1, figsize=compute_figsize(len(q_keys)), dpi=300)
        ax_ring = None

    stacked_bar(
        ax_bar,
        q_keys,
        plot_counts,
        categories,
        title,
        ylabel,
        category_colors,
        reverse_categories=reverse_categories,
        y_max=y_max,
        hline=hline,
        show_legend=not include_ring,
    )

    if include_ring and ax_ring:
        # Aggregate data across all shown quarters
        total_counts = Counter()
        source_counts = ring_counts_by_q if ring_counts_by_q is not None else counts_by_q
        for qk in q_keys:
            total_counts.update(source_counts.get(qk, Counter()))

        # Get colors (redundant with stacked_bar logic but needed here)
        cmap = plt.get_cmap('tab20')
        default_colors = [cmap(i % 20) for i in range(len(categories))]
        cat_colors: dict[str, Any] = {}
        category_colors_fixed = category_colors or {}
        for idx, cat in enumerate(categories):
            if cat in category_colors_fixed:
                cat_colors[cat] = category_colors_fixed[cat]
            elif cat == "Other" or cat == "Unknown":
                cat_colors[cat] = COLORS["neutral"]
            elif cat == "No Signatory" or cat == "Neither":
                cat_colors[cat] = COLORS["negative"]
            else:
                cat_colors[cat] = default_colors[idx]

        sizes = [total_counts.get(cat, 0) for cat in categories]
        colors = [cat_colors[cat] for cat in categories]

        # Only plot if there's data
        if sum(sizes) > 0:
            def func(pct, allvals):
                absolute = int(round(pct / 100. * sum(allvals)))
                return f"{absolute:,}\n({pct:.1f}%)"

            wedges, texts, autotexts = ax_ring.pie(
                sizes,
                labels=None,
                autopct=lambda pct: func(pct, sizes),
                startangle=90,
                colors=colors,
                pctdistance=0.75,
                wedgeprops={'width': 0.45, 'edgecolor': 'white'}
            )
            plt.setp(autotexts, size=10, weight="bold", color="white")
            plt.title("Total")

            # Shift ring plot slightly to the top to make space for the legend
            # pos = ax_ring.get_position()
            # ax_ring.set_position([pos.x0, pos.y0 + 0.3, pos.width, pos.height])

            # Add total count in the center
            total_sum = int(sum(sizes))
            ax_ring.text(0, 0, f"{total_sum:,}", ha='center', va='center', fontweight='bold', size=20)

            # Add horizontal legend below the ring chart
            # We use categories reversed to match the stacked bar order if it was reversed, 
            # but usually bar chart legends are shown in plot order.
            # Use a fixed width for the legend bounding box to prevent it from
            # influencing the overall layout width. We anchor it to the ring axes.
            ncol = min(len(categories), 2 if len(categories) > 3 else 1)
            legend_width = 0.9 if ncol >= 2 else 0.6
            ax_ring.legend(
                wedges,
                categories,
                title=None,
                loc='upper center',
                bbox_to_anchor=((1 - legend_width) / 2, 0.1, legend_width, 0.0),
                ncol=ncol,
                frameon=True,
                fontsize=9,
                mode="expand",
                borderaxespad=0
            )
        else:
            ax_ring.text(0.5, 0.5, "No data", ha='center', va='center')
            ax_ring.axis('off')

    fig.tight_layout()
    return (suffix, fig)


def select_top_categories(counts_by_q: dict[tuple[int, int], Counter], top_k: int, other_label: str = "Other") -> list[
    str]:
    total = Counter()
    for c in counts_by_q.values():
        total.update(c)
    most = [name for name, _ in total.most_common(top_k)]
    return most + ([other_label] if len(total) > len(most) else [])


def remap_to_top(counts_by_q: dict[tuple[int, int], Counter], top_categories: list[str], other_label: str = "Other") -> \
dict[tuple[int, int], Counter]:
    top_set = set(top_categories)
    remapped: dict[tuple[int, int], Counter] = {}
    for qk, cnt in counts_by_q.items():
        c = Counter()
        for k, v in cnt.items():
            if k in top_set:
                c[k] += v
            else:
                if other_label in top_set:
                    c[other_label] += v
        remapped[qk] = c
    return remapped


def to_percentage_shares(counts_by_q: dict[tuple[int, int], Counter]) -> dict[tuple[int, int], dict[str, float]]:
    shares: dict[tuple[int, int], dict[str, float]] = {}
    for qk, cnt in counts_by_q.items():
        total = sum(cnt.values())
        if total <= 0:
            shares[qk] = {k: 0.0 for k in cnt.keys()}
        else:
            shares[qk] = {k: (v * 100.0) / total for k, v in cnt.items()}
    return shares
