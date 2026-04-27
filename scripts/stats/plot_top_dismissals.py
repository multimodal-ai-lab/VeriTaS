"""Plots the distribution of dismissal reasons as a horizontal bar chart (top 10).

Run this script manually (requires DB access configured in config/globals.yaml):
    python -m scripts.stats.plot_top_dismissals
"""

import asyncio
import os

from veritas.db import db
from scripts.stats.common import COLORS, _ensure_plots_dir

MAX_LABEL_LENGTH = 80
TOP_N = 10


SCRAPE_AGGREGATE_KEY = "Unable to scrape article"


def _aggregate_scrape_reasons(reasons: dict[str, int]) -> dict[str, int]:
    """Aggregate all reasons containing 'Unable to scrape article' into one entry."""
    aggregated: dict[str, int] = {}
    scrape_total = 0
    for reason, count in reasons.items():
        if SCRAPE_AGGREGATE_KEY in reason:
            scrape_total += count
        else:
            aggregated[reason] = count
    if scrape_total:
        aggregated[SCRAPE_AGGREGATE_KEY] = scrape_total
    return aggregated


def plot_top_dismissals(reasons: dict[str, int]):
    """Plot a horizontal bar chart of the top 10 dismissal reasons."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib is required to generate plots. Please install it: pip install matplotlib")
        return

    if not reasons:
        print("No dismissal reasons to plot.")
        return

    # Aggregate scrape-related reasons
    reasons = _aggregate_scrape_reasons(reasons)

    # Exclude reasons containing "obsolete" or "Could not reformulate claim" (case-insensitive)
    reasons = {r: c for r, c in reasons.items()
               if "obsolete" not in r.lower()
               and "could not reformulate claim" not in r.lower()}

    # Sort by count descending and take top N
    sorted_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:TOP_N]

    # Reverse for horizontal bar chart (highest at top)
    sorted_reasons = list(reversed(sorted_reasons))

    labels = [r[:MAX_LABEL_LENGTH] for r, _ in sorted_reasons]
    counts = [c for _, c in sorted_reasons]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    bars = ax.barh(range(len(counts)), counts, color=COLORS["negative"])

    # Place reason strings inside the bars
    for bar, label in zip(bars, labels):
        ax.text(
            bar.get_width() * 0.01 + bar.get_x(),
            bar.get_y() + bar.get_height() / 2,
            f"  {label}",
            va="center",
            ha="left",
            fontsize=14,
            color="black",
        )

    ax.set_yticks([])
    ax.set_xlabel("Count")
    ax.set_title("Top 10 Dismissal Reasons", fontsize=18)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, axis="x", alpha=0.25, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out_dir = _ensure_plots_dir()
    out_path = os.path.join(out_dir, "../../plots/top_dismissal_reasons.pdf")
    plt.savefig(out_path)
    plt.show()
    plt.close(fig)
    print(f"Saved plot: {out_path}")


async def run():
    await db.connect_maybe_initialize()
    reasons = await db.get_dismissed_reasons()
    plot_top_dismissals(dict(reasons))


if __name__ == "__main__":
    asyncio.run(run())
