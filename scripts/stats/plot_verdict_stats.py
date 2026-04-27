"""Reads all verdicts from the database, computes, and plots statistics.

What is plotted:
1) The distribution of tendencies for each single property in a 2x3 grid of subplots
   (six properties: clarity, veracity, context_coverage, intent, authenticity, contextualization).
2) Per property, the distribution of the per-claim tendency standard deviation.
3) Per property, the distribution of the maximum tendency difference per claim.

Run this script manually (requires DB access configured in config/globals.yaml):
    python -m scripts.plot_verdict_stats
"""

import asyncio
import os
from collections import defaultdict, Counter
from statistics import stdev
from typing import Iterable

import numpy as np

from veritas.common.annotation.rating import Rating, RatingAggregated, map_3_bins, map_7_bins, Category3Bin, \
    Category7Bin
from veritas.common.verdict import Verdict
from veritas.db import db
from scripts.stats.common import COLORS, _ensure_plots_dir

# Properties to consider from the top-level Verdict
CLAIM_LEVEL_PROPERTIES = [
    "veracity",
    "context_coverage",
]
MEDIA_PROPERTIES = ["authenticity", "contextualization"]


def _label_to_tendency(label: Rating | None) -> float | None:
    if label is None:
        return None
    return float(label.score)


def _extract_tendencies(
        verdicts: Iterable[Verdict],
) -> dict[str, list[list[float]]]:
    """Collect raw tendencies per property across all verdicts.

    Includes medium-level properties under keys 'authenticity' and 'contextualization'.
    """
    per_prop: dict[str, list[list[float]]] = defaultdict(list)
    for v in verdicts:

        # Top-level properties
        for prop in CLAIM_LEVEL_PROPERTIES:
            rating: RatingAggregated = getattr(v, prop, None)
            if rating:
                per_prop[prop].append([rating.score])

        # Medium-level properties
        if v.media_verdicts:
            for mv in v.media_verdicts:
                for prop in MEDIA_PROPERTIES:
                    rating = getattr(mv, prop, None)
                    if rating:
                        per_prop[prop].append([rating.score])

        # Integrity (single score derived from decisive property)
        try:
            per_prop["integrity"].append([v.integrity.score])
        except Exception:
            pass

        # All properties combined
        all_scores = []
        for prop in CLAIM_LEVEL_PROPERTIES:
            rating = getattr(v, prop, None)
            if rating:
                all_scores.append(rating.score)
        if v.media_verdicts:
            for mv in v.media_verdicts:
                for prop in MEDIA_PROPERTIES:
                    rating = getattr(mv, prop, None)
                    if rating:
                        all_scores.append(rating.score)
        if all_scores:
            per_prop["all"].append(all_scores)

    return per_prop


def _group_by_claim(verdicts: Iterable[Verdict]) -> dict[int, list[Verdict]]:
    by_claim: dict[int, list[Verdict]] = defaultdict(list)
    for v in verdicts:
        by_claim[v.claim_id].append(v)
    return by_claim


def _per_claim_stats(
        per_prop_and_claim_tendencies: dict[str, list[list[float]]]
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Compute, per property, per-claim tendency standard deviation and max-diff.

    Returns (per_prop_stddevs, per_prop_maxdiffs).
    """
    per_prop_variances: dict[str, list[float]] = {}
    per_prop_maxdiffs: dict[str, list[float]] = {}

    for prop, per_claim_lists in per_prop_and_claim_tendencies.items():
        variances: list[float] = []
        maxdiffs: list[float] = []

        for vals in per_claim_lists:
            # Clean values: keep finite floats only
            clean = [float(x) for x in vals if x is not None]
            # Filter non-finite (nan, inf)
            clean = [x for x in clean if np.isfinite(x)]

            if not clean:
                # No data for this claim instance
                variances.append(0.0)
                maxdiffs.append(0.0)
                continue

            # Max difference across model tendencies for this claim/media
            maxdiffs.append(max(clean) - min(clean))

            # Sample standard deviation requires at least 2 data points; fall back to 0.0 otherwise
            if len(clean) >= 2:
                try:
                    variances.append(float(stdev(clean)))
                except Exception:
                    variances.append(0.0)
            else:
                variances.append(0.0)

        per_prop_variances[prop] = variances
        per_prop_maxdiffs[prop] = maxdiffs

    return per_prop_variances, per_prop_maxdiffs


def _plot_properties_hist_grid(
        per_prop_values: dict[str, list[float]],
        title: str,
        filename: str,
        *,
        xticks: np.ndarray,
        xlim: tuple[float, float],
        xlabel: str,
        ylabel: str,
        gradient_colors: tuple[str, str, str],  # (neg, mid, pos)
        neutral_value: float,
):
    """Shared plotting routine used by plot_tendency_dist and plot_disagreement_dist.

    Parameters:
    - xticks: positions of bar centers and histogram bins.
    - xlim: axis limits.
    - gradient_colors: hex colors for (neg, mid, pos) ends of the gradient.
    - neutral_value: center point for the gradient (e.g., 0 for tendencies).
    - scale: multiply input values before binning (default 3 to map [-1,1] -> [-3,3]).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required to generate plots. Please install it: pip install matplotlib")
        return

    if not per_prop_values:
        print(f"No data available for: {title}")
        return

    order = [
        "authenticity",
        "contextualization",
        "veracity",
        "context_coverage",
        "integrity",
        "all",
    ]

    # Lazy import to get human-readable property names
    try:
        from veritas.common.annotation.property import PROPERTIES
    except Exception:
        PROPERTIES = {}

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True, dpi=300)
    axes = axes.flatten()

    def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))  # type: ignore

    def _interp(c1, c2, t: float):
        t = max(0.0, min(1.0, float(t)))
        return (c1[0] + (c2[0] - c1[0]) * t,
                c1[1] + (c2[1] - c1[1]) * t,
                c1[2] + (c2[2] - c1[2]) * t)

    NEG = _hex_to_rgb01(gradient_colors[0])
    MID = _hex_to_rgb01(gradient_colors[1])
    POS = _hex_to_rgb01(gradient_colors[2])

    min_tick = float(xticks[0])
    max_tick = float(xticks[-1])

    def color_for_value(x: float):
        x = max(min_tick, min(max_tick, x))
        if x <= neutral_value:
            if neutral_value == min_tick:
                return MID
            t = (neutral_value - x) / (neutral_value - min_tick)
            return _interp(MID, NEG, t)
        else:
            if max_tick == neutral_value:
                return MID
            t = (x - neutral_value) / (max_tick - neutral_value)
            return _interp(MID, POS, t)

    from fractions import Fraction
    def format_fraction(x: float) -> str:
        # Round to nearest 1/3
        val = round(x * 3) / 3
        if val == 0:
            return "0"
        frac = Fraction(val).limit_denominator(3)
        if frac.denominator == 1:
            return str(frac.numerator)
        return f"{frac.numerator}/{frac.denominator}"

    for i, prop in enumerate(order):
        ax = axes[i]
        ax.set_axisbelow(True)
        ax.grid(True, which="both", axis="both", alpha=0.25, linestyle="--", linewidth=0.5)

        values = per_prop_values.get(prop, []) or []
        data = np.asarray(values, dtype=float)
        data = data[np.isfinite(data)] if data.size else data
        if data.size > 0:
            if len(xticks) in (3, 7) and xticks[0] < 0:
                counts = compute_hist(data, bins=len(xticks))
                plot_xticks = xticks
            else:
                # Custom binning for continuous values
                step = xticks[1] - xticks[0] if len(xticks) > 1 else 0.33
                bins = np.concatenate([xticks - step/2, [xticks[-1] + step/2]])
                counts, _ = np.histogram(data, bins=bins)
                plot_xticks = xticks

            bar_colors = [color_for_value(float(c)) for c in plot_xticks]
            # Use a smaller width to avoid overlap (step is ~0.33)
            # Default width is 0.8 which is too much.
            step = plot_xticks[1] - plot_xticks[0] if len(plot_xticks) > 1 else 0.33
            ax.bar(plot_xticks, counts, color=bar_colors, edgecolor="white", align='center', width=step * 0.8)

        PRETTY_NAMES = {"integrity": "Integrity", "all": "All Properties"}
        pretty = PRETTY_NAMES.get(prop) or (PROPERTIES.get(prop).name if PROPERTIES.get(prop) else prop)
        ax.set_title(pretty)
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.set_xticklabels([format_fraction(x) for x in xticks])
        if i % 3 != 0:
            ax.set_ylabel("")
        if i < 3:
            ax.set_xlabel("")

    # Hide any unused axes
    for j in range(len(order), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=18)
    fig.supxlabel(xlabel, y=0.04)
    fig.supylabel(ylabel, x=0.04)
    fig.tight_layout(rect=(0.04, 0.03, 1, 0.99))

    out_dir = _ensure_plots_dir()
    out_path = os.path.join(out_dir, filename)
    plt.show()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def compute_hist(scores: Iterable[float], bins: int = 3) -> list[float]:
    """Turns scores into a histogram of counts in either 3 or 7 bins."""
    assert bins in (3, 7), f"Invalid number of bins: {bins}"
    binning_fn = map_3_bins if bins == 3 else map_7_bins
    categories = [binning_fn(score) for score in scores]
    counter = Counter(categories)
    unique_categories = list(Category3Bin) if bins == 3 else list(Category7Bin)
    return [counter.get(cat, 0) for cat in unique_categories]


def plot_score_dist(per_prop_values: dict[str, list[float]], title: str, filename: str):
    """Plot properties as separate histograms in a single figure.

    The plotting code is delegated to a shared helper and uses a red-grey-green
    gradient across the range [-3, 3] (after scaling by 3).
    """
    xticks = np.array([cat.value for cat in Category7Bin])
    _plot_properties_hist_grid(
        per_prop_values,
        title,
        filename,
        xticks=xticks,
        xlim=(-1.2, 1.2),
        xlabel="Score",
        ylabel="Count",
        gradient_colors=(COLORS["negative"], COLORS["neutral"], COLORS["positive"]),
        neutral_value=0.0,
    )


def plot_disagreement_dist(per_prop_values: dict[str, list[float]], title: str, filename: str):
    """Plot six properties as separate histograms in a single figure (2x3) for disagreement metrics.

    Expects per-prop arrays of non-negative values (e.g., standard deviation or max difference).
    Uses its own color gradient and x-ticks.
    """
    _plot_properties_hist_grid(
        per_prop_values,
        title,
        filename,
        xticks=np.arange(0, 2.3, step=1 / 3),
        xlim=(-0.2, 2.2),
        xlabel="Maximum score difference among ensemble predictions",
        ylabel="Count",
        # Custom gradient: from grey at 0 to blue at high disagreement
        gradient_colors=(COLORS["neutral"], COLORS["neutral"], COLORS["orange"]),
        neutral_value=0.0,
    )


async def run(only_released: bool = False):
    await db.connect_maybe_initialize()
    verdicts = await db.get_verdicts(only_released=only_released)

    per_prop_and_claim_tendencies = _extract_tendencies(verdicts)

    # 1) Per-property distributions in a single figure
    per_prop_tendencies = {prop: flatten(t) for prop, t in per_prop_and_claim_tendencies.items()}
    plot_score_dist(
        per_prop_tendencies,
        title="Distribution of Scores",
        filename="score_distributions.pdf",
    )

    # 2) Per-claim max difference distributions per property
    per_prop_variances, per_prop_maxdiffs = _per_claim_stats(per_prop_and_claim_tendencies)
    plot_disagreement_dist(
        per_prop_maxdiffs,
        title="Ensemble Agreement",
        filename="ensemble_agreement.pdf",
    )


def flatten(nested: list[list]) -> list:
    return [item for sublist in nested for item in sublist]


if __name__ == "__main__":
    asyncio.run(run(only_released=True))
