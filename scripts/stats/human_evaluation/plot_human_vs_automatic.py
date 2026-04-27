"""Generate scatter plots and confusion matrices comparing human vs. automatic annotations.

This script reads comparison_results.json and creates scatter plots and confusion
matrices showing the relationship between human and automated annotations for each property.

What it does:
1) Loads comparison data from the DB
2) Extracts human and automated tendencies for each property
3) Creates two types of plots:
   - Combined plot showing all properties on one scatter plot
   - Individual plots for each property
4) Adds diagonal reference lines:
   - Solid line for perfect alignment (y=x)
   - Dashed lines for intermediate steps (±1/3, ±2/3)
5) Creates confusion matrices by discretizing tendencies:
   - [-1.0, -0.33) → -1 (negative)
   - [-0.33, 0.33] → 0 (neutral)
   - (0.33, 1.0] → 1 (positive)

Output:
    - human_vs_automatic_combined.pdf (all properties scatter plot)
    - human_vs_automatic_<property>.pdf (individual scatter plots)
    - confusion_matrix_7_combined.pdf (7-bucket combined confusion matrix)
    - confusion_matrix_7_<property>.pdf (7-bucket individual confusion matrices)
    - confusion_matrix_3_combined.pdf (3-bucket combined confusion matrix)
    - confusion_matrix_3_<property>.pdf (3-bucket individual confusion matrices)
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from sklearn.metrics import confusion_matrix

from ezmm import MultimodalSequence
from scripts.stats.common import COLORS, _ensure_plots_dir
from veritas.common.annotation.rating import Rating, RatingAggregated
from veritas.common.verdict import MediumVerdict, Verdict
from veritas.db.annotation_db import annotation_db
from veritas.db.veritas_db import db
from veritas.metric import calculate_metrics_for_property

# Property names and display labels
CLAIM_LEVEL_PROPERTIES = [
    "veracity",
    "context_coverage",
]

MEDIA_PROPERTIES = ["authenticity", "contextualization"]

ALL_PROPERTIES = MEDIA_PROPERTIES + CLAIM_LEVEL_PROPERTIES + ["integrity"]

PROPERTY_LABELS = {
    "veracity": "Veracity",
    "context_coverage": "Context Coverage",
    "authenticity": "Authenticity",
    "contextualization": "Contextualization",
    "intent": "Intent",
    "integrity": "Integrity",
}

# Colors for each property from common.py
PROPERTY_COLORS = {
    "veracity": COLORS["orange"],
    "context_coverage": COLORS["blue"],
    "intent": COLORS["positive"],
    "authenticity": COLORS["light_orange"],
    "contextualization": COLORS["darkblue"],
    "integrity": COLORS["negative"],
}


async def fetch_verdicts(no_agreement_filtering: bool = False) -> list[tuple[int, Verdict, Verdict]]:
    """Fetch (claim_id, human_verdict, automated_verdict) triples from DBs.

    Args:
        no_agreement_filtering: If True, also includes claims that were dismissed
            for low verdict agreement.

    Returns:
        List of (claim_id, human_verdict, automated_verdict) tuples
    """
    # 1. Load human ratings
    print("Fetching human ratings...")
    claim_ratings = await annotation_db.get_completed_ratings()
    media_ratings = await annotation_db.get_completed_media_ratings()

    # 2. Load automated verdicts
    print("Fetching automated verdicts...")
    verdicts = await db.get_verdicts()

    verdict_by_claim = {v.claim_id: v for v in verdicts}

    # 3. Group human ratings by claim
    # claim_id -> {prop: [Rating]}
    human_claim_groups: dict[int, dict[str, list[Rating]]] = {}
    for r in claim_ratings:
        cid = r['claim_id']
        if cid not in human_claim_groups:
            human_claim_groups[cid] = {p: [] for p in CLAIM_LEVEL_PROPERTIES}
        for prop, rating in r['ratings'].items():
            if prop in human_claim_groups[cid]:
                human_claim_groups[cid][prop].append(rating)

    # claim_id -> media_id -> {prop: [Rating], "claim_data": str}
    human_media_groups: dict[int, dict[int, dict[str, list[Rating] | str]]] = {}
    for r in media_ratings:
        cid = r['claim_id']
        mid = r['media_id']
        if cid not in human_media_groups:
            human_media_groups[cid] = {}
        if mid not in human_media_groups[cid]:
            human_media_groups[cid][mid] = {p: [] for p in MEDIA_PROPERTIES}
            human_media_groups[cid][mid]["claim_data"] = r["claim_data"]
        for prop, rating in r['ratings'].items():
            if prop in human_media_groups[cid][mid]:
                human_media_groups[cid][mid][prop].append(rating)

    # 4. Create Verdict objects
    results: list[tuple[int, Verdict, Verdict]] = []

    # We iterate over all claims that have both human and automated annotations
    all_claim_ids = (set(human_claim_groups.keys()) | set(human_media_groups.keys())) & set(verdict_by_claim.keys())

    for cid in all_claim_ids:
        auto_verdict = verdict_by_claim[cid]

        # Ignore claims that were dismissed by the VeriTaS pipeline
        claim = await auto_verdict.claim
        if claim.dismissed:
            # But if the claim was dismissed for low agreement, we may still want to include it in the results
            if not no_agreement_filtering or not claim.dismissed_reason == "Verdict agreement too low.":
                continue

        # Aggregate human ratings for claim-level properties
        h_props = {}
        if cid in human_claim_groups:
            for prop, ratings in human_claim_groups[cid].items():
                if ratings:
                    h_props[prop] = RatingAggregated(individual_ratings=ratings, rater="human")

        # Aggregate human ratings for media-level properties
        h_media_verdicts = []
        if cid in human_media_groups:
            for mid, group_data in human_media_groups[cid].items():
                h_media_props = {}
                for prop in MEDIA_PROPERTIES:
                    ratings = group_data.get(prop)
                    if ratings:
                        h_media_props[prop] = RatingAggregated(individual_ratings=ratings, rater="human")

                if "authenticity" in h_media_props and "contextualization" in h_media_props:
                    # Determine media reference from claim data
                    claim_data = group_data["claim_data"]
                    media_items = MultimodalSequence(claim_data).unique_items()
                    reference = None
                    for item in media_items:
                        if item.id == mid:
                            reference = item.reference
                            break

                    h_media_verdicts.append(MediumVerdict(
                        reference=reference,
                        authenticity=h_media_props["authenticity"],
                        contextualization=h_media_props["contextualization"]
                    ))

        try:
            human_verdict = Verdict(
                claim_id=cid,
                review_ids=set(),
                veracity=h_props.get("veracity"),
                context_coverage=h_props.get("context_coverage"),
                intent=h_props.get("intent"),
                media_verdicts=h_media_verdicts
            )
            results.append((cid, human_verdict, auto_verdict))
        except Exception:
            continue

    return results


def filter_verdicts(
        data_points: list[tuple[int, Verdict, Verdict]],
        exclude_single_human_ratings: bool = True,
        exclude_human_disagreement: bool = True
) -> list[tuple[int, Verdict, Verdict]]:
    """Apply filters to the data points.
    
    Filters:
    - Only human annotations with at least two ratings are kept.
    - Exclude human annotations that have internal score difference of more than 1.
    """
    filtered = []
    for cid, hv, av in data_points:
        if exclude_single_human_ratings and hv.n_ratings < 2:
            continue
        if exclude_human_disagreement and not hv.sufficient_agreement:
            continue
        filtered.append((cid, hv, av))

    return filtered


def data_points_to_prop_dict(
        data_points: list[tuple[int, Verdict, Verdict]],
        llm: str = None,
) -> dict[str, list[tuple[RatingAggregated, RatingAggregated, str]]]:
    """Convert Verdict pairs into property-specific data points for plotting and stats."""
    result: dict[str, list[tuple[RatingAggregated, RatingAggregated, str]]] = {prop: [] for prop in ALL_PROPERTIES}

    def _get_a_agg(agg: RatingAggregated | None) -> RatingAggregated | None:
        if not agg or not llm:
            return agg

        gpt_ratings = [r for r in agg.individual_ratings if llm in r.rater.lower()]
        if not gpt_ratings:
            return None

        return RatingAggregated(individual_ratings=gpt_ratings, rater=llm)

    for cid, hv, av in data_points:
        # Claim level
        for prop in CLAIM_LEVEL_PROPERTIES:
            h_agg = getattr(hv, prop, None)
            a_agg = _get_a_agg(getattr(av, prop, None))
            if h_agg and a_agg:
                result[prop].append((h_agg, a_agg, f"claim:{cid}"))

        # Media level
        # We need to match media by ID
        h_media = {mv.medium.id: mv for mv in hv.media_verdicts}
        a_media = {mv.medium.id: mv for mv in av.media_verdicts}

        for mid in h_media.keys() & a_media.keys():
            h_mv = h_media[mid]
            a_mv = a_media[mid]
            for prop in MEDIA_PROPERTIES:
                h_agg = getattr(h_mv, prop, None)
                a_agg = _get_a_agg(getattr(a_mv, prop, None))
                if h_agg and a_agg:
                    result[prop].append((h_agg, a_agg, f"claim:{cid}, media:{mid}"))

        # Integrity
        # Integrity is a Rating, but we need RatingAggregated for type consistency in plotting functions
        # We wrap it in RatingAggregated with a single rating.
        h_integrity = hv.integrity
        a_integrity = av.integrity
        if h_integrity and a_integrity:
            # For auto integrity, we still need to apply llm if requested
            if llm:
                # If we want integrity of only one LLM, we should look at the decisive property's LLM rating.
                decisive_prop = av.compromising_property_name
                a_agg_decisive = _get_a_agg(getattr(av, decisive_prop, None))
                if a_agg_decisive:
                    a_integrity_agg = RatingAggregated(individual_ratings=a_agg_decisive.individual_ratings,
                                                       rater=llm)
                else:
                    a_integrity_agg = None
            else:
                a_integrity_agg = RatingAggregated(individual_ratings=[a_integrity], rater="auto")

            if a_integrity_agg:
                result["integrity"].append((
                    RatingAggregated(individual_ratings=[h_integrity], rater="human"),
                    a_integrity_agg,
                    f"claim:{cid}"
                ))

    return result


def add_reference_lines(ax):
    """Add diagonal reference lines to show alignment and deviations."""
    # Perfect alignment (y = x)
    ax.plot([-1, 1], [-1, 1], 'k-', linewidth=2, alpha=0.8, label='Perfect alignment', zorder=1)

    # Intermediate steps (±1/3, ±2/3)
    steps = [1 / 3, 2 / 3]
    for step in steps:
        # Positive offset (y = x + step)
        ax.plot([-1, 1 - step], [-1 + step, 1], 'k--', linewidth=1, alpha=0.3, zorder=1)
        # Negative offset (y = x - step)
        ax.plot([-1 + step, 1], [-1, 1 - step], 'k--', linewidth=1, alpha=0.3, zorder=1)


def plot_combined(data_points: dict[str, list[tuple[RatingAggregated, RatingAggregated, str]]], output_path: str):
    """Create a combined scatter plot with all properties."""
    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

    # Add reference lines first (so they're behind the points)
    add_reference_lines(ax)

    # Plot data points for each property
    for prop in ALL_PROPERTIES:
        points = data_points.get(prop, [])
        if not points:
            continue

        human_vals = [p[0].score for p in points]
        auto_vals = [p[1].score for p in points]

        ax.scatter(
            human_vals,
            auto_vals,
            color=PROPERTY_COLORS[prop],
            label=PROPERTY_LABELS[prop],
            alpha=0.6,
            s=100,
            edgecolors='white',
            linewidths=0.5,
            zorder=2
        )

    # Configure plot
    ax.set_xlabel("Human Annotation (tendency)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Automatic Annotation (tendency)", fontsize=12, fontweight='bold')
    ax.set_title("Human vs. Automatic Annotations - All Properties", fontsize=14, fontweight='bold', pad=20)

    # Set axis limits and ticks
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(np.arange(-1, 1.1, 0.5))
    ax.set_yticks(np.arange(-1, 1.1, 0.5))

    # Add grid
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Add legend
    ax.legend(loc='upper left', framealpha=0.9, fontsize=10)

    # Make it square
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"Saved combined plot to {output_path}")


def plot_individual(prop: str, data_points: list[tuple[RatingAggregated, RatingAggregated, str]], output_path: str):
    """Create a scatter plot for a single property."""
    if not data_points:
        print(f"No data points for {prop}, skipping")
        return

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)

    # Add reference lines first
    add_reference_lines(ax)

    # Extract values
    human_vals = [p[0].score for p in data_points]
    auto_vals = [p[1].score for p in data_points]

    # Plot data points
    ax.scatter(
        human_vals,
        auto_vals,
        color=PROPERTY_COLORS[prop],
        alpha=0.6,
        s=80,
        edgecolors='white',
        linewidths=1,
        zorder=2
    )

    # Calculate correlation
    if len(data_points) > 1:
        corr = np.corrcoef(human_vals, auto_vals)[0, 1]
        # Add correlation to title
        title = f"{PROPERTY_LABELS[prop]} - Human vs. Automatic\n(n={len(data_points)}, r={corr:.3f})"
    else:
        title = f"{PROPERTY_LABELS[prop]} - Human vs. Automatic\n(n={len(data_points)})"

    # Configure plot
    ax.set_xlabel("Human Annotation (tendency)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Automatic Annotation (tendency)", fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Set axis limits and ticks
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(np.arange(-1, 1.1, 0.5))
    ax.set_yticks(np.arange(-1, 1.1, 0.5))

    # Add grid
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Add legend for reference lines
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc='upper left', framealpha=0.9, fontsize=10)

    # Make it square
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"Saved {prop} plot to {output_path}")


def create_confusion_matrix_3(data_points: list[tuple[RatingAggregated, RatingAggregated, str]]) -> tuple[
    np.ndarray, list[int], list[int]]:
    """Create 3x3 confusion matrix from continuous tendency pairs.

    Args:
        data_points: List of (human_rating, auto_rating, id) tuples

    Returns:
        Tuple of (confusion_matrix, human_discretized, auto_discretized)
    """
    if not data_points:
        return np.zeros((3, 3)), [], []

    human_vals = [p[0].as_3_bin().value for p in data_points]
    auto_vals = [p[1].as_3_bin().value for p in data_points]

    # Create confusion matrix with labels [-1, 0, 1]
    cm = confusion_matrix(human_vals, auto_vals, labels=[-1, 0, 1])

    return cm, human_vals, auto_vals


def create_confusion_matrix_7(data_points: list[tuple[RatingAggregated, RatingAggregated, str]]) -> tuple[
    np.ndarray, list[int], list[int]]:
    """Create 7x7 confusion matrix from continuous tendency pairs using Veritas buckets.

    Args:
        data_points: List of (human_rating, auto_rating, id) tuples

    Returns:
        Tuple of (confusion_matrix, human_discretized_indices, auto_discretized_indices)
    """
    if not data_points:
        return np.zeros((7, 7), dtype=int), [], []

    # Get fine-grained categories
    human_cats = [p[0].as_7_bin() for p in data_points]
    auto_cats = [p[1].as_7_bin() for p in data_points]

    # Create confusion matrix with 7 labels from CategoryFineGrained
    from veritas.common.annotation.rating import Category7Bin
    labels = list(Category7Bin)

    # Create a mapping from enum to integer index
    label_to_idx = {label: i for i, label in enumerate(labels)}

    # Convert values to indices
    human_indices = [label_to_idx[v] for v in human_cats]
    auto_indices = [label_to_idx[v] for v in auto_cats]

    # Create confusion matrix using indices
    cm = confusion_matrix(human_indices, auto_indices, labels=list(range(7)))

    return cm, human_indices, auto_indices


def plot_confusion_matrix_3(cm: np.ndarray, title: str, output_path: str, n_samples: int = None):
    """Plot a 3x3 confusion matrix with distance-based coloring and value-based saturation."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

    # Labels for 3 buckets
    labels = ['-1', '0', '+1']
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(labels, fontsize=18)
    ax.set_yticklabels(labels, fontsize=18)

    # Labels
    ax.set_xlabel('Automatic Annotation', fontsize=24)
    ax.set_ylabel('Human Annotation', fontsize=24)
    ax.set_title(title, fontsize=28, pad=20)

    max_count = cm.max() if cm.max() > 0 else 1

    pos_rgb = np.array(mcolors.to_rgb(COLORS["positive"]))
    neu_rgb = np.array(mcolors.to_rgb(COLORS["neutral"]))
    neg_rgb = np.array(mcolors.to_rgb(COLORS["negative"]))

    for i in range(3):
        for j in range(3):
            count = int(cm[i, j])

            # Distance from diagonal (0, 1, or 2)
            distance = abs(i - j)

            if distance == 0:
                base_rgb = pos_rgb
            elif distance == 1:
                base_rgb = neu_rgb
            else:
                base_rgb = neg_rgb

            # Scale saturation based on count
            alpha = count / max_count

            # Draw a rectangle for each cell
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=base_rgb, alpha=alpha)
            ax.add_patch(rect)

            # Choose text color
            luminance = 0.299 * base_rgb[0] + 0.587 * base_rgb[1] + 0.114 * base_rgb[2]
            if alpha > 0.5 and luminance < 0.5:
                text_color = 'white'
            else:
                text_color = 'black'

            # Display count
            if count > 0:
                ax.text(j, i, str(count), ha='center', va='center',
                        color=text_color, fontsize=22, fontweight='bold')

    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(2.5, -0.5)  # Invert y axis to match confusion matrix convention

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"Saved 3x3 confusion matrix to {output_path}")


def plot_confusion_matrix_7(cm: np.ndarray, title: str, output_path: str, n_samples: int = None):
    """Plot a 7x7 confusion matrix with distance-based coloring and value-based saturation."""
    fig, ax = plt.subplots(figsize=(11, 10), dpi=300)

    # Set ticks and labels for 7 buckets
    labels = [
        '-1',
        '-2/3',
        '-1/3',
        '0',
        '+1/3',
        '+2/3',
        '+1'
    ]
    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    ax.set_xticklabels(labels, fontsize=18)
    ax.set_yticklabels(labels, fontsize=18)

    # Labels
    ax.set_xlabel('Automatic Annotation', fontsize=24)
    ax.set_ylabel('Human Annotation', fontsize=24)
    ax.set_title(title, fontsize=28, pad=20)

    max_count = cm.max() if cm.max() > 0 else 1

    pos_rgb = np.array(mcolors.to_rgb(COLORS["positive"]))
    neu_rgb = np.array(mcolors.to_rgb(COLORS["neutral"]))
    neg_rgb = np.array(mcolors.to_rgb(COLORS["negative"]))

    for i in range(7):
        for j in range(7):
            count = int(cm[i, j])

            # Distance from diagonal (0 to 6)
            distance = abs(i - j) / 3

            # Interpolate color based on distance from diagonal
            # distance 0 -> positive
            # distance 1 -> neutral (as requested: "For distance = 1, the color should be 'neutral'")
            # distance >= 2 -> negative (as requested: "for distance = 2, it should be 'negative'")
            if 0 <= distance <= 1:
                base_rgb = pos_rgb * (1 - distance) + neu_rgb * distance
            else:
                base_rgb = neu_rgb * (2 - distance) + neg_rgb * (distance - 1)

            # Scale saturation based on count
            alpha = count / max_count

            if count > 0 or True:  # Always draw cells to show the background grid if desired, but here we just draw
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=base_rgb, alpha=alpha)
                ax.add_patch(rect)

            # Choose text color
            luminance = 0.299 * base_rgb[0] + 0.587 * base_rgb[1] + 0.114 * base_rgb[2]
            if alpha > 0.5 and luminance < 0.5:
                text_color = 'white'
            else:
                text_color = 'black'

            # Display count
            if count > 0:
                ax.text(j, i, str(count), ha='center', va='center',
                        color=text_color, fontsize=18, fontweight='bold')

    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(6.5, -0.5)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"Saved 7x7 confusion matrix to {output_path}")


def print_statistics(data_points: dict[str, list[tuple[RatingAggregated, RatingAggregated, str]]],
                     title: str = "SUMMARY STATISTICS"):
    """Print summary statistics for each property using calculate_metrics_for_property."""
    print("\n" + "=" * 60)
    print(title.upper())
    print("=" * 60)

    metric_keys = ["mse", "mae", "acc_3bin", "acc_7bin"]
    metric_labels = {"mse": "MSE", "mae": "MAE", "acc_3bin": "Acc (3-bin)", "acc_7bin": "Acc (7-bin)"}

    # Transposed table: rows = properties, columns = metrics
    header = f"| {'Property':<20} | {'N':>5} | " + " | ".join([f"{metric_labels[k]:>12}" for k in metric_keys]) + " |"
    separator = f"|:{'-' * 20}-|{'-' * 5}:-|-" + "-|-".join(['-' * 12 for _ in metric_keys]) + "-|"
    print(header)
    print(separator)

    # Collect all ratings for the "Total" row (excluding integrity, as per calculate_metrics)
    all_auto_ratings: list[RatingAggregated] = []
    all_human_ratings: list[RatingAggregated] = []

    for prop in ALL_PROPERTIES:
        points = data_points.get(prop, [])
        if not points:
            row = f"| {PROPERTY_LABELS.get(prop, prop):<20} | {'N/A':>5} | " + " | ".join([f"{'N/A':>12}" for _ in metric_keys]) + " |"
            print(row)
            continue

        human_ratings = [p[0] for p in points]
        auto_ratings = [p[1] for p in points]

        result = calculate_metrics_for_property(auto_ratings, human_ratings)

        n = len(points)
        if result is None:
            row = f"| {PROPERTY_LABELS.get(prop, prop):<20} | {n:>5} | " + " | ".join([f"{'N/A':>12}" for _ in metric_keys]) + " |"
        else:
            row = f"| {PROPERTY_LABELS.get(prop, prop):<20} | {n:>5} | " + " | ".join([f"{result[k]:>12.3f}" for k in metric_keys]) + " |"

        print(row)

        # Accumulate for total (exclude integrity, matching calculate_metrics behaviour)
        if prop != "integrity":
            all_auto_ratings.extend(auto_ratings)
            all_human_ratings.extend(human_ratings)

    # Total row
    total_n = len(all_auto_ratings)
    total_result = calculate_metrics_for_property(all_auto_ratings, all_human_ratings) if total_n > 0 else None
    if total_result is None:
        row = f"| {'Total':<20} | {total_n:>5} | " + " | ".join([f"{'N/A':>12}" for _ in metric_keys]) + " |"
    else:
        row = f"| {'Total':<20} | {total_n:>5} | " + " | ".join([f"{total_result[k]:>12.3f}" for k in metric_keys]) + " |"
    print(row)

    print("\n" + "=" * 60)


async def main(
        exclude_single_human: bool,
        exclude_human_disagreement: bool,
        plot_scatter: bool,
        plot_cm_7: bool,
        plot_cm_3: bool,
        llm: str = None,
        no_agreement_filtering: bool = False
):
    # Initialize DBs
    await db.connect_maybe_initialize()
    await annotation_db.connect_maybe_initialize()

    # There is no "ensemble" anymore if we only consider one LLM's predictions
    if llm is not None:
        no_agreement_filtering = True

    # Extract data points from DB
    verdicts = await fetch_verdicts(no_agreement_filtering=no_agreement_filtering)

    # Apply filtering
    verdicts = filter_verdicts(
        verdicts,
        exclude_single_human_ratings=exclude_single_human,
        exclude_human_disagreement=exclude_human_disagreement
    )

    # Convert to property-specific dicts
    data = data_points_to_prop_dict(verdicts, llm=llm)

    # Print statistics
    print_statistics(data, title=f"SUMMARY STATISTICS" + f" ({llm} only)" if llm else "")

    # Determine output directory
    output_dir = Path(_ensure_plots_dir()) / "human_vs_automatic"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate scatter plots
    if plot_scatter:
        print(f"\nGenerating scatter plots...")
        combined_path = output_dir / f"human_vs_automatic_combined.pdf"
        plot_combined(data, str(combined_path))

        for prop in ALL_PROPERTIES:
            if data[prop]:
                output_path = output_dir / f"human_vs_automatic_{prop}.pdf"
                plot_individual(prop, data[prop], str(output_path))

    # Generate confusion matrices (7 buckets - Veritas pipeline)
    if plot_cm_7:
        print(f"\nGenerating 7-bucket confusion matrices...")

        all_points = []
        for prop in ALL_PROPERTIES:
            all_points.extend(data[prop])

        if all_points:
            cm_combined_7, _, _ = create_confusion_matrix_7(all_points)
            cm_combined_path_7 = output_dir / f"confusion_matrix_7_combined.pdf"
            plot_confusion_matrix_7(
                cm_combined_7,
                f"Confusion Matrix (all properties)",
                str(cm_combined_path_7),
                n_samples=len(all_points)
            )

        for prop in ALL_PROPERTIES:
            if data[prop]:
                cm7, _, _ = create_confusion_matrix_7(data[prop])
                output_path_7 = output_dir / f"confusion_matrix_7_{prop}.pdf"
                plot_confusion_matrix_7(
                    cm7,
                    f"Confusion Matrix ({PROPERTY_LABELS[prop]})",
                    str(output_path_7),
                    n_samples=len(data[prop])
                )

    # Generate 3-bucket confusion matrices
    if plot_cm_3:
        if all_points:
            cm_combined_3, _, _ = create_confusion_matrix_3(all_points)
            cm_combined_path_3 = output_dir / f"confusion_matrix_3_combined.pdf"
            plot_confusion_matrix_3(
                cm_combined_3,
                f"Confusion Matrix (all properties)",
                str(cm_combined_path_3),
                n_samples=len(all_points)
            )

        for prop in ALL_PROPERTIES:
            if data[prop]:
                cm3, _, _ = create_confusion_matrix_3(data[prop])
                output_path_3 = output_dir / f"confusion_matrix_3_{prop}.pdf"
                plot_confusion_matrix_3(
                    cm3,
                    f"Confusion Matrix ({PROPERTY_LABELS[prop]})",
                    str(output_path_3),
                    n_samples=len(data[prop])
                )


if __name__ == "__main__":
    asyncio.run(main(
        exclude_single_human=True,
        exclude_human_disagreement=True,
        plot_scatter=False,
        plot_cm_7=False,
        plot_cm_3=False,
        llm=None,
        no_agreement_filtering=False
    ))
