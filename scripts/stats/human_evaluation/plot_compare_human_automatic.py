#!/usr/bin/env python3
"""
Compare human annotations vs automatic annotations (ensemble + individual models).

This script:
1. Loads human annotations and verdicts directly from the database
2. Aggregates human annotations per claim (similar to ensemble aggregation)
3. Compares ensemble vs humans and each individual model vs humans
4. Generates tables and bar charts for each metric (MSE, MAE, ACC)
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from veritas.common.annotation.rating import Rating, RatingAggregated
from veritas.db.annotation_db import annotation_db
from veritas.db.veritas_db import db
from veritas.metric import mse, mae, acc_3bin


# Properties to evaluate
CLAIM_PROPERTIES = ["clarity", "veracity", "context_coverage", "intent"]
MEDIA_PROPERTIES = ["media_authenticity", "media_contextualization"]


@dataclass
class ComparisonResult:
    """Results for comparing one system (ensemble or model) vs humans."""
    system_name: str  # "ensemble" or model name
    property_name: str
    mse_score: float
    mae_score: float
    acc_score: float
    n_samples: int


def annotation_value_to_score(value: str, confidence: int) -> float:
    """
    Convert human annotation (value + confidence) to a score in [-1, 1].

    Args:
        value: The categorical value (e.g., "true", "false", "authentic", etc.)
        confidence: Confidence level (0-3)

    Returns:
        Score in [-1, 1] range
    """
    # Map value to direction
    positive_values = [
        "true", "clear", "complete", "authentic", "contextualized",
        "informative", "humorous", "promotional", "educational"
    ]
    negative_values = [
        "false", "unclear", "incomplete", "inauthentic", "decontextualized",
        "misleading", "harmful"
    ]

    value_lower = value.lower()

    if value_lower in positive_values:
        direction = 1
    elif value_lower in negative_values:
        direction = -1
    elif value_lower in ["unknown", "uncertain", "neutral"]:
        return 0.0
    else:
        # Unknown value type - treat as uncertain
        return 0.0

    # Map confidence (0-3) to certainty level
    # 3 (certain) -> 1.0
    # 2 (rather certain) -> 2/3
    # 1 (rather uncertain) -> 1/3
    # 0 (totally uncertain) -> 0.0
    certainty = confidence / 3.0

    return direction * certainty


async def load_human_annotations() -> dict[int, dict[str, list[Rating]]]:
    """
    Load all completed human annotations from the database.

    Returns:
        Dict mapping claim_id -> property_name -> list of Rating objects
    """
    # Get all completed annotations
    query = """
        SELECT
            a.id,
            a.claim_id,
            a.user_id,
            a.clarity,
            a.clarity_confidence,
            a.veracity,
            a.veracity_confidence,
            a.veracity_tags,
            a.context_coverage,
            a.context_coverage_confidence,
            a.intent,
            a.intent_confidence,
            a.intent_tags,
            u.code as username
        FROM annotations a
        JOIN annotation_users u ON a.user_id = u.id
        WHERE a.status = 'completed' AND a.dismissed = FALSE
        ORDER BY a.claim_id, a.user_id
    """

    rows = await annotation_db._fetch(query)

    # Organize by claim_id and property
    annotations_by_claim: dict[int, dict[str, list[Rating]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for row in rows:
        claim_id = row["claim_id"]
        username = row["username"]

        # Process claim-level properties
        for prop in CLAIM_PROPERTIES:
            value = row[prop]
            confidence = row[f"{prop}_confidence"]

            if value is not None and confidence is not None:
                score = annotation_value_to_score(value, confidence)
                tags = row.get(f"{prop}_tags", []) or []

                rating = Rating(
                    score=score,
                    rater=f"human:{username}",
                    explanation=None,
                    tags=tags if isinstance(tags, list) else []
                )
                annotations_by_claim[claim_id][prop].append(rating)

    # Load media evaluations
    media_query = """
        SELECT
            me.id,
            me.annotation_id,
            me.media_id,
            me.media_authenticity,
            me.media_authenticity_confidence,
            me.media_authenticity_tags,
            me.media_contextualization,
            me.media_contextualization_confidence,
            a.claim_id,
            u.code as username
        FROM media_evaluations me
        JOIN annotations a ON me.annotation_id = a.id
        JOIN annotation_users u ON a.user_id = u.id
        WHERE me.dismissed = FALSE AND a.status = 'completed'
        ORDER BY a.claim_id, me.media_id
    """

    media_rows = await annotation_db._fetch(media_query)

    for row in media_rows:
        claim_id = row["claim_id"]
        username = row["username"]

        for prop in MEDIA_PROPERTIES:
            value = row[prop]
            confidence = row[f"{prop}_confidence"]

            if value is not None and confidence is not None:
                score = annotation_value_to_score(value, confidence)
                tags = row.get(f"{prop}_tags", []) or []

                rating = Rating(
                    score=score,
                    rater=f"human:{username}",
                    explanation=None,
                    tags=tags if isinstance(tags, list) else []
                )
                annotations_by_claim[claim_id][prop].append(rating)

    return dict(annotations_by_claim)


async def load_verdicts() -> dict[int, Any]:
    """
    Load all verdicts from the database.

    Returns:
        Dict mapping claim_id -> Verdict object
    """
    verdicts = await db.get_verdicts()
    return {v.claim_id: v for v in verdicts}


def aggregate_human_ratings(ratings: list[Rating]) -> Rating:
    """
    Aggregate multiple human ratings into a single rating.
    Uses the same logic as RatingAggregated.

    Args:
        ratings: List of Rating objects from different annotators

    Returns:
        Aggregated Rating object
    """
    if not ratings:
        return Rating(score=0.0, rater="human_aggregated", explanation=None, tags=[])

    # Average scores
    avg_score = sum(r.score for r in ratings) / len(ratings)

    # Aggregate tags (50%+ consensus)
    tag_counts = defaultdict(int)
    for rating in ratings:
        for tag in rating.tags:
            tag_counts[tag] += 1

    threshold = len(ratings) / 2
    consensus_tags = [tag for tag, count in tag_counts.items() if count >= threshold]

    return Rating(
        score=avg_score,
        rater="human_aggregated",
        explanation=None,
        tags=consensus_tags
    )


def extract_automatic_ratings(
    verdict: Any, property_name: str
) -> tuple[Rating | None, list[Rating]]:
    """
    Extract ensemble rating and individual model ratings for a property.

    Args:
        verdict: Verdict object
        property_name: Name of the property

    Returns:
        Tuple of (ensemble_rating, individual_model_ratings)
    """
    # Handle media properties
    if property_name in MEDIA_PROPERTIES:
        # For media properties, we need to aggregate across all media
        all_ensemble_ratings = []
        all_individual_ratings = defaultdict(list)

        if hasattr(verdict, "media_verdicts") and verdict.media_verdicts:
            prop_suffix = property_name.replace("media_", "")

            for media_verdict in verdict.media_verdicts:
                rating_agg = getattr(media_verdict, prop_suffix, None)
                if rating_agg:
                    all_ensemble_ratings.append(
                        Rating(
                            score=rating_agg.score,
                            rater="ensemble",
                            explanation=rating_agg.explanation,
                            tags=rating_agg.tags
                        )
                    )

                    # Extract individual ratings
                    if hasattr(rating_agg, "individual_ratings"):
                        for ind_rating in rating_agg.individual_ratings:
                            all_individual_ratings[ind_rating.rater].append(
                                ind_rating
                            )

        # Aggregate ensemble ratings
        if all_ensemble_ratings:
            ensemble_rating = Rating(
                score=sum(r.score for r in all_ensemble_ratings) / len(all_ensemble_ratings),
                rater="ensemble",
                explanation=None,
                tags=[]
            )
        else:
            ensemble_rating = None

        # Aggregate individual model ratings
        individual_ratings = []
        for model_name, ratings in all_individual_ratings.items():
            avg_score = sum(r.score for r in ratings) / len(ratings)
            individual_ratings.append(
                Rating(score=avg_score, rater=model_name, explanation=None, tags=[])
            )

        return ensemble_rating, individual_ratings

    # Handle claim properties
    rating_agg = getattr(verdict, property_name, None)

    if not rating_agg:
        return None, []

    # Extract ensemble rating
    ensemble_rating = Rating(
        score=rating_agg.score,
        rater="ensemble",
        explanation=rating_agg.explanation,
        tags=rating_agg.tags
    )

    # Extract individual model ratings
    individual_ratings = []
    if hasattr(rating_agg, "individual_ratings"):
        individual_ratings = list(rating_agg.individual_ratings)

    return ensemble_rating, individual_ratings


async def compare_systems() -> list[ComparisonResult]:
    """
    Compare ensemble and individual models against human annotations.

    Returns:
        List of ComparisonResult objects
    """
    print("Loading data from database...")
    human_annotations = await load_human_annotations()
    verdicts = await load_verdicts()

    print(f"Loaded {len(human_annotations)} claims with human annotations")
    print(f"Loaded {len(verdicts)} verdicts")

    # Find claims that have both human annotations and verdicts
    common_claim_ids = set(human_annotations.keys()) & set(verdicts.keys())
    print(f"Found {len(common_claim_ids)} claims with both human and automatic annotations")

    results = []

    # Evaluate each property
    all_properties = CLAIM_PROPERTIES + MEDIA_PROPERTIES

    for prop in all_properties:
        print(f"\nEvaluating property: {prop}")

        # Collect predictions and targets for each system
        systems_data: dict[str, tuple[list[Rating], list[Rating]]] = defaultdict(
            lambda: ([], [])
        )

        for claim_id in common_claim_ids:
            # Get human ratings for this claim and property
            human_ratings = human_annotations[claim_id].get(prop, [])
            if not human_ratings:
                continue

            # Aggregate human ratings to create target
            target = aggregate_human_ratings(human_ratings)

            # Get automatic ratings
            verdict = verdicts[claim_id]
            ensemble_rating, individual_ratings = extract_automatic_ratings(verdict, prop)

            # Add ensemble comparison
            if ensemble_rating:
                predictions, targets = systems_data["ensemble"]
                predictions.append(ensemble_rating)
                targets.append(target)

            # Add individual model comparisons
            for ind_rating in individual_ratings:
                model_name = ind_rating.rater
                predictions, targets = systems_data[model_name]
                predictions.append(ind_rating)
                targets.append(target)

        # Calculate metrics for each system
        for system_name, (predictions, targets) in systems_data.items():
            if len(predictions) < 1:  # Need at least 1 sample
                continue

            mse_score = mse(predictions, targets)
            mae_score = mae(predictions, targets)
            acc_score = acc_3bin(predictions, targets)

            result = ComparisonResult(
                system_name=system_name,
                property_name=prop,
                mse_score=mse_score,
                mae_score=mae_score,
                acc_score=acc_score,
                n_samples=len(predictions)
            )
            results.append(result)

            print(f"  {system_name}: MSE={mse_score:.4f}, MAE={mae_score:.4f}, ACC={acc_score:.4f} (n={len(predictions)})")

    return results


def generate_tables(results: list[ComparisonResult]) -> None:
    """
    Generate and print tables for each metric.

    Args:
        results: List of ComparisonResult objects
    """
    # Group results by property
    results_by_property: dict[str, list[ComparisonResult]] = defaultdict(list)
    for result in results:
        results_by_property[result.property_name].append(result)

    metrics = ["mse_score", "mae_score", "acc_score"]
    metric_names = {"mse_score": "MSE", "mae_score": "MAE", "acc_score": "Accuracy"}

    for metric in metrics:
        print(f"\n{'='*80}")
        print(f"{metric_names[metric]} Comparison: Human vs Automatic Annotations")
        print(f"{'='*80}\n")

        for prop, prop_results in sorted(results_by_property.items()):
            # Sort by ensemble first, then by model name
            sorted_results = sorted(
                prop_results,
                key=lambda r: (r.system_name != "ensemble", r.system_name)
            )

            print(f"\nProperty: {prop.upper()}")
            print("-" * 70)
            print(f"{'System':<40} {metric_names[metric]:>15} {'Samples':>10}")
            print("-" * 70)

            for result in sorted_results:
                score = getattr(result, metric)
                print(f"{result.system_name:<40} {score:>15.4f} {result.n_samples:>10}")


def generate_charts(results: list[ComparisonResult]) -> None:
    """
    Generate bar charts for each metric.

    Args:
        results: List of ComparisonResult objects
    """
    # Group results by property
    results_by_property: dict[str, list[ComparisonResult]] = defaultdict(list)
    for result in results:
        results_by_property[result.property_name].append(result)

    if not results_by_property:
        print("\nNo results to generate charts for.")
        return

    metrics = ["mse_score", "mae_score", "acc_score"]
    metric_names = {"mse_score": "MSE", "mae_score": "MAE", "acc_score": "Accuracy"}
    metric_ranges = {
        "mse_score": (0, 4),   # Max: (1 - (-1))^2 = 4
        "mae_score": (0, 2),   # Max: |1 - (-1)| = 2
        "acc_score": (0, 1)    # Proportion in [0, 1]
    }

    for metric in metrics:
        # Create a figure with subplots for each property
        n_properties = len(results_by_property)
        n_cols = min(3, n_properties)
        n_rows = (n_properties + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_properties == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        fig.suptitle(f"{metric_names[metric]}: Human vs Automatic Annotations", fontsize=16)

        for idx, (prop, prop_results) in enumerate(sorted(results_by_property.items())):
            ax = axes[idx]

            # Sort by ensemble first, then by model name
            sorted_results = sorted(
                prop_results,
                key=lambda r: (r.system_name != "ensemble", r.system_name)
            )

            # Extract data
            systems = [r.system_name for r in sorted_results]
            scores = [getattr(r, metric) for r in sorted_results]

            # Shorten system names for display
            display_names = []
            for name in systems:
                if name == "ensemble":
                    display_names.append("Ensemble")
                elif ":" in name:
                    display_names.append(name.split(":")[-1][:20])  # Take model name part
                else:
                    display_names.append(name[:20])

            # Create bar chart
            colors = ["#2ecc71" if s == "ensemble" else "#3498db" for s in systems]
            bars = ax.bar(range(len(systems)), scores, color=colors)

            # Customize chart
            ax.set_xlabel("System", fontsize=10)
            ax.set_ylabel(metric_names[metric], fontsize=10)
            ax.set_title(prop.replace("_", " ").title(), fontsize=12)
            ax.set_xticks(range(len(systems)))
            ax.set_xticklabels(display_names, rotation=45, ha="right", fontsize=8)
            ax.set_ylim(metric_ranges[metric])  # Set proper y-axis range
            ax.grid(axis="y", alpha=0.3)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8
                )

        # Hide unused subplots
        for idx in range(len(results_by_property), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()

        # Save figure
        filename = f"human_vs_automatic_{metric}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"\nSaved chart: {filename}")

        plt.close()


async def main():
    """Main function."""
    print("Human vs Automatic Annotation Comparison")
    print("=" * 80)

    # Initialize databases
    print("\nConnecting to databases...")
    await annotation_db.connect_maybe_initialize()
    await db.connect_maybe_initialize()

    try:
        # Run comparison
        results = await compare_systems()

        # Generate outputs
        print("\n" + "=" * 80)
        print("GENERATING TABLES")
        print("=" * 80)
        generate_tables(results)

        print("\n" + "=" * 80)
        print("GENERATING CHARTS")
        print("=" * 80)
        generate_charts(results)

        print("\n" + "=" * 80)
        print("DONE!")
        print("=" * 80)

    finally:
        # Close database connections
        await annotation_db.close()
        await db.close()


if __name__ == "__main__":
    asyncio.run(main())
