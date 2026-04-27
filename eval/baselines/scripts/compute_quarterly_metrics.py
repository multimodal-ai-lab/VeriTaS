#!/usr/bin/env python3
"""Compute per-quarter metrics from baseline results CSV."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from eval.baselines.common.claims import get_claim_quarter
from eval.baselines.common.metrics import compute_metrics, print_metrics


def load_results_by_quarter(results_path: Path) -> dict[str, list[dict]]:
    """
    Load results CSV and group by quarter.

    Args:
        results_path: Path to the results CSV file.

    Returns:
        Dictionary mapping quarter strings to list of result rows.
    """
    results_by_quarter = defaultdict(list)

    with open(results_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            claim_id = int(row["claim_id"])
            quarter = get_claim_quarter(claim_id)
            results_by_quarter[quarter].append(row)

    return dict(results_by_quarter)


def compute_quarterly_metrics(results_path: Path) -> dict[str, dict]:
    """
    Compute metrics for each quarter.

    Args:
        results_path: Path to the results CSV file.

    Returns:
        Dictionary mapping quarter strings to metrics dictionaries.
    """
    results_by_quarter = load_results_by_quarter(results_path)

    quarterly_metrics = {}
    for quarter in sorted(results_by_quarter.keys(), key=lambda q: (int(q.split()[1]), int(q[1]))):
        rows = results_by_quarter[quarter]

        # Filter to successful predictions only
        successful_rows = [r for r in rows if r["status"] == "success"]

        if not successful_rows:
            quarterly_metrics[quarter] = {"error": "No successful predictions"}
            continue

        y_true = [r["gt_class"] for r in successful_rows]
        y_pred = [r["verdict"] for r in successful_rows]

        metrics = compute_metrics(y_true, y_pred)
        metrics["total_claims"] = len(rows)
        metrics["successful_claims"] = len(successful_rows)

        quarterly_metrics[quarter] = metrics

    return quarterly_metrics


def print_quarterly_metrics(quarterly_metrics: dict[str, dict]):
    """Print quarterly metrics summary."""
    print("\n" + "=" * 60)
    print("QUARTERLY METRICS SUMMARY")
    print("=" * 60)

    # Summary table header
    print(f"\n{'Quarter':<12} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12} {'N':>6}")
    print("-" * 52)

    for quarter, metrics in quarterly_metrics.items():
        if "error" in metrics:
            print(f"{quarter:<12} {'N/A':>10} {'N/A':>10} {'N/A':>12} {0:>6}")
        else:
            print(
                f"{quarter:<12} "
                f"{metrics['accuracy']:>10.2%} "
                f"{metrics['macro_f1']:>10.4f} "
                f"{metrics['weighted_f1']:>12.4f} "
                f"{metrics['successful_claims']:>6}"
            )

    # Detailed per-quarter breakdown
    for quarter, metrics in quarterly_metrics.items():
        if "error" not in metrics:
            print(f"\n{'=' * 40}")
            print(f"Quarter: {quarter}")
            print_metrics(metrics)


def main():
    parser = argparse.ArgumentParser(description="Compute per-quarter metrics from baseline results.")
    parser.add_argument("results_path", type=Path, help="Path to the results CSV file.")
    parser.add_argument("--output", "-o", type=Path, help="Optional path to save JSON output.")
    parser.add_argument("--summary-only", action="store_true", help="Only print summary table, not detailed metrics.")
    args = parser.parse_args()

    if not args.results_path.exists():
        raise FileNotFoundError(f"Results file not found: {args.results_path}")

    quarterly_metrics = compute_quarterly_metrics(args.results_path)

    if args.summary_only:
        print(f"\n{'Quarter':<12} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12} {'N':>6}")
        print("-" * 52)
        for quarter, metrics in quarterly_metrics.items():
            if "error" in metrics:
                print(f"{quarter:<12} {'N/A':>10} {'N/A':>10} {'N/A':>12} {0:>6}")
            else:
                print(
                    f"{quarter:<12} "
                    f"{metrics['accuracy']:>10.2%} "
                    f"{metrics['macro_f1']:>10.4f} "
                    f"{metrics['weighted_f1']:>12.4f} "
                    f"{metrics['successful_claims']:>6}"
                )
    else:
        print_quarterly_metrics(quarterly_metrics)

    if args.output:
        if not args.output.parent.exists():
            args.output.parent.mkdir(parents=True, exist_ok=True)

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(quarterly_metrics, f, indent=2)
        print(f"\nMetrics saved to: {args.output}")


if __name__ == "__main__":
    main()
