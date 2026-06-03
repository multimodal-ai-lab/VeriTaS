#!/usr/bin/env python3
"""
Compute MSE and MAE metrics for existing baseline results.

Converts verdicts to numerical values:
- Compromised -> -1
- Unknown -> 0
- Intact -> 1

Then computes MSE and MAE between gt_integrity and the converted verdict.

Usage:
    python compute_regression_metrics.py /path/to/results/directory
"""

import argparse
import csv
from pathlib import Path


VERDICT_TO_NUMERIC = {
    "Compromised": -1.0,
    "Unknown": 0.0,
    "Intact": 1.0,
}


def compute_mse_mae(gt_values: list[float], pred_values: list[float]) -> tuple[float, float]:
    """Compute MSE and MAE between ground truth and predictions."""
    if not gt_values:
        return None, None

    n = len(gt_values)
    mse = sum((gt - pred) ** 2 for gt, pred in zip(gt_values, pred_values)) / n
    mae = sum(abs(gt - pred) for gt, pred in zip(gt_values, pred_values)) / n

    return mse, mae


def process_results_csv(csv_path: Path) -> dict:
    """Process a results.csv file and compute metrics per model."""
    results_by_model = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] != "success":
                continue

            gt_integrity = row.get("gt_integrity")
            verdict = row.get("verdict")
            model = row.get("model", "unknown")

            if gt_integrity is None or gt_integrity == "" or verdict not in VERDICT_TO_NUMERIC:
                continue

            try:
                gt_value = float(gt_integrity)
            except ValueError:
                continue

            pred_value = VERDICT_TO_NUMERIC[verdict]

            if model not in results_by_model:
                results_by_model[model] = {"gt": [], "pred": []}

            results_by_model[model]["gt"].append(gt_value)
            results_by_model[model]["pred"].append(pred_value)

    return results_by_model


def compute_metrics_for_directory(results_dir: Path) -> dict:
    """Compute metrics for all model subdirectories in a results directory."""
    metrics = {}

    # Check for model subdirectories
    subdirs = [d for d in results_dir.iterdir() if d.is_dir()]

    if subdirs:
        # Process each model subdirectory
        for model_dir in sorted(subdirs):
            csv_path = model_dir / "results.csv"
            if csv_path.exists():
                results_by_model = process_results_csv(csv_path)
                for model, data in results_by_model.items():
                    mse, mae = compute_mse_mae(data["gt"], data["pred"])
                    metrics[model_dir.name] = {
                        "n_samples": len(data["gt"]),
                        "mse": round(mse, 4) if mse is not None else None,
                        "mae": round(mae, 4) if mae is not None else None,
                    }
    else:
        # Process single results.csv in the directory
        csv_path = results_dir / "results.csv"
        if csv_path.exists():
            results_by_model = process_results_csv(csv_path)
            for model, data in results_by_model.items():
                mse, mae = compute_mse_mae(data["gt"], data["pred"])
                metrics[model] = {
                    "n_samples": len(data["gt"]),
                    "mse": round(mse, 4) if mse is not None else None,
                    "mae": round(mae, 4) if mae is not None else None,
                }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Compute MSE and MAE metrics for baseline results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python compute_regression_metrics.py /path/to/results/longitudinal_custom-search
    python compute_regression_metrics.py /path/to/results/longitudinal_no-search
        """
    )
    parser.add_argument("results_dir", type=str, help="Path to results directory")

    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        return 1

    print(f"Computing metrics for: {results_dir}")
    print("=" * 60)

    metrics = compute_metrics_for_directory(results_dir)

    if not metrics:
        print("No results found.")
        return 1

    # Print results in a table format
    print(f"\n{'Model':<45} {'N':>8} {'MSE':>10} {'MAE':>10}")
    print("-" * 75)

    for model, m in sorted(metrics.items()):
        mse_str = f"{m['mse']:.4f}" if m['mse'] is not None else "N/A"
        mae_str = f"{m['mae']:.4f}" if m['mae'] is not None else "N/A"
        print(f"{model:<45} {m['n_samples']:>8} {mse_str:>10} {mae_str:>10}")

    print("-" * 75)

    # Compute averages
    valid_mse = [m['mse'] for m in metrics.values() if m['mse'] is not None]
    valid_mae = [m['mae'] for m in metrics.values() if m['mae'] is not None]

    if valid_mse:
        avg_mse = sum(valid_mse) / len(valid_mse)
        avg_mae = sum(valid_mae) / len(valid_mae)
        print(f"{'Average':<45} {'':<8} {avg_mse:>10.4f} {avg_mae:>10.4f}")

    return 0


if __name__ == "__main__":
    exit(main())
