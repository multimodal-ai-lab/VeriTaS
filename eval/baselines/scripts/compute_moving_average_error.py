#!/usr/bin/env python3
"""
Compute moving average MAE and MSE from baseline results.

This script converts verdicts to numerical values:
- Intact -> 1.0
- Compromised -> -1.0
- Unknown -> 0.0

Then computes:
1. MAE (Mean Absolute Error) moving average
2. MSE (Mean Squared Error) moving average

Supports multi-model comparison mode.

```
python baselines/scripts/compute_moving_average_error.py \
	--multi baselines/results/longitudinal_custom-search/ \
	-o baselines/results/longitudinal_custom-search/ \
	-n 200 --show-abstain-baseline \
	&& python baselines/scripts/compute_moving_average_error.py \
	--multi baselines/results/longitudinal_no-search/ \
	-o baselines/results/longitudinal_no-search/ \
	-n 200 --show-abstain-baseline
```
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from plotting_utils import (
    get_model_cutoff,
    find_model_results,
    plot_multi_model_metric,
)

# Default path to claims file
DEFAULT_CLAIMS_PATH = Path(__file__).parent.parent / "data" / "veritas_release" / "veritas_longitudinal_2020_q1_2025_q4" / "claims.json"

# Verdict to numerical value mapping
VERDICT_TO_VALUE = {
    "INTACT": 1.0,
    "COMPROMISED": -1.0,
    "UNKNOWN": 0.0,
}


def verdict_to_value(verdict: str) -> float:
    """Convert verdict string to numerical value."""
    return VERDICT_TO_VALUE.get(verdict.upper(), 0.0)


def load_claim_dates(claims_path: Path) -> dict[int, datetime]:
    """Load claims and return mapping from claim_id to datetime."""
    with open(claims_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    id_to_date = {}
    for claim in data["claims"]:
        claim_id = claim["id"]
        date_str = claim["date"]
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        id_to_date[claim_id] = dt

    return id_to_date


def load_results_with_dates(
    results_path: Path,
    claims_path: Path,
) -> list[dict]:
    """
    Load results CSV and merge with claim dates.

    Returns list of dicts with keys: claim_id, date, gt_value, pred_value, abs_error, sq_error
    """
    claim_dates = load_claim_dates(claims_path)

    results = []
    with open(results_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] != "success":
                continue

            claim_id = int(row["claim_id"])
            if claim_id not in claim_dates:
                continue

            gt_value = verdict_to_value(row["gt_class"])
            pred_value = verdict_to_value(row["verdict"])

            error = pred_value - gt_value
            abs_error = abs(error)
            sq_error = error ** 2

            results.append({
                "claim_id": claim_id,
                "date": claim_dates[claim_id],
                "gt_value": gt_value,
                "pred_value": pred_value,
                "abs_error": abs_error,
                "sq_error": sq_error,
            })

    results.sort(key=lambda x: x["date"])
    return results


def compute_n_claim_moving_average_errors(
    results: list[dict],
    n: int = 100,
) -> tuple[list[datetime], list[float], list[float]]:
    """
    Compute moving average MAE and MSE over N claims.

    Returns:
        Tuple of (dates, mae_values, mse_values)
    """
    if len(results) < n:
        print(f"Warning: Only {len(results)} results, less than window size {n}")
        n = len(results)

    dates = []
    mae_values = []
    mse_values = []

    for i in range(n - 1, len(results)):
        window = results[i - n + 1:i + 1]

        mae = sum(r["abs_error"] for r in window) / len(window)
        mse = sum(r["sq_error"] for r in window) / len(window)

        dates.append(window[-1]["date"])
        mae_values.append(mae)
        mse_values.append(mse)

    return dates, mae_values, mse_values


def compute_abstain_baseline_errors(
    results: list[dict],
    n: int = 100,
) -> tuple[list[datetime], list[float], list[float]]:
    """
    Compute moving average MAE and MSE for always predicting NEI (0.0).

    This baseline shows what the error would be if we always abstained
    (predicted NEI = 0.0) instead of making a prediction.

    Returns:
        Tuple of (dates, mae_values, mse_values)
    """
    if len(results) < n:
        n = len(results)

    dates = []
    mae_values = []
    mse_values = []

    for i in range(n - 1, len(results)):
        window = results[i - n + 1:i + 1]

        # Error when always predicting 0.0: error = 0.0 - gt_value = -gt_value
        # abs_error = |gt_value|, sq_error = gt_value^2
        mae = sum(abs(r["gt_value"]) for r in window) / len(window)
        mse = sum(r["gt_value"] ** 2 for r in window) / len(window)

        dates.append(window[-1]["date"])
        mae_values.append(mae)
        mse_values.append(mse)

    return dates, mae_values, mse_values


def main():
    parser = argparse.ArgumentParser(
        description="Compute moving average MAE and MSE from baseline results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Verdict to value mapping:
  INTACT      ->  1.0
  COMPROMISED -> -1.0
  UNKNOWN     ->  0.0

Examples:
  # Multi-model comparison
  python compute_moving_average_error.py --multi path/to/results_dir/

  # Custom window size
  python compute_moving_average_error.py --multi results_dir/ -n 200
        """
    )
    parser.add_argument("--multi", type=Path, required=True,
                        help="Directory containing model subdirectories with results.csv files")
    parser.add_argument("-n", "--n-claims", type=int, default=100,
                        help="Window size for N-claim moving average (default: 100)")
    parser.add_argument("--claims", type=Path, default=None,
                        help="Path to claims.json (default: auto-detect)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory for plots")
    parser.add_argument("--show-abstain-baseline", action="store_true",
                        help="Show baseline curve for always predicting NEI (0.0)")

    args = parser.parse_args()

    claims_path = args.claims or DEFAULT_CLAIMS_PATH
    if not claims_path.exists():
        raise FileNotFoundError(f"Claims file not found: {claims_path}")

    if not args.multi.exists():
        raise FileNotFoundError(f"Directory not found: {args.multi}")

    model_results = find_model_results(args.multi)
    if not model_results:
        raise ValueError(f"No model results found in: {args.multi}")

    output_dir = args.output or args.multi
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(model_results)} models:")
    for model_name, _ in model_results:
        print(f"  - {model_name}")
    print(f"\nClaims file: {claims_path}")
    print(f"N-claim window: {args.n_claims}")
    print(f"Output directory: {output_dir}")

    # Load and compute for each model
    all_mae_data = {}
    all_mse_data = {}
    first_results = None  # Store results from first model for abstain baseline

    for model_name, results_path in model_results:
        print(f"\nProcessing {model_name}...")
        results = load_results_with_dates(results_path, claims_path)
        print(f"  Loaded {len(results)} results")

        if not results:
            print(f"  Skipping {model_name}: no valid results")
            continue

        if first_results is None:
            first_results = results

        dates, mae_values, mse_values = compute_n_claim_moving_average_errors(results, args.n_claims)

        all_mae_data[model_name] = (dates, mae_values)
        all_mse_data[model_name] = (dates, mse_values)

    # Compute abstain baseline if requested
    if args.show_abstain_baseline and first_results:
        print("\nComputing 'Always Abstain' baseline (NEI = 0.0)...")
        abstain_dates, abstain_mae, abstain_mse = compute_abstain_baseline_errors(
            first_results, args.n_claims
        )
        all_mae_data["Always Abstain"] = (abstain_dates, abstain_mae)
        all_mse_data["Always Abstain"] = (abstain_dates, abstain_mse)

    # Build cutoff dates
    model_cutoffs = {}
    for model_name in all_mae_data.keys():
        cutoff = get_model_cutoff(model_name)
        if cutoff:
            model_cutoffs[model_name] = cutoff

    # Generate plots
    if all_mae_data:
        print("\nGenerating combined plots...")
        for model_name, cutoff in model_cutoffs.items():
            print(f"  Cutoff for {model_name}: {cutoff.strftime('%Y-%m-%d')}")

        n = args.n_claims

        # MAE plot
        plot_multi_model_metric(
            all_mae_data,
            output_dir / f"MA{n}_claims_MAE.pdf",
            metric_name="MAE",
            title="",
            ylabel='Mean Absolute Error',
            cutoff_dates=model_cutoffs if model_cutoffs else None,
            max_jump=0.3,
            legend_loc='upper left',
        )

        # MSE plot
        plot_multi_model_metric(
            all_mse_data,
            output_dir / f"MA{n}_claims_MSE.pdf",
            metric_name="MSE",
            title="",
            ylabel='Mean Squared Error',
            cutoff_dates=model_cutoffs if model_cutoffs else None,
            max_jump=0.5,
            legend_loc='upper left',
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
