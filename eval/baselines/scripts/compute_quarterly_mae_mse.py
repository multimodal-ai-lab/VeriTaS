#!/usr/bin/env python3
"""
Compute per-quarter MAE and MSE from baseline results.

Like compute_moving_average_error.py but aggregates one data point per
quarter instead of a rolling window.

Example:
  python baselines/scripts/compute_quarterly_mae_mse.py \
      --multi baselines/results/camera_ready/longitudinal/no-search \
      -o baselines/results/camera_ready/longitudinal/no-search \
      --show-abstain-baseline
"""

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from plotting_utils import (
    get_model_cutoff,
    find_model_results,
    get_display_name,
    build_model_colors,
    build_model_line_styles,
    add_cutoff_lines,
    FIGURE_SIZE,
    FONT_SIZES,
    ABSTAIN_COLOR,
)

DEFAULT_CLAIMS_PATH = Path("/mnt/vast/workspaces/PI_Rohrbach/mk79honu/data/VeriTaS/veritas_release/veritas_longitudinal_2020_q1_2026_q1/claims.json")

VERDICT_TO_VALUE = {
    "INTACT": 1.0,
    "COMPROMISED": -1.0,
    "UNKNOWN": 0.0,
}


def verdict_to_value(verdict: str) -> float:
    key = verdict.split("(")[0].strip().upper()
    return VERDICT_TO_VALUE.get(key, 0.0)


def date_to_quarter(dt: datetime) -> tuple[int, int]:
    return dt.year, (dt.month - 1) // 3 + 1


def quarter_to_datetime(year: int, quarter: int) -> datetime:
    return datetime(year, (quarter - 1) * 3 + 1, 1)


def load_claim_dates(claims_path: Path) -> dict[int, datetime]:
    with open(claims_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    id_to_date = {}
    for claim in data["claims"]:
        dt = datetime.fromisoformat(claim["date"].replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)
        id_to_date[claim["id"]] = dt
    return id_to_date


def load_results_with_dates(results_path: Path, claims_path: Path) -> list[dict]:
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
            results.append({
                "claim_id": claim_id,
                "date": claim_dates[claim_id],
                "gt_value": gt_value,
                "pred_value": pred_value,
                "abs_error": abs(error),
                "sq_error": error ** 2,
            })
    return results


def compute_quarterly_errors(
    results: list[dict],
) -> tuple[list[datetime], list[float], list[float], list[int]]:
    quarter_data: dict[tuple, list] = defaultdict(list)
    for r in results:
        quarter_data[date_to_quarter(r["date"])].append(r)

    dates, mae_values, mse_values, counts = [], [], [], []
    for year, q in sorted(quarter_data.keys()):
        window = quarter_data[(year, q)]
        mae = sum(r["abs_error"] for r in window) / len(window)
        mse = sum(r["sq_error"] for r in window) / len(window)
        dates.append(quarter_to_datetime(year, q))
        mae_values.append(mae)
        mse_values.append(mse)
        counts.append(len(window))

    return dates, mae_values, mse_values, counts


def compute_quarterly_abstain_errors(
    results: list[dict],
) -> tuple[list[datetime], list[float], list[float]]:
    quarter_data: dict[tuple, list] = defaultdict(list)
    for r in results:
        quarter_data[date_to_quarter(r["date"])].append(r)

    dates, mae_values, mse_values = [], [], []
    for year, q in sorted(quarter_data.keys()):
        window = quarter_data[(year, q)]
        mae = sum(abs(r["gt_value"]) for r in window) / len(window)
        mse = sum(r["gt_value"] ** 2 for r in window) / len(window)
        dates.append(quarter_to_datetime(year, q))
        mae_values.append(mae)
        mse_values.append(mse)

    return dates, mae_values, mse_values


def plot_quarterly_metric(
    all_model_data: dict[str, tuple[list, list]],
    output_path: Path,
    metric_name: str,
    ylabel: str,
    cutoff_dates: dict[str, datetime] | None = None,
    legend_loc: str = 'upper left',
):
    model_colors = build_model_colors(list(all_model_data.keys()))
    model_line_styles = build_model_line_styles(list(all_model_data.keys()))
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    def model_sort_key(name: str) -> tuple:
        if name == "Always Abstain":
            return (0, 0)
        n = name.lower()
        if "llama" in n:
            return (1, 0)
        if "gemini" in n:
            try:
                v = int(n.split('-')[1].split('.')[0])
            except (IndexError, ValueError):
                v = 0
            return (2, v)
        if "gpt" in n:
            try:
                v = int(n.split('-')[1].split('.')[0])
            except (IndexError, ValueError):
                v = 0
            return (3, v)
        return (4, 0)

    for model_name in sorted(all_model_data.keys(), key=model_sort_key):
        dates, values = all_model_data[model_name]
        if not dates:
            continue
        if model_name == "Always Abstain":
            mean_val = sum(values) / len(values)
            ax.axhline(y=mean_val, color=ABSTAIN_COLOR, linestyle='--', linewidth=2.0,
                       label='Always NEI', alpha=0.6)
        else:
            color = model_colors[model_name]
            linestyle = model_line_styles[model_name]
            ax.plot(dates, values, linestyle, linewidth=1.5,
                    color=color, label=get_display_name(model_name), alpha=0.8)

    add_cutoff_lines(ax, cutoff_dates, model_colors,
                     'top' if metric_name == 'MAE' else 'bottom')

    ylim = {'MAE': (0.0, 1.05), 'MSE': (0.0, 1.5)}.get(metric_name)
    if ylim:
        ax.set_ylim(*ylim)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))

    ax.set_xlabel('Claim Date', fontsize=FONT_SIZES['axis_label'])
    ax.set_ylabel(ylabel, fontsize=FONT_SIZES['axis_label'])
    ax.tick_params(axis='both', labelsize=FONT_SIZES['tick'])
    ax.legend(loc=legend_loc, fontsize=FONT_SIZES['legend'], ncol=2, handlelength=1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-quarter MAE and MSE from baseline results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--multi", type=Path, required=True,
                        help="Directory containing model subdirectories with results.csv files")
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
    print(f"Output directory: {output_dir}")

    all_mae_data: dict[str, tuple] = {}
    all_mse_data: dict[str, tuple] = {}
    first_results = None

    for model_name, results_path in model_results:
        print(f"\nProcessing {model_name}...")
        results = load_results_with_dates(results_path, claims_path)
        print(f"  Loaded {len(results)} results")
        if not results:
            print(f"  Skipping {model_name}: no valid results")
            continue
        if first_results is None:
            first_results = results

        dates, mae_values, mse_values, counts = compute_quarterly_errors(results)
        print(f"  Quarters: {len(dates)}, total claims: {sum(counts)}")
        all_mae_data[model_name] = (dates, mae_values)
        all_mse_data[model_name] = (dates, mse_values)

    if args.show_abstain_baseline and first_results:
        print("\nComputing 'Always Abstain' baseline (NEI = 0.0)...")
        abstain_dates, abstain_mae, abstain_mse = compute_quarterly_abstain_errors(first_results)
        all_mae_data["Always Abstain"] = (abstain_dates, abstain_mae)
        all_mse_data["Always Abstain"] = (abstain_dates, abstain_mse)

    model_cutoffs = {
        name: cutoff
        for name in all_mae_data
        if (cutoff := get_model_cutoff(name))
    }

    if all_mae_data:
        print("\nGenerating quarterly plots...")
        for model_name, cutoff in model_cutoffs.items():
            print(f"  Cutoff for {model_name}: {cutoff.strftime('%Y-%m-%d')}")

        plot_quarterly_metric(
            all_mae_data,
            output_dir / "quarterly_MAE.pdf",
            "MAE",
            "Mean Absolute Error",
            model_cutoffs,
            legend_loc='upper left',
        )
        plot_quarterly_metric(
            all_mse_data,
            output_dir / "quarterly_MSE.pdf",
            "MSE",
            "Mean Squared Error",
            model_cutoffs,
            legend_loc='upper left',
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
