#!/usr/bin/env python3
"""
Compute moving average accuracy from baseline results.

This script computes two types of moving averages to analyze how model
performance changes over time (useful for detecting knowledge cutoff effects):

1. N-claim moving average: Rolling window of N claims (sorted by date)
2. M-day moving average: Rolling window of M days

Supports both single-model and multi-model comparison modes.

```
python baselines/scripts/compute_moving_average.py \
	--multi baselines/results/longitudinal_custom-search/ \
	-o baselines/results/longitudinal_custom-search/ -n 200 \
	&& python baselines/scripts/compute_moving_average.py \
	--multi baselines/results/longitudinal_no-search/ \
	-o baselines/results/longitudinal_no-search/ -n 200
```
"""

import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

from plotting_utils import (
    get_model_cutoff,
    find_model_results,
    plot_multi_model_metric,
)

# Default path to claims file
DEFAULT_CLAIMS_PATH = Path(__file__).parent.parent / "data" / "veritas_release" / "veritas_longitudinal_2020_q1_2025_q4" / "claims.json"
DEFAULT_CLAIMS_PATH = Path("/mnt/vast/workspaces/PI_Rohrbach/mk79honu/data/VeriTaS/veritas_release/veritas_longitudinal_2020_q1_2026_q1/claims.json")


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

    Returns list of dicts with keys: claim_id, date, gt_class, verdict, correct
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

            results.append({
                "claim_id": claim_id,
                "date": claim_dates[claim_id],
                "gt_class": row["gt_class"],
                "verdict": row["verdict"],
                "correct": row["correct"] == "True",
            })

    results.sort(key=lambda x: x["date"])
    return results


def compute_n_claim_moving_average(
    results: list[dict],
    n: int = 100,
) -> tuple[list[datetime], list[float], list[int]]:
    """
    Compute moving average accuracy over N claims.

    Returns:
        Tuple of (dates, accuracies, counts)
    """
    if len(results) < n:
        print(f"Warning: Only {len(results)} results, less than window size {n}")
        n = len(results)

    dates = []
    accuracies = []
    counts = []

    for i in range(n - 1, len(results)):
        window = results[i - n + 1:i + 1]
        correct_count = sum(1 for r in window if r["correct"])
        accuracy = correct_count / len(window)

        dates.append(window[-1]["date"])
        accuracies.append(accuracy)
        counts.append(len(window))

    return dates, accuracies, counts


def compute_m_day_moving_average(
    results: list[dict],
    m: int = 30,
) -> tuple[list[datetime], list[float], list[int]]:
    """
    Compute moving average accuracy over M-day windows.

    Returns:
        Tuple of (dates, accuracies, counts)
    """
    if not results:
        return [], [], []

    dates = []
    accuracies = []
    counts = []

    unique_dates = sorted(set(r["date"].date() for r in results))

    for current_date in unique_dates:
        window_start = current_date - timedelta(days=m - 1)

        window_results = [
            r for r in results
            if window_start <= r["date"].date() <= current_date
        ]

        if not window_results:
            continue

        correct_count = sum(1 for r in window_results if r["correct"])
        accuracy = correct_count / len(window_results)

        dates.append(datetime.combine(current_date, datetime.min.time()))
        accuracies.append(accuracy)
        counts.append(len(window_results))

    return dates, accuracies, counts


def plot_single_model_moving_averages(
    n_claim_data: tuple[list[datetime], list[float], list[int]],
    m_day_data: tuple[list[datetime], list[float], list[int]],
    n: int,
    m: int,
    output_dir: Path,
    model_name: str = "",
    cutoff_date: datetime | None = None,
):
    """Generate plots for a single model."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("Warning: matplotlib not installed, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: N-claim moving average
    if n_claim_data[0]:
        fig, ax = plt.subplots(figsize=(12, 6))

        dates, accuracies, counts = n_claim_data
        ax.plot(dates, accuracies, 'b-', linewidth=1.5, label=f'{n}-claim moving average')

        if cutoff_date:
            ax.axvline(x=cutoff_date, color='r', linestyle='--', linewidth=2,
                      label=f'Knowledge cutoff ({cutoff_date.strftime("%Y-%m-%d")})')

        ax.set_xlabel('Claim Date')
        ax.set_ylabel(f'Accuracy, {n} Moving Average')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plot_path = output_dir / f"MA{n}_claims_ACC.pdf"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plot_path}")

    # Plot 2: M-day moving average
    if m_day_data[0]:
        fig, ax = plt.subplots(figsize=(12, 6))

        dates, accuracies, counts = m_day_data
        ax.plot(dates, accuracies, 'g-', linewidth=1.5, label=f'{m}-day moving average')

        if cutoff_date:
            ax.axvline(x=cutoff_date, color='r', linestyle='--', linewidth=2,
                      label=f'Knowledge cutoff ({cutoff_date.strftime("%Y-%m-%d")})')

        ax.set_xlabel('Date')
        ax.set_ylabel(f'Accuracy, {m}-day Moving Average')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plot_path = output_dir / f"avg_{m}_days.pdf"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plot_path}")


def save_moving_average_data(
    n_claim_data: tuple[list[datetime], list[float], list[int]],
    m_day_data: tuple[list[datetime], list[float], list[int]],
    n: int,
    m: int,
    output_path: Path,
    model_name: str = "",
):
    """Save moving average data to JSON."""
    data = {
        "model": model_name,
        "n_claim_window": n,
        "m_day_window": m,
        "n_claim_moving_average": [
            {"date": d.isoformat(), "accuracy": acc, "count": cnt}
            for d, acc, cnt in zip(*n_claim_data)
        ],
        "m_day_moving_average": [
            {"date": d.isoformat(), "accuracy": acc, "count": cnt}
            for d, acc, cnt in zip(*m_day_data)
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Compute moving average accuracy from baseline results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single model
  python compute_moving_average.py results.csv

  # Multi-model comparison (provide directory containing model subdirs)
  python compute_moving_average.py --multi path/to/results_dir/

  # Custom window sizes
  python compute_moving_average.py --multi results_dir/ -n 50 -m 14

  # With output directory
  python compute_moving_average.py --multi results_dir/ -o output/
        """
    )
    parser.add_argument("results_path", type=Path, nargs="?", default=None,
                        help="Path to results CSV (single model) or directory (multi-model)")
    parser.add_argument("--multi", type=Path, default=None,
                        help="Directory containing model subdirectories with results.csv files")
    parser.add_argument("-n", "--n-claims", type=int, default=100,
                        help="Window size for N-claim moving average (default: 100)")
    parser.add_argument("-m", "--m-days", type=int, default=30,
                        help="Window size in days for M-day moving average (default: 30)")
    parser.add_argument("--claims", type=Path, default=None,
                        help="Path to claims.json (default: auto-detect)")
    parser.add_argument("--cutoff", type=str, default=None,
                        help="Knowledge cutoff date in YYYY-MM-DD format (for single model)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory for plots")
    parser.add_argument("--model-name", type=str, default="",
                        help="Model name for plot titles (single model mode)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip generating plots, only output data")

    args = parser.parse_args()

    claims_path = args.claims or DEFAULT_CLAIMS_PATH
    if not claims_path.exists():
        raise FileNotFoundError(f"Claims file not found: {claims_path}")

    cutoff_date = None
    if args.cutoff:
        cutoff_date = datetime.strptime(args.cutoff, "%Y-%m-%d")

    # Multi-model mode
    if args.multi:
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
        print(f"M-day window: {args.m_days}")
        print(f"Output directory: {output_dir}")

        # Load and compute for each model
        all_n_claim_data = {}
        all_m_day_data = {}

        for model_name, results_path in model_results:
            print(f"\nProcessing {model_name}...")
            results = load_results_with_dates(results_path, claims_path)
            print(f"  Loaded {len(results)} results")

            if not results:
                print(f"  Skipping {model_name}: no valid results")
                continue

            n_claim_data = compute_n_claim_moving_average(results, args.n_claims)
            m_day_data = compute_m_day_moving_average(results, args.m_days)

            all_n_claim_data[model_name] = (n_claim_data[0], n_claim_data[1])
            all_m_day_data[model_name] = (m_day_data[0], m_day_data[1])

            # Save individual model data
            data_path = output_dir / f"{model_name}_moving_average.json"
            save_moving_average_data(n_claim_data, m_day_data, args.n_claims, args.m_days, data_path, model_name)

        # Build cutoff dates
        model_cutoffs = {}
        for model_name in all_n_claim_data.keys():
            cutoff = get_model_cutoff(model_name)
            if cutoff:
                model_cutoffs[model_name] = cutoff

        # Generate combined plots
        if not args.no_plot and all_n_claim_data:
            print("\nGenerating combined plots...")
            for model_name, cutoff in model_cutoffs.items():
                print(f"  Cutoff for {model_name}: {cutoff.strftime('%Y-%m-%d')}")

            n = args.n_claims
            m = args.m_days

            # N-claim accuracy plot
            plot_multi_model_metric(
                all_n_claim_data,
                output_dir / f"MA{n}_claims_ACC.pdf",
                metric_name="Accuracy",
                title="",
                ylabel=f'Accuracy, {n} Moving Average',
                cutoff_dates=model_cutoffs if model_cutoffs else None,
                max_jump=0.15,
                legend_loc='lower left',
            )

            # M-day accuracy plot
            plot_multi_model_metric(
                all_m_day_data,
                output_dir / f"MA{m}_days_ACC.pdf",
                metric_name="Accuracy",
                title="",
                ylabel=f'Accuracy, {m}-day Moving Average',
                cutoff_dates=model_cutoffs if model_cutoffs else None,
                max_jump=0.15,
                legend_loc='lower left',
            )

        print("\nDone!")

    # Single model mode
    elif args.results_path:
        if not args.results_path.exists():
            raise FileNotFoundError(f"Results file not found: {args.results_path}")

        output_dir = args.output or args.results_path.parent

        model_name = args.model_name
        if not model_name:
            model_name = args.results_path.parent.name

        print(f"Loading results from: {args.results_path}")
        print(f"Claims file: {claims_path}")
        print(f"N-claim window: {args.n_claims}")
        print(f"M-day window: {args.m_days}")
        if cutoff_date:
            print(f"Knowledge cutoff: {cutoff_date.strftime('%Y-%m-%d')}")

        results = load_results_with_dates(args.results_path, claims_path)
        print(f"Loaded {len(results)} successful results")

        if not results:
            print("Error: No valid results found")
            return

        print(f"Date range: {results[0]['date'].strftime('%Y-%m-%d')} to {results[-1]['date'].strftime('%Y-%m-%d')}")

        print("\nComputing moving averages...")
        n_claim_data = compute_n_claim_moving_average(results, args.n_claims)
        m_day_data = compute_m_day_moving_average(results, args.m_days)

        print(f"  N-claim average: {len(n_claim_data[0])} data points")
        print(f"  M-day average: {len(m_day_data[0])} data points")

        data_path = output_dir / "moving_average_data.json"
        save_moving_average_data(n_claim_data, m_day_data, args.n_claims, args.m_days, data_path, model_name)
        print(f"  Data saved: {data_path}")

        if not args.no_plot:
            print("\nGenerating plots...")
            plot_single_model_moving_averages(
                n_claim_data,
                m_day_data,
                args.n_claims,
                args.m_days,
                output_dir,
                model_name,
                cutoff_date,
            )

        print("\nDone!")

    else:
        parser.print_help()
        print("\nError: Provide either a results CSV path or use --multi with a directory")


if __name__ == "__main__":
    main()
