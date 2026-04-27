#!/usr/bin/env python3
"""Compare quarterly metrics across multiple models."""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Knowledge cutoff dates for models
KNOWLEDGE_CUTOFFS = {
    "gpt-4o": "2023-10",
    "gpt-5.2": "2025-08",
    "gemini-2.0-flash": "2024-08",
    "gemini-2.5-flash": "2025-01",
    "gemini-3-pro-preview": "2025-01",
    "llama-4-maverick": "2024-08",
}


def extract_model_name(file_path: Path) -> str:
    """Extract model name from file path."""
    # Try to extract from parent directory name (e.g., gemini-2.0-flash_custom-search_2026-01-02_14-19)
    parent_name = file_path.parent.name
    # Take the part before the first underscore that looks like a date
    match = re.match(r"([a-zA-Z0-9\-\.]+)", parent_name)
    if match:
        return match.group(1)
    return parent_name


def cutoff_to_quarter(cutoff: str) -> str:
    """Convert cutoff string (e.g., '2023-10') to quarter string (e.g., 'Q4 2023')."""
    year, month = cutoff.split("-")
    quarter = (int(month) - 1) // 3 + 1
    return f"Q{quarter} {year}"


def load_metrics(file_path: Path) -> dict:
    """Load quarterly metrics from JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_comparison_table(models_data: dict[str, dict], metric: str = "accuracy"):
    """Print comparison table for all models."""
    # Get all quarters across all models
    all_quarters = set()
    for metrics in models_data.values():
        all_quarters.update(metrics.keys())

    # Sort quarters chronologically
    sorted_quarters = sorted(all_quarters, key=lambda q: (int(q.split()[1]), int(q[1])))

    model_names = list(models_data.keys())

    # Header
    header = f"{'Quarter':<12}"
    for name in model_names:
        short_name = name[:15]
        header += f" | {short_name:>15}"
    print("\n" + "=" * len(header))
    print("QUARTERLY COMPARISON")
    print("=" * len(header))
    print(f"\nMetric: {metric}")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # Data rows
    for quarter in sorted_quarters:
        row = f"{quarter:<12}"
        for name in model_names:
            metrics = models_data[name].get(quarter, {})
            if "error" in metrics or not metrics:
                row += f" | {'N/A':>15}"
            else:
                value = metrics.get(metric, 0)
                if metric == "accuracy":
                    row += f" | {value:>14.2%}"
                else:
                    row += f" | {value:>15.4f}"
        print(row)

    print("-" * len(header))

    # Averages
    row = f"{'Average':<12}"
    for name in model_names:
        values = [
            m.get(metric, 0)
            for q, m in models_data[name].items()
            if "error" not in m and m
        ]
        if values:
            avg = sum(values) / len(values)
            if metric == "accuracy":
                row += f" | {avg:>14.2%}"
            else:
                row += f" | {avg:>15.4f}"
        else:
            row += f" | {'N/A':>15}"
    print(row)
    print("=" * len(header))


def create_comparison_plot(
    models_data: dict[str, dict],
    metric: str = "accuracy",
    output_path: Path | None = None,
    title: str | None = None,
):
    """Create a line plot comparing models across quarters."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Get all quarters and sort
    all_quarters = set()
    for metrics in models_data.values():
        all_quarters.update(metrics.keys())
    sorted_quarters = sorted(all_quarters, key=lambda q: (int(q.split()[1]), int(q[1])))

    # Create quarter index mapping
    quarter_to_idx = {q: i for i, q in enumerate(sorted_quarters)}
    x_positions = list(range(len(sorted_quarters)))

    # Color palette
    colors = plt.cm.tab10.colors

    # Plot each model
    for i, (model_name, metrics) in enumerate(models_data.items()):
        values = []
        positions = []
        for q in sorted_quarters:
            m = metrics.get(q, {})
            if "error" not in m and m and metric in m:
                values.append(m[metric] * 100 if metric == "accuracy" else m[metric])
                positions.append(quarter_to_idx[q])

        color = colors[i % len(colors)]
        ax.plot(positions, values, marker="o", linewidth=2, markersize=5,
                label=model_name, color=color)

        # Add knowledge cutoff marker if available
        model_key = model_name.lower().replace(" ", "-").replace("_", "-")
        cutoff = None
        for key, value in KNOWLEDGE_CUTOFFS.items():
            if key in model_key or model_key in key:
                cutoff = value
                break

        if cutoff:
            cutoff_quarter = cutoff_to_quarter(cutoff)
            if cutoff_quarter in quarter_to_idx:
                cutoff_idx = quarter_to_idx[cutoff_quarter]
                # Get the y value at this quarter for this model
                m = metrics.get(cutoff_quarter, {})
                if "error" not in m and m and metric in m:
                    y_val = m[metric] * 100 if metric == "accuracy" else m[metric]
                    # Add a star marker at the cutoff point
                    ax.plot(cutoff_idx, y_val, marker="*", markersize=18, color=color,
                            markeredgecolor="black", markeredgewidth=0.5, zorder=10)

    # X-axis formatting with quarter labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(sorted_quarters, rotation=45, ha="right", fontsize=9)

    ax.set_xlabel("Quarter", fontsize=12)
    if metric == "accuracy":
        ax.set_ylabel("Accuracy (%)", fontsize=12)
    else:
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    else:
        ax.set_title(f"Model Comparison: {metric.replace('_', ' ').title()} by Quarter",
                     fontsize=14, fontweight="bold")

    # Add legend entry for knowledge cutoff stars
    ax.plot([], [], marker="*", markersize=12, color="gray", linestyle="None",
            markeredgecolor="black", markeredgewidth=0.5, label="Knowledge cutoff")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare quarterly metrics across multiple models."
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        type=Path,
        help="Paths to quarterly metrics JSON files.",
    )
    parser.add_argument(
        "--labels",
        "-l",
        nargs="+",
        help="Optional labels for each model (must match number of input files).",
    )
    parser.add_argument(
        "--metric",
        "-m",
        default="accuracy",
        choices=["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"],
        help="Metric to compare (default: accuracy).",
    )
    parser.add_argument(
        "--plot",
        "-p",
        type=Path,
        help="Output path for the comparison plot (e.g., comparison.png).",
    )
    parser.add_argument(
        "--title",
        "-t",
        help="Custom title for the plot.",
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="Skip printing the comparison table.",
    )
    args = parser.parse_args()

    # Validate labels
    if args.labels and len(args.labels) != len(args.input_files):
        raise ValueError(
            f"Number of labels ({len(args.labels)}) must match number of input files ({len(args.input_files)})"
        )

    # Load all metrics
    models_data = {}
    for i, file_path in enumerate(args.input_files):
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if args.labels:
            model_name = args.labels[i]
        else:
            model_name = extract_model_name(file_path)

        models_data[model_name] = load_metrics(file_path)

    # Print comparison table
    if not args.no_table:
        print_comparison_table(models_data, args.metric)

    # Create plot
    if args.plot:
        create_comparison_plot(models_data, args.metric, args.plot, args.title)


if __name__ == "__main__":
    main()
