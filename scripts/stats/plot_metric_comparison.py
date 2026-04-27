"""Plots a grouped bar chart comparing penalty by metric (MSE, MAE, 7-bin Accuracy).

The x-axis shows the absolute difference between ground truth and prediction
(0 to 2 in steps of 1/3).  For each tick three bars are drawn – one per metric.

Run this script manually:
    python -m scripts.stats.plot_metric_comparison
"""

import os

from scripts.stats.common import COLORS, _ensure_plots_dir

# ---------------------------------------------------------------------------
# Metric penalty functions
# ---------------------------------------------------------------------------

def mse_penalty(diff: float) -> float:
    """Mean Squared Error penalty for a single difference."""
    return diff ** 2


def mae_penalty(diff: float) -> float:
    """Mean Absolute Error penalty for a single difference."""
    return abs(diff)


def seven_bin_accuracy_penalty(diff: float) -> float:
    """7-bin accuracy penalty: 0 when the prediction falls in the same bin, 1 otherwise."""
    return 0.0 if abs(diff) < 1e-9 else 1.0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metric_comparison():
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import numpy as np
    except ImportError:
        print("matplotlib and numpy are required. pip install matplotlib numpy")
        return

    # X-axis: differences from 0 to 2 in steps of 1/3
    diffs = [i / 3 for i in range(7)]  # 0, 1/3, 2/3, 1, 4/3, 5/3, 2

    metrics = {
        "MSE": mse_penalty,
        "MAE": mae_penalty,
        "7-bin Accuracy": seven_bin_accuracy_penalty,
    }

    tick_labels = ["0", "1/3", "2/3", "1", "1 1/3", "1 2/3", "2"]
    x = np.arange(len(diffs))
    n_metrics = len(metrics)
    bar_width = 0.25

    # Use the orange / negative colour from stats.common
    base_colors = [COLORS["orange"], COLORS["light_orange"], COLORS["negative"]]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    for idx, (name, fn) in enumerate(metrics.items()):
        values = [fn(d) for d in diffs]
        offset = (idx - (n_metrics - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            values,
            width=bar_width,
            label=name,
            color=base_colors[idx],
            edgecolor="white",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Difference (ground truth – prediction)")
    ax.set_ylabel("Effective penalty")
    ax.set_title("Penalty by Metric", fontsize=18)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out_dir = _ensure_plots_dir()
    out_path = os.path.join(out_dir, "metric_comparison.pdf")
    plt.savefig(out_path)
    plt.show()
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def plot_metric_comparison_continuous():
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib and numpy are required. pip install matplotlib numpy")
        return

    x = np.linspace(0, 2, 500)

    penalties = {
        "MSE": x ** 2,
        "MAE": x,
        "7-bin Accuracy": np.where(x < 1 / 6, 0.0, 1.0),
    }

    base_colors = [COLORS["orange"], COLORS["light_orange"], COLORS["negative"]]

    tick_positions = [i / 3 for i in range(7)]  # 0, 1/3, 2/3, 1, 4/3, 5/3, 2
    tick_labels = ["0", "1/3", "2/3", "1", "1 1/3", "1 2/3", "2"]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)

    lines = {}
    for idx, (name, y) in enumerate(penalties.items()):
        line, = ax.plot(x, y, color=base_colors[idx], linewidth=2)
        lines[name] = (line, y, base_colors[idx])

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Difference (ground truth – prediction)")
    ax.set_ylabel("Effective penalty")
    ax.set_title("Penalty by Metric")

    # Place labels directly at the end of each line (right side)
    for name, (line, y, color) in lines.items():
        ax.text(
            x[-1], y[-1] + 0.03, name,
            color=color, fontsize=10, fontweight="bold",
            va="bottom", ha='right',
        )

    ax.grid(True, axis="y", alpha=0.25, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out_dir = _ensure_plots_dir()
    out_path = os.path.join(out_dir, "metric_comparison_continuous.pdf")
    plt.savefig(out_path)
    plt.show()
    plt.close(fig)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    plot_metric_comparison()
    plot_metric_comparison_continuous()
