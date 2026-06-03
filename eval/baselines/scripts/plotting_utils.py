#!/usr/bin/env python3
"""
Shared plotting utilities for moving average analysis scripts.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

# Figure size for single-column paper format
FIGURE_SIZE = (7, 6)

# Font sizes for paper readability
FONT_SIZES = {
    'axis_label': 18,
    'title': 16,
    'legend': 12,
    'tick': 15,
    'cutoff_label': 12,
}

# Default y-axis limits per metric (use None for dynamic limits)
METRIC_YLIM = {
    'MSE': (0.0, 1.32),
    'MAE': (0.0, 1.05),
    'Accuracy': (0.1, 0.9),
}

# Colors for multi-model plots
# MODEL_COLORS = [
#     '#1f77b4',  # blue
#     '#ff7f0e',  # orange
#     '#2ca02c',  # green
#     '#d62728',  # red
#     '#9467bd',  # purple
#     '#8c564b',  # brown
#     '#e377c2',  # pink
#     '#7f7f7f',  # gray
#     '#bcbd22',  # olive
#     '#17becf',  # cyan
# ]
COLORS = dict(
    orange="#EC6500",  # Primary, TUDa
    soft_orange="#FFC599",
    light_orange="#F5A300",  # Secondary, TUDa
    soft_light_orange="#FFCE6B",
    darkblue="#004E73",  # Tertiary, TUDa
    blue="#0083CC",  # Quaternary, TUDa
    soft_blue="#83D3FF",
    negative="#BD0A0A",
    neutral="#a1a1a1",
    positive="#09C479",
)


# abstaining is dark gray
ABSTAIN_COLOR = '#444444'

# Model display name abbreviations (for cleaner legends)
MODEL_DISPLAY_NAMES = {
    "Llama-4-Maverick-17B-128E-Instruct-FP8": "Llama 4 Maverick",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "gemini-3-pro-preview": "Gemini 3 Pro",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gpt-4.1": "GPT 4.1",
    "gpt-4o": "GPT 4o",
    "gpt-4o_no": "GPT 4o",
    "gpt-5.2": "GPT 5.2",
    "gpt-5.2_no": "GPT 5.2",
    "sonar-pro": "Sonar-Pro",
    "claude-opus-4-6": "Claude Opus 4.6",
    "google/gemma-4-31B-it": "Gemma 4",
    "gemma-4-31B-it": "Gemma 4",
}

COLORS_TO_MODELS = {
    "gpt-4o": COLORS['darkblue'],
    "gpt-5.2": COLORS['blue'],
    "gemini-3.1-pro-preview": COLORS['orange'],
    "gemini-3-flash-preview": COLORS['light_orange'],
    "google/gemma-4-31B-it": COLORS['soft_orange'],
    "claude-opus-4-6": COLORS['negative'],
    "Llama-4-Maverick-17B-128E-Instruct-FP8": COLORS['positive'],
}

MODEL_LINE_STYLES = {
    "gpt-4o": "--",
    "gpt-5.2": "--",
    "gemini-3.1-pro-preview": "-.",
    "gemini-3-flash-preview": "-.",
    "google/gemma-4-31B-it": "-.",
    "claude-opus-4-6": "-",
    "Llama-4-Maverick-17B-128E-Instruct-FP8": ":",
}


def get_display_name(model_name: str) -> str:
    """Get abbreviated display name for a model."""
    return MODEL_DISPLAY_NAMES.get(model_name, model_name)


# Override labels for cutoff lines when multiple models share a date
CUTOFF_LABELS: dict[datetime, str] = {
    datetime(2025, 1, 1): "Gemini & Gemma",
}

# Known knowledge cutoff dates for models
MODEL_CUTOFFS = {
    # OpenAI models
    "gpt-5.2": datetime(2025, 8, 1),
    "gpt-4o": datetime(2023, 10, 1),
    # Google models
    "gemini-3-pro": datetime(2025, 1, 1),
    "gemini-3.1-pro-preview": datetime(2025, 1, 1),
    "gemini-3-flash-preview": datetime(2025, 1, 1),
    "gemini-2.5-flash": datetime(2025, 1, 9),
    "gemini-2.0-flash": datetime(2024, 8, 1),
    "gemma-4-31B-it": datetime(2025, 1, 1),
    # Meta models
    "llama-4": datetime(2024, 8, 9),
    # Perplexity
    "sonar-pro": None,
    "sonar": None,
    # Anthropic models
    "claude-opus-4-6": datetime(2025, 5, 1),
}


def get_model_cutoff(model_name: str) -> datetime | None:
    """Get the knowledge cutoff date for a model based on its name."""
    model_lower = model_name.lower()
    for pattern, cutoff in MODEL_CUTOFFS.items():
        if pattern.lower() == model_lower:
            return cutoff
    for pattern, cutoff in MODEL_CUTOFFS.items():
        if pattern.lower() in model_lower:
            return cutoff
    return None


def build_model_colors(model_names: list[str]) -> dict[str, str]:
    """Build a color mapping for model names."""
    result = {name: COLORS_TO_MODELS.get(name, 'gray') for name in model_names}
    if "Always Abstain" in model_names:
        result["Always Abstain"] = ABSTAIN_COLOR
    return result


def build_model_line_styles(model_names: list[str]) -> dict[str, str]:
    """Build a line style mapping for model names."""
    result = {name: MODEL_LINE_STYLES.get(name, '-') for name in model_names}
    if "Always Abstain" in model_names:
        result["Always Abstain"] = '--'
    return result


def filter_spikes(dates: list, values: list, max_jump: float = 0.15) -> tuple[list, list]:
    """
    Replace values with NaN where there's an unrealistic jump.

    Args:
        dates: List of dates
        values: List of values
        max_jump: Maximum allowed change between consecutive points

    Returns:
        Tuple of (dates, filtered_values)
    """
    if len(dates) < 2:
        return dates, values

    new_values = [values[0]]
    for i in range(1, len(values)):
        jump = abs(values[i] - values[i-1])
        if jump > max_jump:
            new_values.append(float('nan'))
        else:
            new_values.append(values[i])
    return dates, new_values


def compute_cross_model_average(
    all_data: list[tuple[list, list]],
    num_models: int
) -> tuple[list, list]:
    """
    Compute average across all models, only including dates where all models have data.

    Args:
        all_data: List of (dates, values) tuples for each model
        num_models: Total number of models

    Returns:
        Tuple of (avg_dates, avg_values)
    """
    from collections import defaultdict

    date_to_values = defaultdict(list)
    for dates, values in all_data:
        for d, v in zip(dates, values):
            date_to_values[d].append(v)

    # Only include dates where ALL models have data
    avg_dates = sorted([d for d in date_to_values.keys() if len(date_to_values[d]) == num_models])
    avg_values = [sum(date_to_values[d]) / len(date_to_values[d]) for d in avg_dates]

    return avg_dates, avg_values


def add_cutoff_lines(
    ax,
    cutoff_dates: dict[str, datetime] | None,
    model_colors: dict[str, str],
    kcd_position: Literal['top', 'bottom'] = 'top',
    cutoff_labels: dict[datetime, str] | None = None,
):
    """Add vertical cutoff lines with labels to a plot."""
    if not cutoff_dates:
        return

    # Group models by cutoff date
    cutoff_to_models: dict[datetime, list[str]] = {}
    for model_name, cutoff in cutoff_dates.items():
        if cutoff:
            if cutoff not in cutoff_to_models:
                cutoff_to_models[cutoff] = []
            cutoff_to_models[cutoff].append(model_name)

    # Draw cutoff lines with labels
    for idx, (cutoff, models) in enumerate(sorted(cutoff_to_models.items())):
        n = len(models)
        seg = 8  # dash segment length in points
        for i, model_name in enumerate(models):
            line_color = model_colors.get(model_name, 'gray')
            if n == 1:
                linestyle = (0, (1, 0))  # solid
            else:
                linestyle = (i * seg, (seg, seg * (n - 1)))
            ax.axvline(x=cutoff, color=line_color, linestyle=linestyle, linewidth=1.5, alpha=0.9)
        label_text = (cutoff_labels or CUTOFF_LABELS).get(cutoff) or ", ".join(get_display_name(m) for m in models)
        label_color = 'black' if len(models) > 1 else model_colors.get(models[0], 'gray')
        # Shift text left for specific models
        text_x = cutoff
        if any("gpt-5.2" in m.lower() for m in models):
            text_x = cutoff - timedelta(days=54)
        else:
            text_x = cutoff + timedelta(days=2)

        # ax.text(
        #     text_x,
        #     0.98 if kcd_position == 'top' else 0.0,
        #     f" {label_text}",
        #     fontsize=FONT_SIZES['cutoff_label'],
        #     color=label_color,
        #     ha='left',
        #     va=kcd_position,
        #     rotation=90,
        #     transform=ax.get_xaxis_transform(),
        #     fontweight='bold'
        # )


def add_mean_lines(ax, model_means: dict[str, float], model_colors: dict[str, str]):
    """Add horizontal mean lines for each model."""
    for model_name, mean_val in model_means.items():
        color = model_colors[model_name]
        ax.axhline(y=mean_val, color=color, linestyle='--', linewidth=2.0, alpha=0.3, zorder=0)


def setup_time_axis(ax):
    """Configure the x-axis for time series plots."""
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())


def set_dynamic_ylim(ax, all_values: list[float], min_val: float = 0):
    """Set y-axis limits based on actual data with 10% padding."""
    if not all_values:
        return
    y_min = min(all_values)
    y_max = max(all_values)
    padding = (y_max - y_min) * 0.1
    ax.set_ylim(max(min_val, y_min - padding), y_max + padding)


def finalize_plot(ax, xlabel: str, ylabel: str, title: str, legend_loc: str = 'lower left', legend_ncol: int = 3):
    """Apply common plot formatting."""
    ax.set_xlabel(xlabel, fontsize=FONT_SIZES['axis_label'])
    ax.set_ylabel(ylabel, fontsize=FONT_SIZES['axis_label'])
    if title:
        ax.set_title(title, fontsize=FONT_SIZES['title'])
    ax.legend(loc=legend_loc, fontsize=FONT_SIZES['legend'], ncol=legend_ncol, handlelength=1.0)
    ax.tick_params(axis='both', labelsize=FONT_SIZES['tick'])
    ax.grid(True, alpha=0.3)
    setup_time_axis(ax)
    plt.tight_layout()


def plot_multi_model_metric(
    all_model_data: dict[str, tuple[list, list]],
    output_path: Path,
    metric_name: str,
    title: str,
    ylabel: str,
    cutoff_dates: dict[str, datetime] | None = None,
    cutoff_labels: dict[datetime, str] | None = None,
    fixed_ylim: tuple[float, float] | None = None,
    max_jump: float = 0.15,
    legend_loc: str = 'lower left',
):
    """
    Generic function to plot a metric for multiple models.

    Args:
        all_model_data: Dict mapping model_name -> (dates, values)
        output_path: Path to save the plot
        metric_name: Name of the metric (for logging)
        title: Plot title
        ylabel: Y-axis label
        cutoff_dates: Optional dict mapping model names to cutoff dates
        fixed_ylim: Optional fixed y-axis limits (min, max). If None, uses dynamic limits.
        max_jump: Maximum allowed jump for spike filtering
        legend_loc: Legend location
    """
    model_colors = build_model_colors(list(all_model_data.keys()))

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    model_means = {}
    all_data_for_avg = []
    all_values = []

    # Sort models by this:
    # "Always Abstain" first
    # Claude and Llama-4 second
    # Gemini model versions in order
    # GPT model versions in order
    def model_sort_key(name: str) -> tuple[int, int, int, int]:
        if name == "Always Abstain":
            return (0, 0, 0, 0)
        if "claude" in name.lower() or "llama-4" in name.lower():
            return (1, 0, 0, 0)
        if "gemini" in name.lower():
            parts = name.lower().split('-')
            version_num = 0
            if len(parts) > 1:
                try:
                    version_num = int(parts[1].split('.')[0])
                except ValueError:
                    pass
            return (2, version_num, 0, 0)
        if "gpt" in name.lower():
            parts = name.lower().split('-')
            version_num = 0
            if len(parts) > 1:
                try:
                    version_num = int(parts[1].split('.')[0])
                except ValueError:
                    pass
            return (3, version_num, 0, 0)
        return (4, 0, 0, 0)

    sorted_models = sorted(all_model_data.keys(), key=model_sort_key)

    for model_name in sorted_models:
        dates, values = all_model_data[model_name]
        if dates:
            mean_val = sum(values) / len(values)
            if model_name == "Always Abstain":
                # Plot only mean line for "Always Abstain"
                ax.axhline(y=mean_val, color=ABSTAIN_COLOR, linestyle='--', linewidth=2.0,
                           label='Always NEI', alpha=0.6)
            else:
                color = model_colors[model_name]
                plot_dates, plot_vals = filter_spikes(dates, values, max_jump)
                display_name = get_display_name(model_name)
                cutoff = cutoff_dates.get(model_name) if cutoff_dates else None

                if cutoff:
                    cutoff_idx = next(
                        (i for i, d in enumerate(plot_dates) if d > cutoff),
                        len(plot_dates)
                    )
                    if cutoff_idx > 0:
                        ax.plot(plot_dates[:cutoff_idx], plot_vals[:cutoff_idx],
                                ':', linewidth=1.5, color=color, alpha=0.8)
                    if cutoff_idx < len(plot_dates):
                        start = max(0, cutoff_idx - 1)
                        ax.plot(plot_dates[start:], plot_vals[start:],
                                '-', linewidth=1.5, color=color,
                                label=display_name, alpha=0.8)
                else:
                    ax.plot(plot_dates, plot_vals, '-', linewidth=1.5,
                            color=color, label=display_name, alpha=0.8)

                model_means[model_name] = mean_val
                all_data_for_avg.append((dates, values))
                all_values.extend(values)

    # if all_data_for_avg:
    #     avg_dates, avg_values = compute_cross_model_average(all_data_for_avg, len(all_data_for_avg))
    #     plot_avg_dates, plot_avg_vals = filter_spikes(avg_dates, avg_values, max_jump)

    #     ax.plot(plot_avg_dates, plot_avg_vals, '-', linewidth=3, color='black',
    #            label='Average', alpha=0.9)

    #     # Overall average horizontal line
    #     valid_vals = [v for v in avg_values if not (isinstance(v, float) and v != v)]
    #     if valid_vals:
    #         overall_avg = sum(valid_vals) / len(valid_vals)
    #         ax.axhline(y=overall_avg, color='black', linestyle='--', linewidth=2.0, alpha=0.6)

    # Add horizontal mean lines for each model
    # add_mean_lines(ax, model_means, model_colors)

    # Add cutoff lines
    add_cutoff_lines(ax, cutoff_dates, model_colors, 'top' if metric_name == 'MAE' else 'bottom', cutoff_labels)

    # Set y-axis limits (explicit > metric default > dynamic)
    if fixed_ylim:
        ax.set_ylim(*fixed_ylim)
    elif metric_name in METRIC_YLIM and METRIC_YLIM[metric_name]:
        ax.set_ylim(*METRIC_YLIM[metric_name])
    else:
        set_dynamic_ylim(ax, all_values)

    finalize_plot(ax, 'Claim Date', ylabel, title, legend_loc, legend_ncol=2 if metric_name == 'MAE' or metric_name == 'Accuracy' else 3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def find_model_results(base_dir: Path) -> list[tuple[str, Path]]:
    """Find all model result files in a directory structure."""
    model_results = []
    for subdir in sorted(base_dir.iterdir()):
        if subdir.is_dir():
            results_csv = subdir / "results.csv"
            if results_csv.exists():
                model_name = subdir.name
                model_results.append((model_name, results_csv))
    return model_results
