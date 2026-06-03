#!/usr/bin/env python3
"""Generate confusion matrix plots in the human-evaluation style for all camera-ready experiments.

Uses distance-based coloring (green = diagonal, gray = off-by-one, red = far off) with
count-normalized alpha saturation, matching the style of
scripts/stats/human_evaluation/plot_human_vs_automatic.py.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

MODEL_DISPLAY_NAMES = {
    "Llama-4-Maverick-17B-128E-Instruct-FP8": "Llama 4 Maverick",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "gemini-3-pro-preview": "Gemini 3 Pro",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gpt-4.1": "GPT 4.1",
    "gpt-4o": "GPT 4o",
    "gpt-5.2": "GPT 5.2",
    "sonar-pro": "Sonar-Pro",
    "claude-opus-4-6": "Claude Opus 4.6",
    "google/gemma-4-31B-it": "Gemma 4",
    "gemma-4-31B-it": "Gemma 4",
    "Qwen3.5-397B-A17B-FP8": "Qwen 3.5",
}

# TUDa brand colors (same as scripts/stats/common.py)
COLORS = dict(
    negative="#E03440",
    neutral="#a1a1a1",
    positive="#09C479",
)

# 7-bin label order: negative → positive (indices 0–6 map to -1 … +1)
LABELS_7 = [
    "Compromised (certain)",
    "Compromised (rather certain)",
    "Compromised (rather uncertain)",
    "Unknown",
    "Intact (rather uncertain)",
    "Intact (rather certain)",
    "Intact (certain)",
]
TICK_LABELS_7 = ["-1", "-2/3", "-1/3", "0", "+1/3", "+2/3", "+1"]

# 3-bin label order: negative → positive (indices 0–2 map to -1 … +1)
LABELS_3 = ["Compromised", "Unknown", "Intact"]
TICK_LABELS_3 = ["-1", "0", "+1"]

COARSEN_7_TO_3 = {
    "Intact (certain)": "Intact",
    "Intact (rather certain)": "Intact",
    "Intact (rather uncertain)": "Unknown",
    "Unknown": "Unknown",
    "Compromised (rather uncertain)": "Unknown",
    "Compromised (rather certain)": "Compromised",
    "Compromised (certain)": "Compromised",
}


def build_confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> np.ndarray:
    label_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        ti = label_idx.get(t)
        pi = label_idx.get(p)
        if ti is not None and pi is not None:
            cm[ti, pi] += 1
    return cm


def _cell_color(i: int, j: int, n: int, max_count: int, count: int) -> tuple:
    pos_rgb = np.array(mcolors.to_rgb(COLORS["positive"]))
    neu_rgb = np.array(mcolors.to_rgb(COLORS["neutral"]))
    neg_rgb = np.array(mcolors.to_rgb(COLORS["negative"]))

    # Normalise distance so max distance (opposite corner) → 2
    max_dist = (n - 1) / (n / 2)
    distance = abs(i - j) / (n / 2)

    if 0 <= distance <= 1:
        base_rgb = pos_rgb * (1 - distance) + neu_rgb * distance
    else:
        base_rgb = neu_rgb * (2 - distance) + neg_rgb * (distance - 1)

    alpha = count / max_count if max_count > 0 else 0
    return base_rgb, alpha


def plot_cm(cm: np.ndarray, tick_labels: list[str], title: str, output_path: Path, figsize=(8, 7)):
    n = len(tick_labels)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    max_count = int(cm.max()) if cm.max() > 0 else 1

    for i in range(n):
        for j in range(n):
            count = int(cm[i, j])
            base_rgb, alpha = _cell_color(i, j, n, max_count, count)
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=base_rgb, alpha=alpha)
            ax.add_patch(rect)

            if count > 0:
                luminance = 0.299 * base_rgb[0] + 0.587 * base_rgb[1] + 0.114 * base_rgb[2]
                text_color = "white" if alpha > 0.5 and luminance < 0.5 else "black"
                ax.text(j, i, str(count), ha="center", va="center",
                        color=text_color, fontsize=22 if n <= 3 else 18, fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, fontsize=18)
    ax.set_yticklabels(tick_labels, fontsize=18)
    ax.set_xlabel("Prediction", fontsize=24)
    ax.set_ylabel("Ground Truth", fontsize=24)
    ax.set_title(title, fontsize=20, pad=20)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def process_experiment(exp_dir: Path):
    csv_path = exp_dir / "results.csv"
    if not csv_path.exists():
        return

    y_true_7, y_pred_7 = [], []
    y_true_3, y_pred_3 = [], []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "success":
                continue
            gt = row.get("gt_class", "").strip()
            pred = row.get("verdict", "").strip()
            if not gt or not pred:
                continue
            y_true_7.append(gt)
            y_pred_7.append(pred)
            y_true_3.append(COARSEN_7_TO_3.get(gt, gt))
            y_pred_3.append(COARSEN_7_TO_3.get(pred, pred))

    if not y_true_7:
        print(f"  No successful predictions in {exp_dir}, skipping")
        return

    # Detect which scheme is actually used
    unique_labels = set(y_true_7) | set(y_pred_7)
    is_7bin = bool(unique_labels & set(LABELS_7))
    is_3bin = bool(unique_labels & set(LABELS_3)) and not is_7bin

    model_name = exp_dir.name
    search_mode = exp_dir.parent.name
    search_label = "with Search" if search_mode == "custom-search" else "without Search"
    display_name = f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)} ({search_label})"

    if is_7bin or not is_3bin:
        cm7 = build_confusion_matrix(y_true_7, y_pred_7, LABELS_7)
        plot_cm(
            cm7,
            TICK_LABELS_7,
            display_name,
            exp_dir / f"confusion_matrix_{model_name}_human_style_7bin.pdf",
            figsize=(11, 10),
        )

    cm3 = build_confusion_matrix(y_true_3, y_pred_3, LABELS_3)
    plot_cm(
        cm3,
        TICK_LABELS_3,
        display_name,
        exp_dir / f"confusion_matrix_{model_name}_human_style_3bin.pdf",
        figsize=(8, 7),
    )


def main():
    base = Path(__file__).parent.parent / "results" / "camera_ready" / "q1-2026"
    if not base.exists():
        raise SystemExit(f"Results directory not found: {base}")

    for search_mode_dir in sorted(base.iterdir()):
        if not search_mode_dir.is_dir():
            continue
        for model_dir in sorted(search_mode_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            print(f"Processing {search_mode_dir.name}/{model_dir.name} …")
            process_experiment(model_dir)


if __name__ == "__main__":
    main()
