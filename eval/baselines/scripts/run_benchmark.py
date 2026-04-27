#!/usr/bin/env python3
"""
Unified benchmark runner for the VeriTaS dataset.

Supports running fact-checking with any combination of providers:
- OpenAI (GPT-4o, GPT-4.1, etc.)
- Gemini (gemini-2.5-flash, gemini-2.0-flash)
- Perplexity (sonar, sonar-pro)

Search modes:
- Default: Uses provider's built-in search (web_search_preview, Google Search grounding)
- Custom (--custom-search): Uses custom search with:
  - Date filtering (only results before claim date)
  - Full page content retrieval

Features:
- Run single provider or all providers
- Parallel processing with configurable workers
- Resume from previous runs
- CSV and JSONL output
- Confusion matrix visualization
- Metrics summary
"""

import argparse
import csv
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval.baselines import UnifiedFactChecker, FactCheckResult
from eval.baselines.common import (
    classify_integrity,
    compute_metrics,
    compute_regression_metrics,
    print_metrics,
    LABELS,
)
from eval.baselines.common.metrics import compute_coarsened_metrics, COARSEN_7_TO_3
from eval.baselines.common.types import get_label_scheme, get_property_label_scheme, LabelScheme

# Thread-safe CSV writing
csv_lock = Lock()

# =============================================================================
# CSV utilities
# =============================================================================

CSV_COLUMNS = [
    "claim_id",
    "claim_text",
    "has_media",
    "num_media",
    "gt_integrity",
    "gt_class",
    "provider",
    "model",
    "verdict",
    "pred_decisive_property",
    "first_error_step",
    "correct",
    "status",
    "error_message"
]

PROPERTY_POS_NEG_LABELS = {
    "authenticity": ("Pristine", "Fabricated"),
    "contextualization": ("Correct", "Incorrect"),
    "veracity": ("True", "False"),
    "context_coverage": ("Sufficient", "Insufficient"),
}

def extract_score(value) -> float | None:
    """Extract numeric score from scalar or {score: ...} objects."""
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def score_to_property_label(score: float | None, property_name: str, n_classes: int = 3) -> str | None:
    """Map score to property label (3-bin or 7-bin depending on n_classes)."""
    if score is None:
        return None
    pos_label, neg_label = PROPERTY_POS_NEG_LABELS[property_name]
    if n_classes == 3:
        if score > 1 / 3:
            return pos_label
        if score < -1 / 3:
            return neg_label
        return "Unknown"
    if n_classes == 7:
        if score < -5 / 6:
            return f"{neg_label} (certain)"
        if score < -3 / 6:
            return f"{neg_label} (rather certain)"
        if score < -1 / 6:
            return f"{neg_label} (rather uncertain)"
        if score > 5 / 6:
            return f"{pos_label} (certain)"
        if score > 3 / 6:
            return f"{pos_label} (rather certain)"
        if score > 1 / 6:
            return f"{pos_label} (rather uncertain)"
        return "Unknown"
    raise ValueError(f"Unsupported property label classes: {n_classes}")


def merge_usage(total_usage: dict, usage: dict):
    """Accumulate usage counters from heterogeneous provider schemas."""
    for key, value in (usage or {}).items():
        if isinstance(value, (int, float)):
            total_usage[key] = total_usage.get(key, 0) + value


def aggregate_integrity_from_staged(
    worst_contextualization_score: float | None,
    veracity_score: float | None,
    context_coverage_score: float | None,
    label_scheme: LabelScheme | None,
) -> tuple[str, float, str]:
    """Aggregate staged property scores into final integrity and decisive property."""
    candidates = []
    if context_coverage_score is not None:
        candidates.append(("context_coverage", context_coverage_score))
    if veracity_score is not None:
        candidates.append(("veracity", veracity_score))
    if worst_contextualization_score is not None:
        candidates.append(("contextualization", worst_contextualization_score))

    if not candidates:
        return "Unknown", 0.0, "none"

    decisive_property, decisive_score = min(candidates, key=lambda x: x[1])
    integrity_label = classify_integrity(decisive_score, label_scheme)
    return integrity_label, decisive_score, decisive_property


def compute_first_error_step(staged: dict) -> str | None:
    """Find first step where predicted staged decisions diverge from ground truth."""
    pred_stop_ctx = staged.get("pred_stop_after_contextualization")
    gt_stop_ctx = staged.get("gt_stop_after_contextualization")
    if pred_stop_ctx is not None and gt_stop_ctx is not None and pred_stop_ctx != gt_stop_ctx:
        return "contextualization_gate"

    pred_reached_ver = staged.get("pred_reached_veracity")
    gt_reached_ver = staged.get("gt_reached_veracity")
    if pred_reached_ver and gt_reached_ver:
        if staged.get("pred_veracity_label") != staged.get("gt_veracity_label"):
            return "veracity"

    pred_reached_cc = staged.get("pred_reached_context_coverage")
    gt_reached_cc = staged.get("gt_reached_context_coverage")
    if pred_reached_cc and gt_reached_cc:
        if staged.get("pred_context_coverage_label") != staged.get("gt_context_coverage_label"):
            return "context_coverage"

    if staged.get("gt_integrity_class") is not None and staged.get("pred_integrity") != staged.get("gt_integrity_class"):
        return "final_aggregation"
    return None


def init_csv(csv_path: Path):
    """Initialize CSV file with headers."""
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()


def append_to_csv(csv_path: Path, record: dict):
    """Append a single record to CSV."""
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow(record)


def load_processed_claim_ids(csv_path: Path, provider: str | None = None) -> set:
    """Load claim IDs that have already been processed from CSV."""
    if not csv_path.exists():
        return set()

    processed_ids = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if provider is None or row.get("provider") == provider:
                    processed_ids.add(int(row["claim_id"]))
            except (ValueError, KeyError):
                continue

    return processed_ids


def build_csv_record(
    claim_id: int,
    claim_text: str,
    media_refs: list,
    ground_truth: dict,
    result: FactCheckResult | None,
    provider: str,
    status: str,
    error_message: str | None = None,
    label_scheme: LabelScheme | None = None,
    staged: dict | None = None,
) -> dict:
    """Build a CSV record from claim processing results."""
    gt_integrity = ground_truth.get("integrity")
    gt_class = classify_integrity(gt_integrity, label_scheme) if gt_integrity is not None else None

    if status == "success" and result:
        verdict = result.verdict
        model = result.model
        correct = gt_class == verdict if gt_class else None
    else:
        verdict = None
        model = None
        correct = None

    return {
        "claim_id": claim_id,
        "claim_text": claim_text[:200] + "..." if len(claim_text) > 200 else claim_text,
        "has_media": len(media_refs) > 0 if media_refs else False,
        "num_media": len(media_refs) if media_refs else 0,
        "gt_integrity": round(gt_integrity, 4) if gt_integrity is not None else None,
        "gt_class": gt_class,
        "provider": provider,
        "model": model,
        "verdict": verdict,
        "pred_decisive_property": staged.get("pred_decisive_property") if staged else None,
        "first_error_step": staged.get("first_error_step") if staged else None,
        "correct": correct,
        "status": status,
        "error_message": error_message
    }


# =============================================================================
# Plotting utilities
# =============================================================================

def plot_confusion_matrix(metrics: dict, output_path: Path, provider: str, title_suffix: str = ""):
    """Generate and save confusion matrix plot.

    Args:
        metrics: Dict with at least 'accuracy', 'macro_f1', 'weighted_f1',
                 and 'confusion_matrix' keys.
        output_path: Where to save the PNG.
        provider: Provider name (used in the plot title).
        title_suffix: Optional suffix appended to the title (e.g. " (3-bin)").
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  Warning: matplotlib not installed, skipping confusion matrix plot")
        return

    if not metrics or "confusion_matrix" not in metrics:
        return

    labels = metrics["confusion_matrix"]["labels"]
    matrix = np.array(metrics["confusion_matrix"]["matrix"])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(len(labels)),
           yticks=np.arange(len(labels)),
           xticklabels=labels,
           yticklabels=labels,
           ylabel='True Label',
           xlabel='Predicted Label',
           title=f'{provider.title()} Confusion Matrix{title_suffix}')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = matrix.max() / 2.
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, format(matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > thresh else "black",
                    fontsize=14)

    metrics_text = f"Accuracy: {metrics['accuracy']:.2%}\nMacro F1: {metrics['macro_f1']:.4f}\nWeighted F1: {metrics['weighted_f1']:.4f}"
    plt.figtext(0.02, 0.02, metrics_text, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Confusion matrix plot saved to: {output_path}")


def plot_cross_confusion_matrix(metrics_7bin: dict, output_path: Path, provider: str):
    """Generate and save a 3-bin (GT) × 7-bin (predicted) cross confusion matrix.

    Rows are the coarsened 3-class ground-truth buckets; columns are the full
    7-class predicted labels.  Only meaningful when metrics_7bin contains a
    7-class confusion matrix (i.e. when running in 7-bin mode).

    Args:
        metrics_7bin: The full 7-class metrics dict (must contain 'confusion_matrix').
        output_path: Where to save the PNG.
        provider: Provider name (used in the plot title).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  Warning: matplotlib not installed, skipping cross confusion matrix plot")
        return

    if not metrics_7bin or "confusion_matrix" not in metrics_7bin:
        return

    labels_7 = metrics_7bin["confusion_matrix"]["labels"]
    matrix_7 = metrics_7bin["confusion_matrix"]["matrix"]

    # Only makes sense for 7-class output
    if len(labels_7) != 7:
        return

    labels_3 = ["Intact", "Unknown", "Compromised"]
    idx3 = {lbl: i for i, lbl in enumerate(labels_3)}

    cross = np.zeros((3, 7), dtype=int)
    for gt_idx, gt_label in enumerate(labels_7):
        coarsened = COARSEN_7_TO_3.get(gt_label, "Unknown")
        row = idx3[coarsened]
        for pred_idx, count in enumerate(matrix_7[gt_idx]):
            cross[row, pred_idx] += count

    row_totals = cross.sum(axis=1, keepdims=True)
    row_totals_safe = np.where(row_totals == 0, 1, row_totals)
    cross_pct = cross / row_totals_safe

    short = {
        "Intact (certain)":               "Intact\n(certain)",
        "Intact (rather certain)":        "Intact\n(r. certain)",
        "Intact (rather uncertain)":      "Intact\n(r. uncertain)",
        "Unknown":                        "Unknown",
        "Compromised (rather uncertain)": "Comp.\n(r. uncertain)",
        "Compromised (rather certain)":   "Comp.\n(r. certain)",
        "Compromised (certain)":          "Comp.\n(certain)",
    }
    col_labels = [short.get(l, l) for l in labels_7]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    im = ax.imshow(cross_pct, aspect='auto', interpolation='nearest',
                   cmap=plt.cm.Blues, vmin=0, vmax=cross_pct.max())
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Row proportion", fontsize=10)

    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(labels_3, fontsize=11, fontweight='bold')
    ax.set_xlabel("Predicted label (7-bin)", fontsize=11, labelpad=8)
    ax.set_ylabel("True label (3-bin)", fontsize=11, labelpad=8)
    ax.set_title(
        f"{provider.title()} Confusion Matrix (GT 3-bin × Pred 7-bin)",
        fontsize=12, pad=10,
    )

    thresh = cross_pct.max() / 2.0
    for i in range(3):
        for j in range(7):
            color = "white" if cross_pct[i, j] > thresh else "black"
            ax.text(j, i - 0.12, str(int(cross[i, j])),
                    ha="center", va="center", fontsize=11,
                    fontweight='bold', color=color)
            ax.text(j, i + 0.22, f"{cross_pct[i, j]:.0%}",
                    ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 3, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='minor', bottom=False, left=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Cross confusion matrix plot saved to: {output_path}")


def plot_failure_stage_distribution(failure_counts: dict[str, int], output_path: Path, provider: str):
    """Generate and save a bar chart of integrity failure stages."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Warning: matplotlib not installed, skipping failure stage plot")
        return

    if not failure_counts:
        return

    order = ["contextualization_gate", "veracity", "context_coverage", "final_aggregation", "unknown"]
    stages = [s for s in order if s in failure_counts]
    counts = [failure_counts[s] for s in stages]
    if not stages:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(stages, counts)
    ax.set_title(f"{provider.title()} Failure Stage Distribution")
    ax.set_xlabel("First Failing Stage")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Failure stage plot saved to: {output_path}")


# =============================================================================
# Summary generation
# =============================================================================

def generate_summary_from_csv(csv_path: Path, output_dir: Path, providers: list[str], label_scheme: LabelScheme | None = None):
    """Generate summary JSON and confusion matrix plots from CSV results."""
    if not csv_path.exists():
        print(f"  CSV file not found: {csv_path}")
        return

    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)

    # Group by provider
    summary = {"providers": {}}

    for provider in providers:
        provider_results = [r for r in results if r.get("provider") == provider]

        if not provider_results:
            continue

        total = len(provider_results)
        successful = sum(1 for r in provider_results if r["status"] == "success")
        errors = total - successful

        y_true = []
        y_pred = []
        gt_integrity_values = []
        verdict_values = []

        for r in provider_results:
            if r["status"] != "success":
                continue

            gt_class = r.get("gt_class")
            verdict = r.get("verdict")
            gt_integrity = r.get("gt_integrity")

            if gt_class and verdict:
                y_true.append(gt_class)
                y_pred.append(verdict)

            # Collect data for regression metrics
            if gt_integrity and verdict:
                try:
                    gt_integrity_values.append(float(gt_integrity))
                    verdict_values.append(verdict)
                except (ValueError, TypeError):
                    pass

        active_labels = label_scheme.labels if label_scheme else LABELS
        metrics = compute_metrics(y_true, y_pred, labels=active_labels)

        # Compute coarsened 3-bin metrics when running in 7-class mode
        coarsened = compute_coarsened_metrics(y_true, y_pred)
        if coarsened:
            metrics["coarsened_3class"] = coarsened

        # Compute regression metrics (MSE, MAE)
        regression_metrics = compute_regression_metrics(gt_integrity_values, verdict_values, scheme=label_scheme)
        if regression_metrics:
            metrics["mse"] = regression_metrics["mse"]
            metrics["mae"] = regression_metrics["mae"]

        provider_summary = {
            "total_claims": total,
            "successful_runs": successful,
            "errors": errors,
            "comparisons_available": len(y_true),
            "metrics": metrics,
        }
        if coarsened:
            provider_summary["metrics_3class"] = coarsened

        summary["providers"][provider] = provider_summary

        print(f"\n=== {provider.upper()} ===")
        print(f"  Total: {total}, Successful: {successful}, Errors: {errors}")

        if metrics:
            print_metrics(metrics)
            plot_confusion_matrix(metrics, output_dir / f"confusion_matrix_{provider}.png", provider)

            # Plot coarsened 3-bin confusion matrix when running in 7-class mode
            if coarsened:
                plot_confusion_matrix(
                    coarsened,
                    output_dir / f"confusion_matrix_{provider}_3bin.png",
                    provider,
                    title_suffix=" (coarsened 3-bin)",
                )
                plot_cross_confusion_matrix(
                    metrics,
                    output_dir / f"confusion_matrix_{provider}_3bin_vs_7bin.png",
                    provider,
                )

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSummary saved to: {summary_path}")
    return summary


def _binary_accuracy(items: list[bool]) -> float | None:
    if not items:
        return None
    return round(sum(1 for x in items if x) / len(items), 4)


def compute_property_coarsened_metrics(
    y_true: list[str],
    y_pred: list[str],
    property_name: str,
    property_label_classes: int,
) -> dict:
    """Coarsen 7-bin property labels to 3-bin and compute confusion metrics."""
    if property_label_classes != 7 or not y_true:
        return {}

    scheme_7 = get_property_label_scheme(property_name, 7)
    labels_3 = get_property_label_scheme(property_name, 3).labels

    y_true_3 = []
    y_pred_3 = []
    for gt, pred in zip(y_true, y_pred):
        gt_score = scheme_7.verdict_to_numeric.get(gt)
        pred_score = scheme_7.verdict_to_numeric.get(pred)
        if gt_score is None or pred_score is None:
            continue
        y_true_3.append(score_to_property_label(gt_score, property_name, n_classes=3))
        y_pred_3.append(score_to_property_label(pred_score, property_name, n_classes=3))

    if not y_true_3:
        return {}
    return compute_metrics(y_true_3, y_pred_3, labels=labels_3)


def generate_staged_summary_from_jsonl(
    jsonl_path: Path,
    output_dir: Path,
    providers: list[str],
    property_label_classes: int = 3,
):
    """Generate staged property metrics and failure-step attribution summaries."""
    if not jsonl_path.exists():
        return

    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    summary = {"providers": {}}
    for provider in providers:
        provider_rows = [
            r for r in rows
            if r.get("provider") == provider and r.get("status") == "success" and isinstance(r.get("staged"), dict)
        ]
        if not provider_rows:
            continue

        # Media-level metrics
        auth_true, auth_pred = [], []
        ctx_true, ctx_pred = [], []

        # Claim-level metrics and coverage
        ver_conditional_correct = []
        ver_end_to_end_correct = []
        ver_eligibility = 0
        ver_reached = 0
        ver_true_labels = []
        ver_pred_labels = []

        cc_conditional_correct = []
        cc_end_to_end_correct = []
        cc_eligibility = 0
        cc_reached = 0
        cc_true_labels = []
        cc_pred_labels = []

        decisive_correct = []
        failure_steps = Counter()

        for row in provider_rows:
            staged = row["staged"]
            for mp in staged.get("medium_predictions", []):
                gt_auth = mp.get("gt_authenticity")
                pred_auth = mp.get("pred_authenticity")
                if gt_auth and pred_auth:
                    auth_true.append(gt_auth)
                    auth_pred.append(pred_auth)

                gt_ctx = mp.get("gt_contextualization")
                pred_ctx = mp.get("pred_contextualization")
                if gt_ctx and pred_ctx:
                    ctx_true.append(gt_ctx)
                    ctx_pred.append(pred_ctx)

            if staged.get("pred_decisive_property") and staged.get("gt_decisive_property"):
                decisive_correct.append(staged["pred_decisive_property"] == staged["gt_decisive_property"])

            if staged.get("pred_integrity") != staged.get("gt_integrity_class"):
                step = staged.get("first_error_step") or "unknown"
                failure_steps[step] += 1

            gt_reached_ver = staged.get("gt_reached_veracity")
            pred_reached_ver = staged.get("pred_reached_veracity")
            if gt_reached_ver:
                ver_eligibility += 1
                if pred_reached_ver:
                    ver_reached += 1
                    pred_veracity = staged.get("pred_veracity_label")
                    gt_veracity = staged.get("gt_veracity_label")
                    ver_conditional_correct.append(pred_veracity == gt_veracity)
                    ver_end_to_end_correct.append(pred_veracity == gt_veracity)
                    if gt_veracity and pred_veracity:
                        ver_true_labels.append(gt_veracity)
                        ver_pred_labels.append(pred_veracity)
                else:
                    ver_end_to_end_correct.append(False)

            gt_reached_cc = staged.get("gt_reached_context_coverage")
            pred_reached_cc = staged.get("pred_reached_context_coverage")
            if gt_reached_cc:
                cc_eligibility += 1
                if pred_reached_cc:
                    cc_reached += 1
                    pred_cc = staged.get("pred_context_coverage_label")
                    gt_cc = staged.get("gt_context_coverage_label")
                    cc_conditional_correct.append(
                        pred_cc == gt_cc
                    )
                    cc_end_to_end_correct.append(
                        pred_cc == gt_cc
                    )
                    if gt_cc and pred_cc:
                        cc_true_labels.append(gt_cc)
                        cc_pred_labels.append(pred_cc)
                else:
                    cc_end_to_end_correct.append(False)

        auth_labels = get_property_label_scheme("authenticity", property_label_classes).labels
        ctx_labels = get_property_label_scheme("contextualization", property_label_classes).labels
        ver_labels = get_property_label_scheme("veracity", property_label_classes).labels
        cc_labels = get_property_label_scheme("context_coverage", property_label_classes).labels

        auth_metrics = compute_metrics(auth_true, auth_pred, labels=auth_labels) if auth_true else {}
        ctx_metrics = compute_metrics(ctx_true, ctx_pred, labels=ctx_labels) if ctx_true else {}
        ver_confusion = compute_metrics(ver_true_labels, ver_pred_labels, labels=ver_labels) if ver_true_labels else {}
        cc_confusion = compute_metrics(cc_true_labels, cc_pred_labels, labels=cc_labels) if cc_true_labels else {}
        auth_metrics_3bin = compute_property_coarsened_metrics(
            auth_true, auth_pred, "authenticity", property_label_classes
        )
        ctx_metrics_3bin = compute_property_coarsened_metrics(
            ctx_true, ctx_pred, "contextualization", property_label_classes
        )
        ver_confusion_3bin = compute_property_coarsened_metrics(
            ver_true_labels, ver_pred_labels, "veracity", property_label_classes
        )
        cc_confusion_3bin = compute_property_coarsened_metrics(
            cc_true_labels, cc_pred_labels, "context_coverage", property_label_classes
        )

        provider_summary = {
            "n_staged_successes": len(provider_rows),
            "property_metrics": {
                "authenticity": auth_metrics,
                "contextualization": ctx_metrics,
                "veracity": {
                    "eligible_claims": ver_eligibility,
                    "reached_claims": ver_reached,
                    "coverage": round(ver_reached / ver_eligibility, 4) if ver_eligibility else None,
                    "conditional_accuracy": _binary_accuracy(ver_conditional_correct),
                    "end_to_end_accuracy": _binary_accuracy(ver_end_to_end_correct),
                    "conditional_confusion": ver_confusion,
                    "conditional_confusion_3bin": ver_confusion_3bin,
                },
                "context_coverage": {
                    "eligible_claims": cc_eligibility,
                    "reached_claims": cc_reached,
                    "coverage": round(cc_reached / cc_eligibility, 4) if cc_eligibility else None,
                    "conditional_accuracy": _binary_accuracy(cc_conditional_correct),
                    "end_to_end_accuracy": _binary_accuracy(cc_end_to_end_correct),
                    "conditional_confusion": cc_confusion,
                    "conditional_confusion_3bin": cc_confusion_3bin,
                },
            },
            "property_metrics_3bin": {
                "authenticity": auth_metrics_3bin,
                "contextualization": ctx_metrics_3bin,
            },
            "integrity_error_attribution": dict(failure_steps),
            "decisive_property_accuracy": _binary_accuracy(decisive_correct),
        }
        summary["providers"][provider] = provider_summary

        if auth_metrics:
            plot_confusion_matrix(
                auth_metrics,
                output_dir / f"confusion_matrix_{provider}_authenticity.png",
                provider,
                title_suffix=" - Authenticity",
            )
        if ctx_metrics:
            plot_confusion_matrix(
                ctx_metrics,
                output_dir / f"confusion_matrix_{provider}_contextualization.png",
                provider,
                title_suffix=" - Contextualization",
            )
        if auth_metrics_3bin:
            plot_confusion_matrix(
                auth_metrics_3bin,
                output_dir / f"confusion_matrix_{provider}_authenticity_3bin.png",
                provider,
                title_suffix=" - Authenticity (coarsened 3-bin)",
            )
        if ctx_metrics_3bin:
            plot_confusion_matrix(
                ctx_metrics_3bin,
                output_dir / f"confusion_matrix_{provider}_contextualization_3bin.png",
                provider,
                title_suffix=" - Contextualization (coarsened 3-bin)",
            )
        if ver_confusion:
            plot_confusion_matrix(
                ver_confusion,
                output_dir / f"confusion_matrix_{provider}_veracity.png",
                provider,
                title_suffix=" - Veracity (conditional)",
            )
        if ver_confusion_3bin:
            plot_confusion_matrix(
                ver_confusion_3bin,
                output_dir / f"confusion_matrix_{provider}_veracity_3bin.png",
                provider,
                title_suffix=" - Veracity (conditional, coarsened 3-bin)",
            )
        if cc_confusion:
            plot_confusion_matrix(
                cc_confusion,
                output_dir / f"confusion_matrix_{provider}_context_coverage.png",
                provider,
                title_suffix=" - Context Coverage (conditional)",
            )
        if cc_confusion_3bin:
            plot_confusion_matrix(
                cc_confusion_3bin,
                output_dir / f"confusion_matrix_{provider}_context_coverage_3bin.png",
                provider,
                title_suffix=" - Context Coverage (conditional, coarsened 3-bin)",
            )
        if failure_steps:
            plot_failure_stage_distribution(
                dict(failure_steps),
                output_dir / f"failure_stage_distribution_{provider}.png",
                provider,
            )

    summary_path = output_dir / "summary_staged.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Staged summary saved to: {summary_path}")
    return summary


def _metric_support(metrics: dict) -> int | None:
    if not metrics:
        return None
    per_class = metrics.get("per_class")
    if not isinstance(per_class, dict):
        return None
    return int(sum(v.get("support", 0) for v in per_class.values() if isinstance(v, dict)))


def _fmt_metric(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _compute_mse_mae_from_confusion(metrics: dict, label_to_numeric: dict[str, float]) -> tuple[float | None, float | None]:
    """Compute MSE/MAE from a confusion matrix using label->numeric mapping."""
    cm = (metrics or {}).get("confusion_matrix", {})
    labels = cm.get("labels") if isinstance(cm, dict) else None
    matrix = cm.get("matrix") if isinstance(cm, dict) else None
    if not labels or not matrix:
        return None, None

    total = 0
    sse = 0.0
    sae = 0.0
    for i, gt_label in enumerate(labels):
        gt_val = label_to_numeric.get(gt_label)
        if gt_val is None:
            continue
        for j, pred_label in enumerate(labels):
            pred_val = label_to_numeric.get(pred_label)
            if pred_val is None:
                continue
            count = matrix[i][j]
            if not count:
                continue
            diff = gt_val - pred_val
            sse += count * (diff ** 2)
            sae += count * abs(diff)
            total += count

    if total == 0:
        return None, None
    return round(sse / total, 4), round(sae / total, 4)


def write_property_metrics_table(
    output_dir: Path,
    providers: list[str],
    integrity_summary: dict | None,
    staged_summary: dict | None,
    property_label_classes: int,
):
    """Write property metrics as per-property tables (rows=metrics, cols=binning)."""
    metric_order = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "support",
        "coverage",
        "conditional_accuracy",
        "end_to_end_accuracy",
        "mse",
        "mae",
    ]
    metric_names = {
        "accuracy": "Accuracy",
        "macro_f1": "Macro F1",
        "weighted_f1": "Weighted F1",
        "support": "Support",
        "coverage": "Coverage",
        "conditional_accuracy": "Conditional Accuracy",
        "end_to_end_accuracy": "End-to-End Accuracy",
        "mse": "MSE",
        "mae": "MAE",
    }
    bin_columns = ["3-bin", "7-bin"] if property_label_classes == 7 else ["3-bin"]
    properties = ["integrity", "authenticity", "contextualization", "veracity", "context_coverage"]
    provider_tables: dict[str, dict[str, dict[str, dict]]] = {}

    def add_metrics_block(
        provider_data: dict,
        property_name: str,
        binning: str,
        metrics: dict,
        label_to_numeric: dict[str, float],
        extra: dict | None = None,
    ):
        if not metrics:
            return
        if property_name not in provider_data:
            provider_data[property_name] = {}
        entry = {
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "weighted_f1": metrics.get("weighted_f1"),
            "support": _metric_support(metrics),
        }
        mse, mae = _compute_mse_mae_from_confusion(metrics, label_to_numeric)
        entry["mse"] = mse
        entry["mae"] = mae
        if extra:
            entry.update(extra)
        provider_data[property_name][binning] = entry

    for provider in providers:
        provider_data: dict[str, dict[str, dict]] = {}
        provider_tables[provider] = provider_data

        integrity_provider = ((integrity_summary or {}).get("providers") or {}).get(provider, {})
        integrity_metrics = integrity_provider.get("metrics", {})
        if property_label_classes == 7:
            add_metrics_block(
                provider_data,
                "integrity",
                "7-bin",
                integrity_metrics,
                get_label_scheme(7).verdict_to_numeric,
            )
            integrity_3 = integrity_provider.get("metrics_3class") or integrity_metrics.get("coarsened_3class")
            add_metrics_block(
                provider_data,
                "integrity",
                "3-bin",
                integrity_3,
                get_label_scheme(3).verdict_to_numeric,
            )
        else:
            add_metrics_block(
                provider_data,
                "integrity",
                "3-bin",
                integrity_metrics,
                get_label_scheme(3).verdict_to_numeric,
            )

        staged_provider = ((staged_summary or {}).get("providers") or {}).get(provider, {})
        property_metrics = staged_provider.get("property_metrics", {})
        property_metrics_3bin = staged_provider.get("property_metrics_3bin", {})

        for prop in ("authenticity", "contextualization"):
            if property_label_classes == 7:
                add_metrics_block(
                    provider_data,
                    prop,
                    "7-bin",
                    property_metrics.get(prop, {}),
                    get_property_label_scheme(prop, 7).verdict_to_numeric,
                )
                add_metrics_block(
                    provider_data,
                    prop,
                    "3-bin",
                    property_metrics_3bin.get(prop, {}),
                    get_property_label_scheme(prop, 3).verdict_to_numeric,
                )
            else:
                add_metrics_block(
                    provider_data,
                    prop,
                    "3-bin",
                    property_metrics.get(prop, {}),
                    get_property_label_scheme(prop, 3).verdict_to_numeric,
                )

        for prop in ("veracity", "context_coverage"):
            block = property_metrics.get(prop, {})
            extra = {
                "coverage": block.get("coverage"),
                "conditional_accuracy": block.get("conditional_accuracy"),
                "end_to_end_accuracy": block.get("end_to_end_accuracy"),
            }
            if property_label_classes == 7:
                add_metrics_block(
                    provider_data,
                    prop,
                    "7-bin",
                    block.get("conditional_confusion", {}),
                    get_property_label_scheme(prop, 7).verdict_to_numeric,
                    extra=extra,
                )
                add_metrics_block(
                    provider_data,
                    prop,
                    "3-bin",
                    block.get("conditional_confusion_3bin", {}),
                    get_property_label_scheme(prop, 3).verdict_to_numeric,
                    extra=extra,
                )
            else:
                add_metrics_block(
                    provider_data,
                    prop,
                    "3-bin",
                    block.get("conditional_confusion", {}),
                    get_property_label_scheme(prop, 3).verdict_to_numeric,
                    extra=extra,
                )

    if not any(provider_tables.values()):
        return

    def write_property_metrics_pdf(pdf_path: Path):
        """Render one table per property into a single PDF document."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
        except ImportError:
            print("  Warning: matplotlib not installed, skipping property metrics PDF")
            return

        with PdfPages(pdf_path) as pdf:
            for provider in providers:
                provider_data = provider_tables.get(provider, {})
                for prop in properties:
                    prop_data = provider_data.get(prop, {})
                    if not prop_data:
                        continue
                    present_bins = [b for b in bin_columns if b in prop_data]
                    if not present_bins:
                        continue

                    present_metrics = []
                    for m in metric_order:
                        if any(prop_data[b].get(m) is not None for b in present_bins):
                            present_metrics.append(m)
                    if not present_metrics:
                        continue

                    fig, ax = plt.subplots(figsize=(11, 8.5))
                    ax.axis("off")
                    fig.suptitle(f"{provider} - {prop}", fontsize=14, fontweight="bold")

                    col_labels = ["Metric"] + present_bins
                    cell_text = []
                    for m in present_metrics:
                        row = [metric_names[m]]
                        for b in present_bins:
                            row.append(_fmt_metric(prop_data[b].get(m)))
                        cell_text.append(row)

                    table = ax.table(
                        cellText=cell_text,
                        colLabels=col_labels,
                        loc="center",
                        cellLoc="center",
                        colLoc="center",
                    )
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    table.scale(1, 1.6)
                    for (r, c), cell in table.get_celld().items():
                        if r == 0:
                            cell.set_text_props(weight="bold")
                            cell.set_facecolor("#EAEAEA")

                    plt.tight_layout(rect=[0, 0, 1, 0.95])
                    pdf.savefig(fig)
                    plt.close(fig)

    # Markdown and CSV (provider sections, no provider column in tables)
    md_path = output_dir / "property_metrics_table.md"
    with open(md_path, "w", encoding="utf-8") as f_md:
        f_md.write("# Property Metrics Summary\n\n")
        for provider in providers:
            provider_data = provider_tables.get(provider, {})
            if not provider_data:
                continue
            f_md.write(f"## Provider: {provider}\n\n")
            for prop in properties:
                prop_data = provider_data.get(prop, {})
                if not prop_data:
                    continue
                present_bins = [b for b in bin_columns if b in prop_data]
                if not present_bins:
                    continue
                present_metrics = [
                    m for m in metric_order
                    if any(prop_data[b].get(m) is not None for b in present_bins)
                ]
                if not present_metrics:
                    continue

                f_md.write(f"### {prop}\n\n")
                header = "| Metric | " + " | ".join(present_bins) + " |\n"
                divider = "|" + "---|" * (1 + len(present_bins)) + "\n"
                f_md.write(header)
                f_md.write(divider)
                for m in present_metrics:
                    row = [metric_names[m]] + [_fmt_metric(prop_data[b].get(m)) for b in present_bins]
                    f_md.write("| " + " | ".join(row) + " |\n")
                f_md.write("\n")

    csv_path = output_dir / "property_metrics_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        for provider in providers:
            provider_data = provider_tables.get(provider, {})
            if not provider_data:
                continue
            writer.writerow([f"Provider: {provider}"])
            for prop in properties:
                prop_data = provider_data.get(prop, {})
                if not prop_data:
                    continue
                present_bins = [b for b in bin_columns if b in prop_data]
                if not present_bins:
                    continue
                present_metrics = [
                    m for m in metric_order
                    if any(prop_data[b].get(m) is not None for b in present_bins)
                ]
                if not present_metrics:
                    continue
                writer.writerow([prop])
                writer.writerow(["metric"] + present_bins)
                for m in present_metrics:
                    writer.writerow([metric_names[m]] + [prop_data[b].get(m) for b in present_bins])
                writer.writerow([])
            writer.writerow([])

    pdf_path = output_dir / "property_metrics_table.pdf"
    write_property_metrics_pdf(pdf_path)

    print(f"Property metrics table saved to: {csv_path}")
    print(f"Property metrics table saved to: {md_path}")
    print(f"Property metrics table saved to: {pdf_path}")


# =============================================================================
# Main processing
# =============================================================================

def load_dataset(dataset_path: Path) -> dict:
    """Load the VeriTaS benchmark dataset."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def init_staged_checkers(
    providers: list[str],
    models: dict[str, str] | None,
    custom_search: bool,
    use_search: bool,
    property_label_classes: int = 3,
    seven_bin_prediction_mode: str = "direct",
) -> dict[str, dict[str, UnifiedFactChecker]]:
    """Initialize per-provider, per-property checkers for staged mode."""
    per_property_checkers = {}
    for provider in providers:
        per_property_checkers[provider] = {}
        provider_model = (models or {}).get(provider)
        for property_name in ("authenticity", "contextualization", "veracity", "context_coverage"):
            per_property_checkers[provider][property_name] = UnifiedFactChecker(
                provider=provider,
                model=provider_model,
                custom_search=custom_search,
                use_search=use_search,
                label_scheme=get_property_label_scheme(property_name, n_classes=property_label_classes),
                seven_bin_prediction_mode=seven_bin_prediction_mode,
            )
    return per_property_checkers


def run_staged_fact_check(
    claim_text: str,
    claim_date: str | None,
    media_items: list[dict],
    image_paths: list[str],
    video_paths: list[str],
    provider: str,
    label_scheme: LabelScheme | None,
    staged_checkers: dict[str, dict[str, UnifiedFactChecker]],
    property_label_classes: int = 3,
) -> tuple[FactCheckResult, dict]:
    """Run staged property checks mirroring the human/pipeline gating logic."""
    checker_map = staged_checkers[provider]
    citations = []
    total_usage: dict = {}
    medium_predictions = []
    contextualization_scores = []
    contextualization_negative = False
    all_models = set()

    # Per-medium steps: authenticity + contextualization
    image_idx = 0
    video_idx = 0
    for media_item in media_items:
        media_type = media_item.get("type")
        path = None
        if media_type == "image":
            path = image_paths[image_idx] if image_idx < len(image_paths) else None
            image_idx += 1
        elif media_type == "video":
            path = video_paths[video_idx] if video_idx < len(video_paths) else None
            video_idx += 1

        current_image_paths = [path] if media_type == "image" and path else None
        current_video_paths = [path] if media_type == "video" and path else None

        auth_result = checker_map["authenticity"].check_claim(
            claim_text,
            image_paths=current_image_paths,
            video_paths=current_video_paths,
            claim_date=claim_date,
        )
        ctx_result = checker_map["contextualization"].check_claim(
            claim_text,
            image_paths=current_image_paths,
            video_paths=current_video_paths,
            claim_date=claim_date,
        )

        for result in (auth_result, ctx_result):
            citations.extend([c for c in result.citations if c not in citations])
            merge_usage(total_usage, result.usage)
            if result.model:
                all_models.add(result.model)

        auth_score = checker_map["authenticity"].get_provider().label_scheme.verdict_to_numeric.get(auth_result.verdict)
        ctx_score = checker_map["contextualization"].get_provider().label_scheme.verdict_to_numeric.get(ctx_result.verdict)
        contextualization_scores.append(ctx_score if ctx_score is not None else 0.0)
        if ctx_score is not None and ctx_score < -1 / 3:
            contextualization_negative = True

        gt_auth_label = score_to_property_label(
            extract_score(media_item.get("authenticity")), "authenticity", n_classes=property_label_classes
        )
        gt_ctx_label = score_to_property_label(
            extract_score(media_item.get("contextualization")), "contextualization", n_classes=property_label_classes
        )

        medium_predictions.append({
            "media_id": media_item.get("id"),
            "media_type": media_type,
            "pred_authenticity": auth_result.verdict,
            "gt_authenticity": gt_auth_label,
            "pred_contextualization": ctx_result.verdict,
            "gt_contextualization": gt_ctx_label,
        })

    pred_veracity_label = None
    pred_context_coverage_label = None
    pred_veracity_score = None
    pred_context_coverage_score = None
    reached_veracity = False
    reached_context_coverage = False

    # Claim-level veracity (only if no incorrect contextualization)
    if not contextualization_negative:
        reached_veracity = True
        ver_result = checker_map["veracity"].check_claim(
            claim_text,
            image_paths=image_paths if image_paths else None,
            video_paths=video_paths if video_paths else None,
            claim_date=claim_date,
        )
        citations.extend([c for c in ver_result.citations if c not in citations])
        merge_usage(total_usage, ver_result.usage)
        if ver_result.model:
            all_models.add(ver_result.model)

        pred_veracity_label = ver_result.verdict
        pred_veracity_score = checker_map["veracity"].get_provider().label_scheme.verdict_to_numeric.get(ver_result.verdict)

        # Context coverage only if veracity is positive
        if pred_veracity_score is not None and pred_veracity_score > 0:
            reached_context_coverage = True
            cc_result = checker_map["context_coverage"].check_claim(
                claim_text,
                image_paths=image_paths if image_paths else None,
                video_paths=video_paths if video_paths else None,
                claim_date=claim_date,
            )
            citations.extend([c for c in cc_result.citations if c not in citations])
            merge_usage(total_usage, cc_result.usage)
            if cc_result.model:
                all_models.add(cc_result.model)

            pred_context_coverage_label = cc_result.verdict
            pred_context_coverage_score = checker_map["context_coverage"].get_provider().label_scheme.verdict_to_numeric.get(
                cc_result.verdict
            )

    worst_ctx_score = min(contextualization_scores) if contextualization_scores else None
    pred_integrity, pred_integrity_score, pred_decisive_property = aggregate_integrity_from_staged(
        worst_contextualization_score=worst_ctx_score,
        veracity_score=pred_veracity_score,
        context_coverage_score=pred_context_coverage_score,
        label_scheme=label_scheme,
    )

    reasoning = (
        f"Staged loop executed. decisive_property={pred_decisive_property}, "
        f"contextualization_negative={contextualization_negative}, "
        f"reached_veracity={reached_veracity}, reached_context_coverage={reached_context_coverage}."
    )
    model_name = "+".join(sorted(all_models)) if all_models else ""
    result = FactCheckResult(
        verdict=pred_integrity,
        reasoning=reasoning,
        citations=citations,
        model=model_name,
        provider=provider,
        usage=total_usage,
    )

    staged = {
        "mode": "staged",
        "property_label_classes": property_label_classes,
        "medium_predictions": medium_predictions,
        "pred_worst_contextualization_label": score_to_property_label(
            worst_ctx_score, "contextualization", n_classes=property_label_classes
        ),
        "pred_worst_contextualization_score": worst_ctx_score,
        "pred_veracity_label": pred_veracity_label,
        "pred_veracity_score": pred_veracity_score,
        "pred_context_coverage_label": pred_context_coverage_label,
        "pred_context_coverage_score": pred_context_coverage_score,
        "pred_stop_after_contextualization": contextualization_negative,
        "pred_reached_veracity": reached_veracity,
        "pred_reached_context_coverage": reached_context_coverage,
        "pred_integrity": pred_integrity,
        "pred_integrity_score": pred_integrity_score,
        "pred_decisive_property": pred_decisive_property,
    }
    return result, staged


def process_single_claim(
    claim: dict,
    idx: int,
    total: int,
    fc: UnifiedFactChecker | None,
    provider: str,
    dataset_dir: Path,
    csv_path: Path,
    jsonl_path: Path,
    label_scheme: LabelScheme | None = None,
    mode: str = "integrity",
    staged_checkers: dict[str, dict[str, UnifiedFactChecker]] | None = None,
    property_label_classes: int = 3,
) -> dict:
    """Process a single claim and return results."""
    claim_id = claim["id"]
    claim_text = claim["text"]
    claim_date = claim.get("date")
    # Build ground_truth dict compatible with existing code
    # New format has integrity/veracity/context_coverage at claim level
    integrity_data = claim.get("integrity", {})
    ground_truth = {
        "integrity": integrity_data.get("score") if isinstance(integrity_data, dict) else integrity_data,
        "veracity": claim.get("veracity"),
        "context_coverage": claim.get("context_coverage"),
        "media": claim.get("media"),
    }

    print(f"\n[{idx + 1}/{total}] [{provider}] Claim ID: {claim_id}")
    print(f"  Text: {claim_text[:80]}..." if len(claim_text) > 80 else f"  Text: {claim_text}")

    # New format: media is directly in claim["media"] array with file_path
    media_items = claim.get("media", [])
    media_refs = media_items  # Keep reference for CSV record
    image_paths = []
    video_paths = []

    staged = None
    try:
        # Resolve media files if present
        if media_items:
            print(f"  Found {len(media_items)} media item(s)")
            for item in media_items:
                file_path = item.get("file_path")
                if file_path:
                    full_path = dataset_dir / file_path
                    if full_path.exists():
                        if item.get("type") == "image":
                            image_paths.append(str(full_path))
                        elif item.get("type") == "video":
                            video_paths.append(str(full_path))
                    else:
                        print(f"    Warning: Media file not found: {full_path}")

            if provider == "gemini" and video_paths:
                print(f"  Using native video processing for Gemini")

        if mode == "staged":
            if not staged_checkers:
                raise ValueError("Staged mode requested but no staged_checkers were provided")
            result, staged = run_staged_fact_check(
                claim_text=claim_text,
                claim_date=claim_date,
                media_items=media_items,
                image_paths=image_paths,
                video_paths=video_paths,
                provider=provider,
                label_scheme=label_scheme,
                staged_checkers=staged_checkers,
                property_label_classes=property_label_classes,
            )
        else:
            # Run standard integrity-only fact-check
            result = fc.check_claim(
                claim_text,
                image_paths=image_paths if image_paths else None,
                video_paths=video_paths if video_paths else None,
                claim_date=claim_date,
                provider=provider,
            )

        status = "success"
        error_message = None
        print(f"  -> Verdict: {result.verdict}")

    except Exception as e:
        print(f"  -> ERROR: {str(e)}")
        result = None
        status = "error"
        error_message = str(e)

    if status == "success" and staged is not None:
        gt_veracity_score = extract_score(ground_truth.get("veracity"))
        gt_context_coverage_score = extract_score(ground_truth.get("context_coverage"))
        gt_veracity_label = score_to_property_label(
            gt_veracity_score, "veracity", n_classes=property_label_classes
        )
        gt_context_coverage_label = score_to_property_label(
            gt_context_coverage_score, "context_coverage", n_classes=property_label_classes
        )

        gt_contextualization_scores = []
        for media in media_items:
            s = extract_score(media.get("contextualization"))
            if s is not None:
                gt_contextualization_scores.append(s)
        gt_worst_ctx_score = min(gt_contextualization_scores) if gt_contextualization_scores else None
        gt_worst_ctx_label = score_to_property_label(
            gt_worst_ctx_score, "contextualization", n_classes=property_label_classes
        )
        gt_stop_after_ctx = gt_worst_ctx_score is not None and gt_worst_ctx_score < -1 / 3
        gt_reached_veracity = not gt_stop_after_ctx
        gt_reached_context_coverage = gt_reached_veracity and gt_veracity_score is not None and gt_veracity_score > 0

        gt_integrity_score = extract_score(ground_truth.get("integrity"))
        gt_integrity_class = classify_integrity(gt_integrity_score, label_scheme) if gt_integrity_score is not None else None
        gt_decisive_property = None
        if isinstance(integrity_data, dict):
            gt_decisive_property = integrity_data.get("decisive_property")

        staged.update({
            "gt_veracity_label": gt_veracity_label,
            "gt_context_coverage_label": gt_context_coverage_label,
            "gt_worst_contextualization_label": gt_worst_ctx_label,
            "gt_stop_after_contextualization": gt_stop_after_ctx,
            "gt_reached_veracity": gt_reached_veracity,
            "gt_reached_context_coverage": gt_reached_context_coverage,
            "gt_integrity_class": gt_integrity_class,
            "gt_decisive_property": gt_decisive_property,
        })
        staged["first_error_step"] = compute_first_error_step(staged)

    # Build CSV record
    csv_record = build_csv_record(
        claim_id=claim_id,
        claim_text=claim_text,
        media_refs=media_refs,
        ground_truth=ground_truth,
        result=result,
        provider=provider,
        status=status,
        error_message=error_message,
        label_scheme=label_scheme,
        staged=staged,
    )

    # Build JSONL record
    jsonl_record = {
        "claim_id": claim_id,
        "claim_text": claim_text,
        "claim_text_clean": claim_text,  # In new format, text is already clean
        "claim_date": claim_date,
        "num_images": len(image_paths),
        "num_videos": len(video_paths),
        "ground_truth": ground_truth,
        "provider": provider,
        "result": result.to_dict() if result else None,
        "staged": staged,
        "status": status,
        "error_message": error_message
    }

    # Thread-safe file writes
    with csv_lock:
        append_to_csv(csv_path, csv_record)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(jsonl_record, ensure_ascii=False) + "\n")

    return csv_record


def run_benchmark(
    dataset_path: str,
    output_dir: str | None = None,
    resume_dir: str | None = None,
    providers: list[str] | None = None,
    models: dict[str, str] | None = None,
    limit: int | None = None,
    num_workers: int = 1,
    custom_search: bool = False,
    use_search: bool = True,
    label_scheme: LabelScheme | None = None,
    mode: str = "integrity",
    seven_bin_prediction_mode: str = "direct",
):
    """
    Run the fact-checker benchmark on VeriTaS claims.

    Args:
        dataset_path: Path to claims.json
        output_dir: Directory to save results (auto-generated if None)
        resume_dir: Directory of previous run to resume from
        providers: List of providers to use (default: all)
        models: Dict mapping provider names to model identifiers
        limit: Maximum number of claims to process (None = all)
        num_workers: Number of parallel workers
        custom_search: If True, use custom search with date filtering
        use_search: If True (default), use web search. If False, use parametric knowledge only.
        label_scheme: Label scheme to use (3-class or 7-class). Defaults to 3-class.
        mode: "integrity" (current baseline) or "staged" (full property loop).
        seven_bin_prediction_mode: For 7-class schemes, "direct" asks the model
                                  for a combined label, while "two_step" asks for
                                  direction + certainty in one response.
    """
    # Default to all providers
    if providers is None:
        providers = ["openai", "gemini", "perplexity"]

    # Setup paths
    dataset_path = Path(dataset_path)
    dataset_dir = dataset_path.parent

    # Default models for naming
    default_models = {
        "openai": "gpt-5.2",
        "gemini": "gemini-2.5-flash",
        "perplexity": "sonar-pro",
        "llama": "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    }

    # Determine output directory
    if resume_dir:
        output_dir = Path(resume_dir)
        print(f"Resuming from: {output_dir}")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        # Build model name for directory
        if len(providers) == 1:
            model_name = (models or {}).get(providers[0], default_models.get(providers[0], providers[0]))
        else:
            model_name = "+".join(providers)
        # Add search mode suffix
        if not use_search:
            model_name += "_no-search"
        elif custom_search:
            model_name += "_custom-search"
        if mode == "staged":
            model_name += "_staged"
        output_dir = Path(__file__).parent.parent / "results" / f"{model_name}_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # File paths
    csv_path = output_dir / "results.csv"
    jsonl_path = output_dir / "results_detailed.jsonl"
    config_path = output_dir / "run_config.json"

    # Save run configuration immediately so it exists even if the run crashes
    started_at = datetime.now().isoformat()
    run_config = {
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "providers": providers,
        "models": models,
        "limit": limit,
        "num_workers": num_workers,
        "custom_search": custom_search,
        "use_search": use_search,
        "mode": mode,
        "label_scheme": label_scheme.name if label_scheme else "3-class",
        "seven_bin_prediction_mode": seven_bin_prediction_mode,
        "resumed_from": str(resume_dir) if resume_dir else None,
        "started_at": started_at,
        "finished_at": None,
    }
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    data = load_dataset(dataset_path)
    all_claims = data["claims"]
    print(f"Total claims in dataset: {len(all_claims)}")

    # Initialize fact-checker with all requested providers
    if not use_search:
        search_mode = "disabled (parametric knowledge only)"
    elif custom_search:
        search_mode = "custom (date filter + content retrieval)"
    else:
        search_mode = "built-in"
    label_scheme_name = label_scheme.name if label_scheme else "3-class"
    property_label_classes = len(label_scheme.labels) if label_scheme and len(label_scheme.labels) in (3, 7) else 3
    print(f"\nInitializing providers: {', '.join(providers)}")
    print(f"Search mode: {search_mode}")
    print(f"Label scheme: {label_scheme_name}")
    print(f"Benchmark mode: {mode}")
    print(f"7-bin prediction mode: {seven_bin_prediction_mode}")
    fc = None
    staged_checkers = None
    if mode == "staged":
        staged_checkers = init_staged_checkers(
            providers=providers,
            models=models,
            custom_search=custom_search,
            use_search=use_search,
            property_label_classes=property_label_classes,
            seven_bin_prediction_mode=seven_bin_prediction_mode,
        )
    else:
        fc = UnifiedFactChecker(
            providers=providers,
            models=models,
            custom_search=custom_search,
            use_search=use_search,
            label_scheme=label_scheme,
            seven_bin_prediction_mode=seven_bin_prediction_mode,
        )

    # Process each provider
    for provider in providers:
        print(f"\n{'='*60}")
        print(f"Processing with {provider.upper()}")
        print(f"{'='*60}")

        # Load already processed claim IDs for this provider
        processed_ids = set()
        if resume_dir and csv_path.exists():
            processed_ids = load_processed_claim_ids(csv_path, provider)
            print(f"Already processed: {len(processed_ids)} claims")
        elif not csv_path.exists():
            init_csv(csv_path)

        # Filter out already processed claims
        claims_to_process = [c for c in all_claims if c["id"] not in processed_ids]
        print(f"Claims remaining: {len(claims_to_process)}")

        # Apply limit
        if limit is not None:
            claims_to_process = claims_to_process[:limit]
        print(f"Claims to process this run: {len(claims_to_process)}")

        if not claims_to_process:
            print("No claims to process for this provider.")
            continue

        total = len(claims_to_process)

        if num_workers == 1:
            for idx, claim in enumerate(claims_to_process):
                process_single_claim(
                    claim=claim,
                    idx=idx,
                    total=total,
                    fc=fc,
                    provider=provider,
                    dataset_dir=dataset_dir,
                    csv_path=csv_path,
                    jsonl_path=jsonl_path,
                    label_scheme=label_scheme,
                    mode=mode,
                    staged_checkers=staged_checkers,
                    property_label_classes=property_label_classes,
                )
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {}
                for idx, claim in enumerate(claims_to_process):
                    future = executor.submit(
                        process_single_claim,
                        claim=claim,
                        idx=idx,
                        total=total,
                        fc=fc,
                        provider=provider,
                        dataset_dir=dataset_dir,
                        csv_path=csv_path,
                        jsonl_path=jsonl_path,
                        label_scheme=label_scheme,
                        mode=mode,
                        staged_checkers=staged_checkers,
                        property_label_classes=property_label_classes,
                    )
                    futures[future] = claim["id"]

                for future in as_completed(futures):
                    claim_id = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"  Claim {claim_id} failed: {e}")

    # Update run configuration with post-run fields
    run_config["total_claims"] = len(all_claims)
    run_config["finished_at"] = datetime.now().isoformat()
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)

    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"Results directory: {output_dir}")

    integrity_summary = generate_summary_from_csv(csv_path, output_dir, providers, label_scheme=label_scheme)
    staged_summary = None
    if mode == "staged":
        staged_summary = generate_staged_summary_from_jsonl(
            jsonl_path,
            output_dir,
            providers,
            property_label_classes=property_label_classes,
        )
    write_property_metrics_table(
        output_dir=output_dir,
        providers=providers,
        integrity_summary=integrity_summary,
        staged_summary=staged_summary,
        property_label_classes=property_label_classes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run unified fact-checker benchmark on VeriTaS dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with all providers
  python run_benchmark.py --limit 5

  # Run with specific provider
  python run_benchmark.py --provider openai --limit 10

  # Run with multiple providers
  python run_benchmark.py --providers openai gemini --limit 10

  # Run with custom model
  python run_benchmark.py --provider openai --model gpt-5.2 --limit 5

  # Run with parallel workers
  python run_benchmark.py --providers openai gemini --limit 20 --workers 4

  # Resume from previous run
  python run_benchmark.py --resume results/run_20241220_143022
        """
    )
    parser.add_argument("--dataset", required=True,
                        help="Path to claims.json")
    parser.add_argument("--output", default=None,
                        help="Output directory (auto-generated if not specified)")
    parser.add_argument("--resume", default=None,
                        help="Resume from previous run directory")
    parser.add_argument("--provider", default=None,
                        help="Single provider to use (openai, gemini, perplexity)")
    parser.add_argument("--providers", nargs="+", default=None,
                        help="Multiple providers to use")
    parser.add_argument("--model", default=None,
                        help="Model to use (applies to single provider)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of claims to process")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers")
    parser.add_argument("--custom-search", action="store_true",
                        help="Use custom search with date filtering and content retrieval")
    parser.add_argument("--no-search", action="store_true",
                        help="Disable web search, use parametric knowledge only")
    parser.add_argument("--label-scheme", type=int, choices=[3, 7], default=3,
                        help="Label scheme: 3 for 3-class (Intact/Unknown/Compromised), "
                             "7 for 7-class with uncertainty levels (default: 3)")
    parser.add_argument("--mode", choices=["integrity", "staged"], default="integrity",
                        help="Benchmark mode: integrity (single-pass baseline) or staged (property loop)")
    parser.add_argument(
        "--seven-bin-prediction-mode",
        choices=["direct", "two_step"],
        default="direct",
        help="For 7-class labels: direct=combined label, two_step=direction then certainty",
    )
    args = parser.parse_args()

    # Determine providers
    if args.provider:
        providers = [args.provider]
    elif args.providers:
        providers = args.providers
    else:
        providers = None  # Will default to all

    # Set up models dict if model specified
    models = None
    if args.model and args.provider:
        models = {args.provider: args.model}

    run_benchmark(
        dataset_path=args.dataset,
        output_dir=args.output,
        resume_dir=args.resume,
        providers=providers,
        models=models,
        limit=args.limit,
        num_workers=args.workers,
        custom_search=args.custom_search,
        use_search=not args.no_search,
        label_scheme=get_label_scheme(args.label_scheme),
        mode=args.mode,
        seven_bin_prediction_mode=args.seven_bin_prediction_mode,
    )
