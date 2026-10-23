#!/usr/bin/env python3
"""
Benchmark runner for the VeriTaS dataset.

Runs the fact-checking baseline with any combination of providers:
- OpenAI (gpt-5.2, ...)
- Gemini (gemini-2.5-flash, ...)
- Anthropic (claude-sonnet-4-6, ...)
- Self-hosted models via vLLM (Llama 4, ...)
- OpenRouter (z-ai/glm-5.3-flash, ...)

Every provider runs the same baseline: the model verifies the claim with the
web_search tool (results restricted to before the claim date) and the fetch_url
tool (via the scrapeMM server), and answers with the claim's integrity, its certainty
and a justification in JSON, which make a label of the 7-class scheme.

Features:
- Run single provider or multiple providers
- Parallel processing with configurable workers
- Resume from previous runs
- CSV and JSONL output
- Confusion matrix visualization (7-bin and coarsened 3-bin)
- Metrics summary
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from threading import Lock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baselines import UnifiedFactChecker, FactCheckError, FactCheckResult
from baselines.factchecker import DEFAULT_MODELS, DEFAULT_PROVIDERS
from baselines.providers import DEFAULT_REASONING_EFFORT
from baselines.common import (
    classify_integrity,
    compute_metrics,
    compute_regression_metrics,
    print_metrics,
)
from baselines.common.metrics import compute_coarsened_metrics, COARSEN_7_TO_3
from baselines.common.media import MediaLimits
from baselines.common.reviews import lookup_reviews, signatory_scopes
from baselines.common.tools import (MAX_FETCH_CHARS, MAX_SEARCH_PAGES, DEFAULT_MAX_SEARCHES, DEFAULT_MAX_FETCHES,
                                    temporary_media_store)
from baselines.common.types import LABEL_SCHEME_7
from baselines.scripts.fact_check_exposure import write_fact_check_exposure
from baselines.scripts.run_statistics import write_run_statistics

# Thread-safe CSV writing
csv_lock = Lock()

# The URL lists can exceed the csv module's default field size limit (128 KiB) when read back
csv.field_size_limit(sys.maxsize)

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
    "correct",
    "status",
    "error_message",
    "justification",
    "encountered_urls",
    "retrieved_urls",
    "linked_urls",
    "quoted_urls",
]

# Per-claim URL lists, stored in the CSV as JSON arrays
URL_LIST_FIELDS = ("encountered_urls", "retrieved_urls", "linked_urls", "quoted_urls")

JUSTIFICATION_CSV_MAX_CHARS = 1000


def format_justification_for_csv(result: FactCheckResult | None) -> str:
    """Flatten a justification onto one line for the CSV; the JSONL keeps it raw."""
    if not result or not result.justification:
        return ""
    flat = re.sub(r"\s+", " ", result.justification).strip()
    if len(flat) > JUSTIFICATION_CSV_MAX_CHARS:
        return flat[:JUSTIFICATION_CSV_MAX_CHARS - 3] + "..."
    return flat

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


def prepare_resume(csv_path: Path, jsonl_path: Path, providers: list[str]) -> int:
    """Prepare a run's results for resuming it: remove the rows of `providers` that are not
    successes (errors, or lines a crash cut short) from the CSV and JSONL, so that their
    claims are retried, and bring the CSV's header up to date with CSV_COLUMNS, so that the
    rows appended next line up with it.

    Returns the number of CSV rows removed.
    """
    def failed(record: dict) -> bool:
        return record.get("provider") in providers and record.get("status") != "success"

    n_dropped = 0
    if csv_path.exists():
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = CSV_COLUMNS + [c for c in reader.fieldnames or [] if c not in CSV_COLUMNS]
            rows = list(reader)
        kept = [row for row in rows if not failed(row)]
        n_dropped = len(rows) - len(kept)
        tmp_path = csv_path.with_name(csv_path.name + ".tmp")
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)
        os.replace(tmp_path, csv_path)

    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        kept_lines = []
        for line in lines:
            try:
                if not failed(json.loads(line)):
                    kept_lines.append(line if line.endswith("\n") else line + "\n")
            except json.JSONDecodeError:
                pass  # A line a crash cut short
        tmp_path = jsonl_path.with_name(jsonl_path.name + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.writelines(kept_lines)
        os.replace(tmp_path, jsonl_path)

    return n_dropped


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
) -> dict:
    """Build a CSV record from claim processing results."""
    gt_integrity = ground_truth.get("integrity")
    gt_class = classify_integrity(gt_integrity) if gt_integrity is not None else None

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
        "correct": correct,
        "status": status,
        "error_message": error_message,
        "justification": format_justification_for_csv(result),
        **{
            field: json.dumps(getattr(result, field), ensure_ascii=False) if result else None
            for field in URL_LIST_FIELDS
        },
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


# =============================================================================
# Summary generation
# =============================================================================

def generate_summary_from_csv(csv_path: Path, output_dir: Path, providers: list[str], run_statistics: dict | None = None):
    """Generate summary JSON and confusion matrix plots from CSV results. Each provider's summary
    also gets its run statistics (see run_statistics.py), without the list of what was blocked by
    claim, which only run_statistics.json holds. The summary ends with every URL the tools blocked,
    by provider, sorted, each with the rule that first blocked it (`blocked_urls`)."""
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
        # The model gave no valid answer, e.g. because its response was cut off
        no_answer = sum(1 for r in provider_results if r["status"] == "no_answer")
        errors = total - successful - no_answer

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

        metrics = compute_metrics(y_true, y_pred, labels=LABEL_SCHEME_7.labels)

        # Compute coarsened 3-bin metrics
        coarsened = compute_coarsened_metrics(y_true, y_pred)
        if coarsened:
            metrics["coarsened_3class"] = coarsened

        # Compute regression metrics (MSE, MAE): the predicted label's value against the
        # continuous ground-truth integrity. Only for the 7 bins the model predicts on.
        regression_metrics = compute_regression_metrics(gt_integrity_values, verdict_values, scheme=LABEL_SCHEME_7)
        if regression_metrics:
            metrics["mse"] = regression_metrics["mse"]
            metrics["mae"] = regression_metrics["mae"]

        provider_summary = {
            "total_claims": total,
            "successful_runs": successful,
            "no_answer": no_answer,
            "errors": errors,
            "comparisons_available": len(y_true),
            "metrics": metrics,
        }
        if coarsened:
            provider_summary["metrics_3class"] = coarsened
        if statistics := (run_statistics or {}).get(provider):
            blocked = {key: value for key, value in statistics["blocked"].items() if key != "by_claim"}
            provider_summary["run_statistics"] = {**statistics, "blocked": blocked}

        summary["providers"][provider] = provider_summary

        print(f"\n=== {provider.upper()} ===")
        print(f"  Total: {total}, Successful: {successful}, No answer: {no_answer}, Errors: {errors}")

        if metrics:
            print_metrics(metrics)
            plot_confusion_matrix(metrics, output_dir / f"confusion_matrix_{provider}.png", provider)

            # Plot coarsened 3-bin confusion matrix
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

    summary["blocked_urls"] = {}
    for provider in summary["providers"]:
        by_claim = (run_statistics or {}).get(provider, {}).get("blocked", {}).get("by_claim", {})
        rules = {}
        for entries in by_claim.values():
            for entry in entries:
                rules.setdefault(entry["url"], entry["rule"])
        summary["blocked_urls"][provider] = dict(sorted(rules.items()))

    # Save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nSummary saved to: {summary_path}")
    return summary


LEADERBOARD_REASONING_MIN_CHARS = 10
LEADERBOARD_REASONING_MAX_CHARS = 500


def _utf16_length(text: str) -> int:
    """The length of `text` as JavaScript counts it, in UTF-16 code units, as the leaderboard
    does: an emoji or other character outside the Basic Multilingual Plane counts twice."""
    return len(text.encode("utf-16-le")) // 2


def leaderboard_reasoning(result: dict) -> str:
    """The reasoning of a verdict for the leaderboard, which takes LEADERBOARD_REASONING_MIN_CHARS
    to LEADERBOARD_REASONING_MAX_CHARS characters, counted as JavaScript counts them (see
    _utf16_length): the justification, or, if it is shorter than the minimum (or missing from the
    answer), the whole final response, which holds the JSON answer and so is longer. Cut off at
    the maximum, ending in "..."."""
    reasoning = result["justification"]
    if _utf16_length(reasoning) < LEADERBOARD_REASONING_MIN_CHARS:
        reasoning = result["reasoning"]
    if _utf16_length(reasoning) > LEADERBOARD_REASONING_MAX_CHARS:
        # Cut between code units; a character cut in half (a lone surrogate) is dropped
        units = reasoning.encode("utf-16-le")[:2 * (LEADERBOARD_REASONING_MAX_CHARS - 3)]
        reasoning = units.decode("utf-16-le", errors="ignore") + "..."
    return reasoning


def write_leaderboard_results(jsonl_path: Path, output_dir: Path, providers: list[str]):
    """Write each provider's verdicts in the leaderboard's format: a JSON list of
    {"claim_id", "verdict", "reasoning"}, sorted by claim ID. The verdict is the label's value
    (1 = Intact (certain) to -1 = Compromised (certain)), rounded to 2 decimals, the reasoning
    is the justification, within the leaderboard's limits (see leaderboard_reasoning). Claims
    that failed with an error have no verdict and are left out.

    With one provider, the file is results_for_leaderboard.json, otherwise
    results_for_leaderboard_<provider>.json.
    """
    if not jsonl_path.exists():
        return

    entries = {provider: [] for provider in providers}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record["provider"] not in entries or record["status"] != "success":
                continue
            result = record["result"]
            value = round(LABEL_SCHEME_7.verdict_to_numeric[result["verdict"]], 2)
            entries[record["provider"]].append({
                "claim_id": record["claim_id"],
                "verdict": int(value) if value.is_integer() else value,
                "reasoning": leaderboard_reasoning(result),
            })

    for provider, provider_entries in entries.items():
        if not provider_entries:
            continue
        provider_entries.sort(key=lambda e: e["claim_id"])
        name = "results_for_leaderboard.json" if len(providers) == 1 else f"results_for_leaderboard_{provider}.json"
        path = output_dir / name
        with open(path, "w", encoding="utf-8") as f:
            f.write("[\n" + ",\n".join("  " + json.dumps(e, ensure_ascii=False) for e in provider_entries) + "\n]\n")
        print(f"Leaderboard results saved to: {path}")


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


def write_property_metrics_table(
    output_dir: Path,
    providers: list[str],
    integrity_summary: dict | None,
):
    """Write integrity metrics as tables (rows=metrics, cols=binning)."""
    metric_order = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "support",
        "mse",
        "mae",
    ]
    metric_names = {
        "accuracy": "Accuracy",
        "macro_f1": "Macro F1",
        "weighted_f1": "Weighted F1",
        "support": "Support",
        "mse": "MSE",
        "mae": "MAE",
    }
    bin_columns = ["3-bin", "7-bin"]
    properties = ["integrity"]
    provider_tables: dict[str, dict[str, dict[str, dict]]] = {}

    def add_metrics_block(
        provider_data: dict,
        property_name: str,
        binning: str,
        metrics: dict,
    ):
        if not metrics:
            return
        if property_name not in provider_data:
            provider_data[property_name] = {}
        provider_data[property_name][binning] = {
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "weighted_f1": metrics.get("weighted_f1"),
            "support": _metric_support(metrics),
            # Against the continuous ground-truth integrity (see generate_summary_from_csv)
            "mse": metrics.get("mse"),
            "mae": metrics.get("mae"),
        }

    for provider in providers:
        provider_data: dict[str, dict[str, dict]] = {}
        provider_tables[provider] = provider_data

        integrity_provider = ((integrity_summary or {}).get("providers") or {}).get(provider, {})
        integrity_metrics = integrity_provider.get("metrics", {})
        add_metrics_block(provider_data, "integrity", "7-bin", integrity_metrics)
        integrity_3 = integrity_provider.get("metrics_3class") or integrity_metrics.get("coarsened_3class")
        add_metrics_block(provider_data, "integrity", "3-bin", integrity_3)

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


def add_review_info(claims: list[dict], lookup=None) -> list[dict]:
    """The claims with what their fact-check needs to know of their reviews (fact-checks), which
    the tools block:
        review_urls        the URLs of the reviews ("review_url" if the dataset gives no list)
        review_date        the day the first of them was published, from the dataset or else from
                           the database: web_search only returns content published before it.
                           None if neither has it, and web_search falls back to the claim date
        fact_check_scopes  the hosts, or hosts and paths, under which the reviews' publishers
                           publish fact-checks, from the database (blocked along with the
                           signatories', see signatory_scopes)
    `lookup` looks the reviews up by their URLs, by default in the database (see lookup_reviews)."""
    urls = {claim["id"]: claim.get("review_urls") or ([claim["review_url"]] if claim.get("review_url") else [])
            for claim in claims}
    reviews = (lookup or lookup_reviews)([url for claim_urls in urls.values() for url in claim_urls])

    enriched = []
    for claim in claims:
        found = [reviews[url] for url in urls[claim["id"]] if url in reviews]
        dates = [review.published for review in found if review.published]
        enriched.append({
            **claim,
            "review_urls": urls[claim["id"]],
            "review_date": claim.get("review_date") or (min(dates).isoformat() if dates else None),
            "fact_check_scopes": sorted({scope for review in found for scope in review.fact_check_scopes}),
        })
    with_urls = [claim for claim in enriched if claim["review_urls"]]
    print(f"Reviews: {sum(any(url in reviews for url in claim['review_urls']) for claim in with_urls)} of "
          f"{len(with_urls)} claims with review URLs found in the database, "
          f"{sum(bool(claim['fact_check_scopes']) for claim in enriched)} with fact-check scopes, "
          f"{sum(not claim['review_date'] for claim in enriched)} without review date (searched before the claim date)")
    return enriched


def process_single_claim(
    claim: dict,
    idx: int,
    total: int,
    fc: UnifiedFactChecker,
    provider: str,
    dataset_dir: Path,
    csv_path: Path,
    jsonl_path: Path,
    blocked_scopes: list[str],
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
    # Claims run in parallel, so every line names its claim
    tag = f"[{claim_id}]"
    print(f"  {tag} Text: {claim_text[:80]}..." if len(claim_text) > 80 else f"  {tag} Text: {claim_text}")

    # New format: media is directly in claim["media"] array with file_path
    media_items = claim.get("media", [])
    media_refs = media_items  # Keep reference for CSV record
    image_paths = []
    video_paths = []
    started = time.monotonic()

    try:
        # Resolve media files if present
        if media_items:
            print(f"  {tag} Found {len(media_items)} media item(s)")
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
                        print(f"    {tag} Warning: Media file not found: {full_path}")

            if provider == "gemini" and video_paths:
                print(f"  {tag} Using native video processing for Gemini")

        result = fc.check_claim(
            claim_text,
            image_paths=image_paths if image_paths else None,
            video_paths=video_paths if video_paths else None,
            claim_date=claim_date,
            provider=provider,
            claim_id=claim_id,
            # The claim's own fact-checks, the other fact-checks of their publishers and those of
            # all IFCN signatories and EFCSN members would give the answer away; web_search only
            # shows what was published before the first review
            blocked_urls=claim.get("review_urls"),
            cutoff_date=claim.get("review_date"),
            blocked_scopes=[*blocked_scopes, *claim.get("fact_check_scopes", [])],
        )

        status = "success"
        error_message = None
        print(f"  {tag} -> Verdict: {result.verdict}")

    except FactCheckError as e:
        # Keep what the fact-check got to: its trace, URLs and usage
        print(f"  {tag} -> {e.status.upper()}: {str(e)}")
        result = e.result
        status = e.status
        error_message = str(e)
    except Exception as e:
        print(f"  {tag} -> ERROR: {str(e)}")
        result = None
        status = "error"
        error_message = str(e)
    duration = time.monotonic() - started

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
    )

    # Build JSONL record
    jsonl_record = {
        "claim_id": claim_id,
        "claim_text": claim_text,
        "claim_text_clean": claim_text,  # In new format, text is already clean
        "claim_date": claim_date,
        "review_date": claim.get("review_date"),
        "fact_check_scopes": claim.get("fact_check_scopes"),
        "num_images": len(image_paths),
        "num_videos": len(video_paths),
        "ground_truth": ground_truth,
        "provider": provider,
        "result": result.to_dict() if result else None,
        "status": status,
        "error_message": error_message,
        # Wall-clock time of the fact-check, retries included; claims run in parallel and share
        # the scrapeMM server and the APIs' rate limits
        "duration_seconds": round(duration, 1),
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
    scrape_methods: list[str] | str | None = "auto",
    max_searches: int = DEFAULT_MAX_SEARCHES,
    max_fetches: int = DEFAULT_MAX_FETCHES,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    media_limits: MediaLimits = MediaLimits(),
    openrouter_host: str | None = None,
):
    """
    Run the fact-checking baseline on VeriTaS claims.

    Args:
        dataset_path: Path to claims.json
        output_dir: Directory to save results (auto-generated if None). Must not contain results yet.
        resume_dir: Directory of previous run to resume from. Its failed claims are retried.
        providers: List of providers to use (default: openai, gemini, anthropic)
        models: Dict mapping provider names to model identifiers
        limit: Maximum number of claims to process (None = all)
        num_workers: Number of parallel workers
        scrape_methods: Which scrapeMM backends fetch_url uses, in order (subset of
                    integrations/browser/firecrawl/decodo, or "auto"). Default "auto".
        max_searches: Maximum number of web_search calls per claim.
        max_fetches: Maximum number of fetch_url calls per claim.
        reasoning_effort: How much the models reason, passed to each API's effort setting.
        media_limits: Which images and videos the models get, and in what form.
        openrouter_host: The endpoint OpenRouter sends every request to, e.g. "deepinfra/fp8".
                         Required for openrouter.
    """
    # Default providers
    if providers is None:
        providers = list(DEFAULT_PROVIDERS)

    # Setup paths
    dataset_path = Path(dataset_path)
    dataset_dir = dataset_path.parent

    # Determine output directory
    if resume_dir:
        output_dir = Path(resume_dir)
        print(f"Resuming from: {output_dir}")
    elif output_dir:
        output_dir = Path(output_dir)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        # Build model name for directory
        if len(providers) == 1:
            model_name = (models or {}).get(providers[0], DEFAULT_MODELS.get(providers[0], providers[0]))
        else:
            model_name = "+".join(providers)
        output_dir = Path(__file__).parent.parent / "results" / f"{model_name}_{timestamp}"

    # File paths
    csv_path = output_dir / "results.csv"
    jsonl_path = output_dir / "results_detailed.jsonl"
    config_path = output_dir / "run_config.json"

    if not resume_dir and csv_path.exists():
        raise ValueError(f"{output_dir} already contains results. Pass it as --resume to continue that run.")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Retry the claims that failed before
    if resume_dir:
        n_failed = prepare_resume(csv_path, jsonl_path, providers)
        if n_failed:
            print(f"Retrying {n_failed} failed claim(s)")

    blocked_scopes = signatory_scopes()
    print(f"Fact-check scopes of IFCN signatories and EFCSN members blocked: {len(blocked_scopes)}")

    # Save run configuration immediately so it exists even if the run crashes
    started_at = datetime.now().isoformat()
    run_config = {
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "providers": providers,
        "models": models,
        "limit": limit,
        "num_workers": num_workers,
        "scrape_methods": scrape_methods,
        "max_searches": max_searches,
        "max_fetches": max_fetches,
        "max_fetch_chars": MAX_FETCH_CHARS,
        "max_search_pages": MAX_SEARCH_PAGES,
        "media_limits": asdict(media_limits),
        "reasoning_effort": reasoning_effort,
        "openrouter_host": openrouter_host,
        "search_cutoff": "review_date",
        "signatory_scopes": blocked_scopes,
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
    all_claims = add_review_info(all_claims)

    # Initialize fact-checker with all requested providers
    print(f"\nInitializing providers: {', '.join(providers)}")
    print(f"scrapeMM methods for fetch_url: {scrape_methods}")
    print(f"Tool limits per claim: {max_searches} web_search, {max_fetches} fetch_url "
          f"(max {MAX_FETCH_CHARS:,} chars each)")
    print(f"Media limits: {media_limits}")
    print(f"Reasoning effort: {reasoning_effort}")
    if openrouter_host:
        print(f"OpenRouter host: {openrouter_host}")
    fc = UnifiedFactChecker(
        providers=providers,
        models=models,
        scrape_methods=scrape_methods,
        max_searches=max_searches,
        max_fetches=max_fetches,
        reasoning_effort=reasoning_effort,
        media_limits=media_limits,
        openrouter_host=openrouter_host,
    )

    # The fetched pages' media go to a temporary store on local disk, deleted claim by claim
    with temporary_media_store() as media_store:
        print(f"Media of fetched pages: {media_store} (temporary, deleted after each claim)")

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
                        blocked_scopes=blocked_scopes,
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
                            blocked_scopes=blocked_scopes,
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

    run_statistics = write_run_statistics(jsonl_path, output_dir)
    integrity_summary = generate_summary_from_csv(csv_path, output_dir, providers, run_statistics)
    write_property_metrics_table(
        output_dir=output_dir,
        providers=providers,
        integrity_summary=integrity_summary,
    )
    write_leaderboard_results(jsonl_path, output_dir, providers)
    write_fact_check_exposure(jsonl_path, all_claims, output_dir)


def main():
    """Run the benchmark with the command line's options."""
    parser = argparse.ArgumentParser(
        description="Run the fact-checking baseline on the VeriTaS dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with the default providers (openai, gemini, anthropic)
  python run_benchmark.py --dataset claims.json --limit 5

  # Run with specific provider
  python run_benchmark.py --dataset claims.json --provider gemini --limit 10

  # Run with multiple providers
  python run_benchmark.py --dataset claims.json --providers openai gemini --limit 10

  # Run with custom model
  python run_benchmark.py --dataset claims.json --provider openai --model gpt-5.2 --limit 5

  # Run with parallel workers
  python run_benchmark.py --dataset claims.json --providers openai gemini --limit 20 --workers 4

  # Resume from previous run
  python run_benchmark.py --dataset claims.json --provider gemini --resume results/gemini-2.5-flash_2026-01-01_12-00
        """
    )
    parser.add_argument("--dataset", required=True,
                        help="Path to claims.json")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output", default=None,
                              help="Output directory (auto-generated if not specified)")
    output_group.add_argument("--resume", default=None,
                              help="Resume from previous run directory, retrying its failed claims")
    parser.add_argument("--provider", default=None,
                        help="Single provider to use (openai, gemini, anthropic, selfhosted, openrouter)")
    parser.add_argument("--providers", nargs="+", default=None,
                        help="Multiple providers to use")
    parser.add_argument("--model", default=None,
                        help="Model to use (applies to single provider)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of claims to process")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers")
    parser.add_argument(
        "--scrapemm-methods",
        nargs="+",
        choices=["auto", "integrations", "browser", "firecrawl", "decodo"],
        default=["auto"],
        help="Which scrapeMM backends fetch_url uses, in order. Subset of "
             "integrations/browser/firecrawl/decodo, or 'auto' (used alone) to let "
             "the scrapeMM server pick per domain. Default: auto.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["minimal", "low", "medium", "high", "xhigh", "max"],
        default=DEFAULT_REASONING_EFFORT,
        help="How much the model reasons, passed to the API's effort setting (OpenAI: reasoning.effort, "
             "Anthropic: output_config.effort, Gemini: thinking_level, vLLM: reasoning_effort, "
             "OpenRouter: reasoning.effort). Which values a model takes depends on its vendor. "
             f"Default: {DEFAULT_REASONING_EFFORT}.",
    )
    parser.add_argument(
        "--openrouter-host",
        default=None,
        help="The endpoint OpenRouter sends every request to, as its tag in "
             "https://openrouter.ai/api/v1/models/<model>/endpoints, e.g. deepinfra/fp8. Required for openrouter.",
    )
    media_group = parser.add_argument_group("media limits", "Which images and videos the models get, and in what form")
    for limit in fields(MediaLimits):
        media_group.add_argument(f"--{limit.name.replace('_', '-')}", type=type(limit.default), default=limit.default,
                                 help=f"{limit.metadata['help']}. Default: {limit.default}.")
    args = parser.parse_args()

    # Determine providers
    if args.provider:
        providers = [args.provider]
    elif args.providers:
        providers = args.providers
    else:
        providers = None  # Will default to DEFAULT_PROVIDERS

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
        scrape_methods=args.scrapemm_methods,
        reasoning_effort=args.reasoning_effort,
        media_limits=MediaLimits(**{limit.name: getattr(args, limit.name) for limit in fields(MediaLimits)}),
        openrouter_host=args.openrouter_host,
    )


if __name__ == "__main__":
    main()
