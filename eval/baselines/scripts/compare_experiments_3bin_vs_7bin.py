"""Compare 3-bin experiment predictions against 7-bin experiment predictions.

This script expects two different experiment outputs (typically benchmark run
directories), aligns successful predictions by claim_id, and builds a
3x7 confusion matrix:
  - rows: 3-bin predictions from experiment A
  - cols: 7-bin predictions from experiment B
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from eval.baselines.common.metrics import COARSEN_7_TO_3
from eval.baselines.common.types import LABELS_3, LABELS_7


def _resolve_results_csv(path: Path) -> Path:
    """Resolve an experiment directory or direct CSV path to a results CSV file."""
    if path.is_file():
        return path
    candidate = path / "results.csv"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not find results CSV for '{path}'. "
        f"Pass an experiment directory containing results.csv or a direct CSV path."
    )


def _load_predictions(csv_path: Path) -> list[dict[str, str]]:
    """Load successful predictions from benchmark CSV."""
    rows: list[dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"claim_id", "status", "verdict"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
        for row in reader:
            if row.get("status") != "success":
                continue
            claim_id = (row.get("claim_id") or "").strip()
            verdict = (row.get("verdict") or "").strip()
            if not claim_id or not verdict:
                continue
            rows.append(
                {
                    "claim_id": claim_id,
                    "verdict": verdict,
                    "provider": (row.get("provider") or "").strip(),
                }
            )
    return rows


def _to_claim_map(rows: list[dict[str, str]], source_name: str) -> dict[str, str]:
    """Convert rows to claim_id->verdict map; keep first if duplicates exist."""
    claim_map: dict[str, str] = {}
    duplicates = 0
    for row in rows:
        claim_id = row["claim_id"]
        verdict = row["verdict"]
        if claim_id in claim_map:
            duplicates += 1
            continue
        claim_map[claim_id] = verdict
    if duplicates:
        print(f"Warning: {source_name} had {duplicates} duplicate claim_id rows; kept first occurrence.")
    return claim_map


def _validate_label_set(claim_map: dict[str, str], allowed_labels: list[str], label_name: str) -> None:
    invalid = sorted({v for v in claim_map.values() if v not in allowed_labels})
    if invalid:
        preview = ", ".join(invalid[:10])
        more = "" if len(invalid) <= 10 else f" ... (+{len(invalid) - 10} more)"
        raise ValueError(
            f"Invalid {label_name} labels found: {preview}{more}. "
            f"Expected one of: {allowed_labels}"
        )


def _print_confusion(labels_3: list[str], labels_7: list[str], matrix: list[list[int]]) -> None:
    row_label_width = max(len(l) for l in labels_3) + 2
    col_width = max(max(len(l) for l in labels_7) + 2, 12)

    print("\nConfusion Matrix (rows=3-bin predictions, cols=7-bin predictions):")
    header = " " * (row_label_width + 4) + "".join(f"{l:>{col_width}s}" for l in labels_7)
    print(header)
    for i, row_label in enumerate(labels_3):
        row = matrix[i]
        row_str = "".join(f"{v:>{col_width}d}" for v in row)
        print(f"  {row_label:<{row_label_width}s}{row_str}")


def _plot_confusion_matrix(
    matrix: list[list[int]],
    labels_3: list[str],
    labels_7: list[str],
    output_path: Path,
    title: str,
) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not installed; skipping PNG plot.")
        return False

    arr = np.array(matrix, dtype=int)
    row_totals = arr.sum(axis=1, keepdims=True)
    row_totals_safe = np.where(row_totals == 0, 1, row_totals)
    arr_pct = arr / row_totals_safe

    short = {
        "Intact (certain)": "Intact\n(certain)",
        "Intact (rather certain)": "Intact\n(r. certain)",
        "Intact (rather uncertain)": "Intact\n(r. uncertain)",
        "Unknown": "Unknown",
        "Compromised (rather uncertain)": "Comp.\n(r. uncertain)",
        "Compromised (rather certain)": "Comp.\n(r. certain)",
        "Compromised (certain)": "Comp.\n(certain)",
    }
    col_labels = [short.get(l, l) for l in labels_7]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    im = ax.imshow(arr_pct, aspect="auto", interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=arr_pct.max())
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Row proportion", fontsize=10)

    ax.set_xticks(range(len(labels_7)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(labels_3)))
    ax.set_yticklabels(labels_3, fontsize=11, fontweight="bold")
    ax.set_xlabel("7-bin prediction", fontsize=11, labelpad=8)
    ax.set_ylabel("3-bin prediction", fontsize=11, labelpad=8)
    ax.set_title(title, fontsize=12, pad=10)

    thresh = arr_pct.max() / 2.0 if arr_pct.size else 0
    for i in range(len(labels_3)):
        for j in range(len(labels_7)):
            color = "white" if arr_pct[i, j] > thresh else "black"
            ax.text(j, i - 0.12, str(int(arr[i, j])), ha="center", va="center", fontsize=11, fontweight="bold", color=color)
            ax.text(j, i + 0.22, f"{arr_pct[i, j]:.0%}", ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks([x - 0.5 for x in range(1, len(labels_7) + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(labels_3) + 1)], minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def compare_experiments(exp3_path: Path, exp7_path: Path, output_dir: Path) -> dict[str, Any]:
    """Compute confusion matrix between 3-bin and 7-bin predictions."""
    csv_3 = _resolve_results_csv(exp3_path)
    csv_7 = _resolve_results_csv(exp7_path)

    rows_3 = _load_predictions(csv_3)
    rows_7 = _load_predictions(csv_7)
    map_3 = _to_claim_map(rows_3, "3-bin experiment")
    map_7 = _to_claim_map(rows_7, "7-bin experiment")

    _validate_label_set(map_3, LABELS_3, "3-bin")
    _validate_label_set(map_7, LABELS_7, "7-bin")

    shared_claim_ids = sorted(set(map_3).intersection(map_7), key=lambda x: int(x) if x.isdigit() else x)
    if not shared_claim_ids:
        raise ValueError("No overlapping claim_id values found across the two experiments.")

    idx3 = {label: i for i, label in enumerate(LABELS_3)}
    idx7 = {label: i for i, label in enumerate(LABELS_7)}
    matrix = [[0 for _ in LABELS_7] for _ in LABELS_3]

    agreements_after_coarsen = 0
    for claim_id in shared_claim_ids:
        pred_3 = map_3[claim_id]
        pred_7 = map_7[claim_id]
        matrix[idx3[pred_3]][idx7[pred_7]] += 1
        if pred_3 == COARSEN_7_TO_3.get(pred_7, "Unknown"):
            agreements_after_coarsen += 1

    output_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "experiment_3bin": str(exp3_path.resolve()),
        "experiment_7bin": str(exp7_path.resolve()),
        "csv_3bin": str(csv_3.resolve()),
        "csv_7bin": str(csv_7.resolve()),
        "n_success_3bin": len(map_3),
        "n_success_7bin": len(map_7),
        "n_overlapping_claims": len(shared_claim_ids),
        "dropped_non_overlapping_3bin": len(map_3) - len(shared_claim_ids),
        "dropped_non_overlapping_7bin": len(map_7) - len(shared_claim_ids),
        "coarsened_agreement_rate": round(agreements_after_coarsen / len(shared_claim_ids), 4),
        "confusion_matrix": {
            "row_labels": LABELS_3,
            "col_labels": LABELS_7,
            "matrix": matrix,
        },
    }

    json_path = output_dir / "comparison_3bin_pred_vs_7bin_pred.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    csv_path = output_dir / "comparison_3bin_pred_vs_7bin_pred.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["3bin_pred \\ 7bin_pred", *LABELS_7])
        for row_label, row in zip(LABELS_3, matrix):
            writer.writerow([row_label, *row])

    plot_path = output_dir / "comparison_3bin_pred_vs_7bin_pred.png"
    plotted = _plot_confusion_matrix(
        matrix=matrix,
        labels_3=LABELS_3,
        labels_7=LABELS_7,
        output_path=plot_path,
        title="Confusion Matrix (3-bin Pred × 7-bin Pred)",
    )

    _print_confusion(LABELS_3, LABELS_7, matrix)
    print(f"\nOverlapping claims: {len(shared_claim_ids)}")
    print(f"Coarsened agreement rate (3-bin vs coarsened 7-bin): {out['coarsened_agreement_rate']:.2%}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    if plotted:
        print(f"Saved plot: {plot_path}")

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare predictions from a 3-bin experiment vs a 7-bin experiment."
    )
    parser.add_argument(
        "--exp3",
        required=True,
        help="Path to 3-bin experiment directory (containing results.csv) or direct results.csv path.",
    )
    parser.add_argument(
        "--exp7",
        required=True,
        help="Path to 7-bin experiment directory (containing results.csv) or direct results.csv path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: <exp7>/comparison_3bin_vs_7bin_pred).",
    )
    args = parser.parse_args()

    exp3_path = Path(args.exp3).expanduser().resolve()
    exp7_path = Path(args.exp7).expanduser().resolve()

    if exp3_path == exp7_path:
        raise ValueError("--exp3 and --exp7 must point to two different experiments/paths.")

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser().resolve()
    else:
        out_dir = (exp7_path if exp7_path.is_dir() else exp7_path.parent) / "comparison_3bin_vs_7bin_pred"

    compare_experiments(exp3_path=exp3_path, exp7_path=exp7_path, output_dir=out_dir)


if __name__ == "__main__":
    main()
