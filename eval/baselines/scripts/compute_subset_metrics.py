#!/usr/bin/env python3
"""
Compute MAE and MSE per subset for one or multiple runs.

Each run is a model subdirectory containing results.csv. Output is one row per
run with metrics for each claim subset:
  - overall
  - images     (has at least one image, no video)
  - videos     (has at least one video, no image)
  - text_only  (no media)
  - english    (language == "en")
  - non_english
  - intact     (gt_class in {Intact (certain), Intact (rather certain)})
  - compromised (gt_class in {Compromised (certain), Compromised (rather certain)})

Ground truth: gt_integrity float from results.csv (continuous score).
Prediction:   verdict mapped to -1 (Compromised), 0 (Unknown), 1 (Intact).

Usage:
    # Single run directory (contains model subdirs)
    python baselines/scripts/compute_subset_metrics.py \\
        baselines/results/camera_ready/q1-2026/custom-search

    # Multiple run directories
    python baselines/scripts/compute_subset_metrics.py \\
        baselines/results/camera_ready/q1-2026/custom-search \\
        baselines/results/camera_ready/q1-2026/no-search

    # Save CSV output
    python baselines/scripts/compute_subset_metrics.py \\
        baselines/results/camera_ready/q1-2026/custom-search \\
        -o baselines/results/camera_ready/q1-2026/subset_metrics.csv
"""

import argparse
import csv
import json
from pathlib import Path

DEFAULT_CLAIMS_PATH = Path("/mnt/vast/workspaces/PI_Rohrbach/mk79honu/data/VeriTaS/veritas_release/veritas_2026_q1/claims.json")

VERDICT_TO_VALUE = {
    "Intact (certain)": 1.0,
    "Intact (rather certain)": 2 / 3,
    "Intact (rather uncertain)": 1 / 3,
    "Unknown": 0.0,
    "Compromised (rather uncertain)": -1 / 3,
    "Compromised (rather certain)": -2 / 3,
    "Compromised (certain)": -1.0,
}

INTACT_CERTAIN = {"Intact (certain)", "Intact (rather certain)"}
COMPROMISED_CERTAIN = {"Compromised (certain)", "Compromised (rather certain)"}

SUBSETS = [
    "overall",
    "images",
    "videos",
    "text_only",
    "english",
    "non_english",
    "intact",
    "compromised",
]


def verdict_to_value(verdict: str) -> float | None:
    return VERDICT_TO_VALUE.get(verdict)


def load_claim_metadata(claims_path: Path) -> dict[int, dict]:
    """Return claim_id -> {language, modality} for every claim."""
    with open(claims_path, encoding="utf-8") as f:
        data = json.load(f)

    meta = {}
    for claim in data["claims"]:
        media = claim.get("media", [])
        has_image = any(m.get("type") == "image" for m in media)
        has_video = any(m.get("type") == "video" for m in media)

        if has_image and has_video:
            modality = "mixed"
        elif has_image:
            modality = "images"
        elif has_video:
            modality = "videos"
        else:
            modality = "text_only"

        meta[claim["id"]] = {
            "language": claim.get("language", ""),
            "modality": modality,
        }

    return meta


def compute_mae_mse(errors: list[float]) -> tuple[float, float] | tuple[None, None]:
    if not errors:
        return None, None
    n = len(errors)
    mae = sum(abs(e) for e in errors) / n
    mse = sum(e ** 2 for e in errors) / n
    return mae, mse


def process_run(results_path: Path, claim_meta: dict[int, dict]) -> dict[str, list[float]]:
    """Read results.csv and return errors grouped by subset."""
    subset_errors: dict[str, list[float]] = {s: [] for s in SUBSETS}

    with open(results_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] != "success":
                continue

            claim_id = int(row["claim_id"])
            gt_class = row.get("gt_class", "")

            try:
                gt_value = float(row["gt_integrity"])
            except (ValueError, KeyError):
                continue

            pred_value = verdict_to_value(row.get("verdict", ""))
            if pred_value is None:
                continue

            error = pred_value - gt_value
            meta = claim_meta.get(claim_id, {})
            modality = meta.get("modality", "")
            language = meta.get("language", "")

            subset_errors["overall"].append(error)

            if modality == "images":
                subset_errors["images"].append(error)
            elif modality == "videos":
                subset_errors["videos"].append(error)
            elif modality == "text_only":
                subset_errors["text_only"].append(error)

            if language == "en":
                subset_errors["english"].append(error)
            elif language:
                subset_errors["non_english"].append(error)

            if gt_class in INTACT_CERTAIN:
                subset_errors["intact"].append(error)
            elif gt_class in COMPROMISED_CERTAIN:
                subset_errors["compromised"].append(error)

    return subset_errors


def collect_runs(paths: list[Path]) -> list[tuple[str, Path]]:
    """
    Each path may be either:
      - a model run dir directly (contains results.csv), or
      - a parent dir whose subdirs are model run dirs.
    Returns list of (run_label, results_csv_path).
    """
    runs = []
    for path in paths:
        direct_csv = path / "results.csv"
        if direct_csv.exists():
            runs.append((path.name, direct_csv))
        else:
            for subdir in sorted(path.iterdir()):
                if subdir.is_dir() and (subdir / "results.csv").exists():
                    # include parent dir name when multiple top-level paths given
                    label = f"{path.name}/{subdir.name}" if len(paths) > 1 else subdir.name
                    runs.append((label, subdir / "results.csv"))
    return runs


def format_table(rows: list[dict]) -> str:
    if not rows:
        return ""

    # Build column list: run, then for each subset: n, mae, mse
    col_headers = ["run"]
    for s in SUBSETS:
        col_headers += [f"{s}_n", f"{s}_mae", f"{s}_mse"]

    # Determine column widths
    col_widths = {c: len(c) for c in col_headers}
    for row in rows:
        for col in col_headers:
            col_widths[col] = max(col_widths[col], len(str(row.get(col, ""))))

    def fmt_row(row):
        parts = []
        for col in col_headers:
            val = str(row.get(col, ""))
            parts.append(val.rjust(col_widths[col]))
        return "  ".join(parts)

    sep = "  ".join("-" * col_widths[c] for c in col_headers)
    header = "  ".join(c.rjust(col_widths[c]) for c in col_headers)

    lines = [header, sep]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compute MAE/MSE per subset for one or multiple runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("paths", nargs="+", type=Path,
                        help="One or more run/parent directories")
    parser.add_argument("--claims", type=Path, default=None,
                        help="Path to claims.json (default: auto-detect)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Write CSV to this path (default: <parent_dir>/subset_metrics.csv)")
    args = parser.parse_args()

    claims_path = args.claims or DEFAULT_CLAIMS_PATH
    if not claims_path.exists():
        raise FileNotFoundError(f"Claims file not found: {claims_path}")

    runs = collect_runs(args.paths)
    if not runs:
        raise ValueError("No results.csv files found in the given paths.")

    print(f"Claims file : {claims_path}")
    print(f"Runs found  : {len(runs)}")
    print()

    claim_meta = load_claim_metadata(claims_path)

    table_rows = []
    for run_label, csv_path in runs:
        subset_errors = process_run(csv_path, claim_meta)
        row = {"run": run_label}
        for s in SUBSETS:
            errors = subset_errors[s]
            mae, mse = compute_mae_mse(errors)
            row[f"{s}_n"] = len(errors)
            row[f"{s}_mae"] = f"{mae:.4f}" if mae is not None else "N/A"
            row[f"{s}_mse"] = f"{mse:.4f}" if mse is not None else "N/A"
        table_rows.append(row)

    print(format_table(table_rows))

    mse_subsets = ["images", "videos", "text_only", "english", "non_english", "intact", "compromised"]
    print()
    for row in table_rows:
        values = " & ".join(
            f"{float(row[f'{s}_mse']):.3f}" if row[f"{s}_mse"] != "N/A" else "N/A"
            for s in mse_subsets
        )
        print(f"{row['run']}: {values}")

    # Determine output path: explicit -o, or auto-derive from inputs
    if args.output:
        output_path = args.output
    else:
        resolved = [p.resolve() for p in args.paths]
        # Single path: write into it; multiple paths: write into common parent
        if len(resolved) == 1:
            parent_dir = resolved[0] if resolved[0].is_dir() else resolved[0].parent
        else:
            parents = [p if p.is_dir() else p.parent for p in resolved]
            # Walk up until we find a directory that contains all input paths
            common = parents[0]
            for p in parents[1:]:
                while common != p and common not in p.parents:
                    common = common.parent
            parent_dir = common
        output_path = parent_dir / "subset_metrics.csv"

    col_headers = ["run"]
    for s in SUBSETS:
        col_headers += [f"{s}_n", f"{s}_mae", f"{s}_mse"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=col_headers)
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
