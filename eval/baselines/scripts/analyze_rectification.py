#!/usr/bin/env python3
"""
Analyze baseline results split by rectification status and integrity group.

Splits results into three groups:
  1. Intact, not rectified  (is_rectified=False, gt_class=Intact)
  2. Intact, rectified      (is_rectified=True,  gt_class=Intact)
  3. Compromised, not rectified (is_rectified=False, gt_class!=Intact)

Computes classification and regression metrics for each group to reveal
whether AI-rectified intact claims are harder or easier to fact-check than
naturally intact ones.

Usage:
    # Using the rectification split (is_rectified in claims.json):
    python analyze_rectification.py --results-csv results.csv --dataset /path/to/claims.json

    # Using the database as fallback:
    python analyze_rectification.py --results-csv results.csv

    # With 7-class label scheme:
    python analyze_rectification.py --results-csv results.csv --dataset /path/to/claims.json --label-scheme 7
"""

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval.baselines.common import compute_metrics, compute_coarsened_metrics, compute_regression_metrics
from eval.baselines.common.types import get_label_scheme, LabelScheme

INTACT_THRESHOLD = 1 / 3

GROUPS = [
    ("intact_native",    "Intact, not rectified"),
    ("intact_rectified", "Intact, rectified"),
    ("compromised",      "Compromised, not rectified"),
]


def load_rectification_from_dataset(dataset_path: Path) -> dict[int, bool]:
    """Read is_rectified from a claims.json that already includes the field."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    for claim in data.get("claims", []):
        cid = claim.get("id")
        is_rectified = claim.get("is_rectified")
        if cid is not None and is_rectified is not None:
            result[int(cid)] = bool(is_rectified)
    return result


async def _query_db(claim_ids: list[int]) -> dict[int, bool]:
    from veritas.db import db
    await db.connect()
    try:
        rows = await db._fetch(
            "SELECT id, is_rectified FROM claims WHERE id = ANY($1)",
            claim_ids,
        )
        return {row["id"]: row["is_rectified"] for row in rows}
    finally:
        await db.close()


def load_rectification_from_db(claim_ids: list[int]) -> dict[int, bool]:
    """Query the Veritas DB for is_rectified status of given claim IDs."""
    return asyncio.run(_query_db(claim_ids))


def classify_row(row: dict, rectification_map: dict[int, bool]) -> str | None:
    """Return the group key for a result row, or None if it can't be classified."""
    try:
        cid = int(row["claim_id"])
    except (ValueError, KeyError):
        return None

    is_rectified = rectification_map.get(cid)
    if is_rectified is None:
        return None

    gt_integrity = row.get("gt_integrity")
    try:
        score = float(gt_integrity)
    except (ValueError, TypeError):
        return None

    intact = score >= INTACT_THRESHOLD

    if intact and not is_rectified:
        return "intact_native"
    if intact and is_rectified:
        return "intact_rectified"
    if not intact and not is_rectified:
        return "compromised"
    # intact=False, is_rectified=True would be inconsistent in a well-formed split
    return None


def compute_group_metrics(rows: list[dict], label_scheme: LabelScheme) -> dict:
    """Compute classification and regression metrics for a group of result rows."""
    y_true, y_pred = [], []
    gt_integrity_values, verdict_values = [], []

    for r in rows:
        if r.get("status") != "success":
            continue
        gt_class = r.get("gt_class")
        verdict = r.get("verdict")
        gt_integrity = r.get("gt_integrity")

        if gt_class and verdict:
            y_true.append(gt_class)
            y_pred.append(verdict)

        if gt_integrity and verdict:
            try:
                gt_integrity_values.append(float(gt_integrity))
                verdict_values.append(verdict)
            except (ValueError, TypeError):
                pass

    metrics = compute_metrics(y_true, y_pred, labels=label_scheme.labels)

    coarsened = compute_coarsened_metrics(y_true, y_pred)
    if coarsened:
        metrics["coarsened_3class"] = coarsened

    regression = compute_regression_metrics(gt_integrity_values, verdict_values, scheme=label_scheme)
    if regression:
        metrics["mse"] = regression["mse"]
        metrics["mae"] = regression["mae"]

    metrics["n_claims"] = len(rows)
    metrics["n_successful"] = sum(1 for r in rows if r.get("status") == "success")
    return metrics


def _fmt(value, fmt_str=".4f") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:{fmt_str}}"
    return str(value)


def print_provider_table(
    provider: str,
    group_metrics: dict[str, dict],
):
    """Print a comparison table across the three groups for one provider."""
    group_keys = [k for k, _ in GROUPS]
    group_labels = [lbl for _, lbl in GROUPS]

    col_w = 20
    row_label_w = 20

    header = f"{'Metric':<{row_label_w}}" + "".join(f"{lbl:>{col_w}}" for lbl in group_labels)
    separator = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print(f"Provider: {provider.upper()}")
    print(separator)
    print(header)
    print(separator)

    # Detect 7-bin mode: any group has coarsened metrics
    has_coarsened = any(
        (group_metrics.get(gk) or {}).get("coarsened_3class")
        for gk in group_keys
    )

    metric_rows = [
        ("N claims",         "n_claims",     "d"),
        ("N successful",     "n_successful", "d"),
        ("Accuracy",         "accuracy",     ".4f"),
        ("Accuracy (3-bin)", "coarsened_3class.accuracy", ".4f"),
        ("Macro F1",         "macro_f1",     ".4f"),
        ("Weighted F1",      "weighted_f1",  ".4f"),
        ("MSE",              "mse",          ".4f"),
        ("MAE",              "mae",          ".4f"),
    ]

    for label, key, fmt_str in metric_rows:
        # Skip coarsened row when not in 7-bin mode
        if key == "coarsened_3class.accuracy" and not has_coarsened:
            continue
        row = f"{label:<{row_label_w}}"
        for gk in group_keys:
            m = group_metrics.get(gk) or {}
            if "." in key:
                outer, inner = key.split(".", 1)
                val = (m.get(outer) or {}).get(inner)
            else:
                val = m.get(key)
            row += f"{_fmt(val, fmt_str):>{col_w}}"
        print(row)

    print(separator)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze baseline results split by rectification status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--results-csv", required=True,
                        help="Path to results.csv from a benchmark run")
    parser.add_argument("--dataset", default=None,
                        help="Path to claims.json containing is_rectified field "
                             "(if omitted, falls back to database query)")
    parser.add_argument("--label-scheme", type=int, choices=[3, 7], default=3,
                        help="Label scheme (3 or 7 classes, default: 3)")
    args = parser.parse_args()

    csv_path = Path(args.results_csv)
    label_scheme = get_label_scheme(args.label_scheme)

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    # Read results CSV
    results = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)

    if not results:
        print("No results found in CSV.", file=sys.stderr)
        return 1

    claim_ids = []
    for r in results:
        try:
            claim_ids.append(int(r["claim_id"]))
        except (ValueError, KeyError):
            continue

    print(f"Loaded {len(results)} results ({len(claim_ids)} distinct claim IDs)")

    # Resolve is_rectified
    if args.dataset:
        dataset_path = Path(args.dataset)
        print(f"Reading is_rectified from {dataset_path}...")
        rectification_map = load_rectification_from_dataset(dataset_path)
    else:
        print(f"Querying database for rectification status of {len(claim_ids)} claims...")
        rectification_map = load_rectification_from_db(claim_ids)

    n_found = sum(1 for cid in claim_ids if cid in rectification_map)
    print(f"Resolved is_rectified for {n_found}/{len(claim_ids)} claims")

    # Assign each row to a group
    grouped: dict[str, list[dict]] = {k: [] for k, _ in GROUPS}
    n_unclassified = 0
    for row in results:
        group = classify_row(row, rectification_map)
        if group is None:
            n_unclassified += 1
        else:
            grouped[group].append(row)

    if n_unclassified:
        print(f"Warning: {n_unclassified} rows could not be classified (missing data or inconsistent split)")

    for key, label in GROUPS:
        n = len(grouped[key])
        print(f"  {label}: {n} rows")

    # Compute and print metrics per provider
    providers = sorted(set(r.get("provider", "") for r in results if r.get("provider")))

    for provider in providers:
        group_metrics = {}
        for gk, _ in GROUPS:
            provider_rows = [r for r in grouped[gk] if r.get("provider") == provider]
            group_metrics[gk] = compute_group_metrics(provider_rows, label_scheme) if provider_rows else {}
        print_provider_table(provider, group_metrics)

    return 0


if __name__ == "__main__":
    exit(main())
