#!/usr/bin/env python3
"""
Compute moving average MAE by claim modality.

Generates 3 plots:
1. Text-only claims (no media)
2. Text+image claims (images only, no videos)
3. Text+video claims (videos only, no images)

Claims with both images and videos are excluded.

```
python baselines/scripts/compute_mae_by_modality.py \
	--multi <...>/Veritas/baselines/results/longitudinal_no-search/ \
	-n 200 && \
python baselines/scripts/compute_mae_by_modality.py \
	--multi <...>/Veritas/baselines/results/longitudinal_custom-search/ \
	-n 200
```
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from plotting_utils import (
    get_model_cutoff,
    find_model_results,
    plot_multi_model_metric,
)

# Default path to claims file
DEFAULT_CLAIMS_PATH = Path(__file__).parent.parent / "data" / "veritas_release" / "veritas_longitudinal_2020_q1_2025_q4" / "claims.json"

# Verdict to numerical value mapping
VERDICT_TO_VALUE = {
    "INTACT": 1.0,
    "COMPROMISED": -1.0,
    "UNKNOWN": 0.0,
}


def verdict_to_value(verdict: str) -> float:
    """Convert verdict string to numerical value."""
    return VERDICT_TO_VALUE.get(verdict.upper(), 0.0)


def classify_claim_modality(claim: dict) -> str | None:
    """
    Classify a claim by its modality.

    Returns:
        "text_only" - no media
        "text_image" - has image(s), no videos
        "text_video" - has video(s), no images
        None - has both images and videos (excluded)
    """
    media = claim.get("media", [])

    if not media:
        return "text_only"

    has_image = any(m.get("type") == "image" for m in media)
    has_video = any(m.get("type") == "video" for m in media)

    if has_image and has_video:
        return None  # Exclude claims with both
    elif has_image:
        return "text_image"
    elif has_video:
        return "text_video"
    else:
        return "text_only"  # No recognized media types


def load_claims_by_modality(claims_path: Path) -> dict[str, dict[int, datetime]]:
    """
    Load claims and return mappings from claim_id to datetime, grouped by modality.

    Returns:
        Dict with keys "text_only", "text_image", "text_video",
        each mapping claim_id -> datetime
    """
    with open(claims_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    modality_maps = {
        "text_only": {},
        "text_image": {},
        "text_video": {},
    }

    excluded_count = 0

    for claim in data["claims"]:
        claim_id = claim["id"]
        date_str = claim["date"]
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.replace(tzinfo=None)

        modality = classify_claim_modality(claim)
        if modality is None:
            excluded_count += 1
            continue

        modality_maps[modality][claim_id] = dt

    print(f"Claims by modality:")
    print(f"  Text-only:   {len(modality_maps['text_only'])}")
    print(f"  Text+image:  {len(modality_maps['text_image'])}")
    print(f"  Text+video:  {len(modality_maps['text_video'])}")
    print(f"  Excluded (image+video): {excluded_count}")

    return modality_maps


def load_results_for_modality(
    results_path: Path,
    claim_dates: dict[int, datetime],
) -> list[dict]:
    """
    Load results CSV and filter to claims in the given modality.

    Returns list of dicts with keys: claim_id, date, gt_value, pred_value, abs_error
    """
    results = []
    with open(results_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["status"] != "success":
                continue

            claim_id = int(row["claim_id"])
            if claim_id not in claim_dates:
                continue

            gt_value = verdict_to_value(row["gt_class"])
            pred_value = verdict_to_value(row["verdict"])

            error = pred_value - gt_value
            abs_error = abs(error)

            results.append({
                "claim_id": claim_id,
                "date": claim_dates[claim_id],
                "gt_value": gt_value,
                "pred_value": pred_value,
                "abs_error": abs_error,
            })

    results.sort(key=lambda x: x["date"])
    return results


def compute_n_claim_moving_average_mae(
    results: list[dict],
    n: int = 100,
) -> tuple[list[datetime], list[float]]:
    """
    Compute moving average MAE over N claims.

    Returns:
        Tuple of (dates, mae_values)
    """
    if len(results) < n:
        print(f"Warning: Only {len(results)} results, less than window size {n}")
        n = len(results)

    if n == 0:
        return [], []

    dates = []
    mae_values = []

    for i in range(n - 1, len(results)):
        window = results[i - n + 1:i + 1]
        mae = sum(r["abs_error"] for r in window) / len(window)
        dates.append(window[-1]["date"])
        mae_values.append(mae)

    return dates, mae_values


def compute_abstain_baseline_mae(
    results: list[dict],
    n: int = 100,
) -> tuple[list[datetime], list[float]]:
    """
    Compute moving average MAE for always predicting NEI (0.0).

    Returns:
        Tuple of (dates, mae_values)
    """
    if len(results) < n:
        n = len(results)

    if n == 0:
        return [], []

    dates = []
    mae_values = []

    for i in range(n - 1, len(results)):
        window = results[i - n + 1:i + 1]
        # Error when always predicting 0.0: abs_error = |gt_value|
        mae = sum(abs(r["gt_value"]) for r in window) / len(window)
        dates.append(window[-1]["date"])
        mae_values.append(mae)

    return dates, mae_values


def main():
    parser = argparse.ArgumentParser(
        description="Compute moving average MAE by claim modality.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Generates 3 plots:
  1. Text-only claims (no media)
  2. Text+image claims (images only)
  3. Text+video claims (videos only)

Claims with both images and videos are excluded.

Examples:
  python compute_mae_by_modality.py --multi path/to/results_dir/
  python compute_mae_by_modality.py --multi results_dir/ -n 50
        """
    )
    parser.add_argument("--multi", type=Path, required=True,
                        help="Directory containing model subdirectories with results.csv files")
    parser.add_argument("-n", "--n-claims", type=int, default=100,
                        help="Window size for N-claim moving average (default: 100)")
    parser.add_argument("--claims", type=Path, default=None,
                        help="Path to claims.json (default: auto-detect)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output directory for plots")

    args = parser.parse_args()

    claims_path = args.claims or DEFAULT_CLAIMS_PATH
    if not claims_path.exists():
        raise FileNotFoundError(f"Claims file not found: {claims_path}")

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
    print(f"Output directory: {output_dir}")
    print()

    # Load claims by modality
    modality_maps = load_claims_by_modality(claims_path)

    modality_config = {
        "text_only": {
            "title": "Text-Only Claims",
            "filename": f"MA{args.n_claims}_claims_MAE_text-only.pdf",
        },
        "text_image": {
            "title": "Text + Image Claims",
            "filename": f"MA{args.n_claims}_claims_MAE_image-only.pdf",
        },
        "text_video": {
            "title": "Text + Video Claims",
            "filename": f"MA{args.n_claims}_claims_MAE_video-only.pdf",
        },
    }

    # Process each modality
    for modality, config in modality_config.items():
        print(f"\n{'='*60}")
        print(f"Processing: {config['title']}")
        print(f"{'='*60}")

        claim_dates = modality_maps[modality]
        if not claim_dates:
            print(f"  No claims for this modality, skipping.")
            continue

        all_mae_data = {}
        first_results = None

        for model_name, results_path in model_results:
            print(f"  Processing {model_name}...")
            results = load_results_for_modality(results_path, claim_dates)
            print(f"    Loaded {len(results)} results")

            if not results:
                print(f"    Skipping {model_name}: no valid results")
                continue

            if first_results is None:
                first_results = results

            dates, mae_values = compute_n_claim_moving_average_mae(results, args.n_claims)

            if dates:
                all_mae_data[model_name] = (dates, mae_values)

        # Add Always Abstain baseline
        if first_results:
            print(f"  Computing 'Always Abstain' baseline (NEI = 0.0)...")
            abstain_dates, abstain_mae = compute_abstain_baseline_mae(first_results, args.n_claims)
            if abstain_dates:
                all_mae_data["Always Abstain"] = (abstain_dates, abstain_mae)

        # Build cutoff dates
        model_cutoffs = {}
        for model_name in all_mae_data.keys():
            cutoff = get_model_cutoff(model_name)
            if cutoff:
                model_cutoffs[model_name] = cutoff

        # Generate plot
        if all_mae_data:
            print(f"\n  Generating plot for {config['title']}...")
            for model_name, cutoff in model_cutoffs.items():
                print(f"    Cutoff for {model_name}: {cutoff.strftime('%Y-%m-%d')}")

            plot_multi_model_metric(
                all_mae_data,
                output_dir / config["filename"],
                metric_name="MAE",
                title="",
                ylabel='MAE',
                cutoff_dates=model_cutoffs if model_cutoffs else None,
                max_jump=0.3,
                legend_loc='upper left',
            )
        else:
            print(f"  No data to plot for {config['title']}")

    print("\nDone!")


if __name__ == "__main__":
    main()
