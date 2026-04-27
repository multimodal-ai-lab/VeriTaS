"""Print examples from the Disputed group and show all explanations.

This script scans all verdicts in the database and, for each of the six
properties covered by our labeling scheme, identifies the examples with the
highest disagreement between the individual model annotations. Disagreement is
measured as the maximum difference between the individual tendencies
max(tendency) - min(tendency) for the respective LabelAggregated.

For each top-ranked example, the script prints:
- The claim id
- The property (and medium reference when applicable)
- The list of individual tendencies
- All individual explanations (aiming to show all four, if available)

Run (requires DB access configured in config/globals.yaml):
    python -m scripts.stats.print_disagreements

Configuration:
- Use get_params() to provide parameters programmatically if desired.
- Defaults: top=20, prop=None, include_media=True, only_four=True
"""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from veritas.common.annotation.rating import RatingAggregated
from veritas.common.verdict import Verdict
from veritas.db import db


# Properties as defined/used in plot_verdict_stats.py
CLAIM_LEVEL_PROPERTIES = [
    "clarity",
    "veracity",
    "context_coverage",
    "intent",
]
MEDIA_PROPERTIES = ["authenticity", "contextualization"]


@dataclass
class Example:
    claim_id: int
    property: str
    label: RatingAggregated
    medium_ref: Optional[str] = None  # for media-level properties
    review_ids: set[int] | None = None  # reviews that the verdict is based on

    @property
    def tendencies(self) -> list[float]:
        return [float(l.score) for l in self.label.individual_ratings]

    @property
    def max_diff(self) -> float:
        vals = [float(x) for x in self.tendencies]
        vals = [x for x in vals if np.isfinite(x)]
        if not vals:
            return 0.0
        return max(vals) - min(vals)

    @property
    def n_individuals(self) -> int:
        return len(self.label.individual_ratings)


def _iter_examples(verdicts: Iterable[Verdict], include_media: bool = True, prop_filter: str | None = None):
    for v in verdicts:
        # Claim-level
        for prop in CLAIM_LEVEL_PROPERTIES:
            if prop_filter and prop != prop_filter:
                continue
            lbl = getattr(v, prop, None)
            if isinstance(lbl, RatingAggregated):
                yield Example(claim_id=v.claim_id, property=prop, label=lbl, review_ids=getattr(v, "review_ids", None))

        # Media-level
        if include_media and v.media_verdicts:
            for mv in v.media_verdicts:
                for prop in MEDIA_PROPERTIES:
                    if prop_filter and prop != prop_filter:
                        continue
                    lbl = getattr(mv, prop, None)
                    if isinstance(lbl, RatingAggregated):
                        yield Example(claim_id=v.claim_id, property=prop, label=lbl, medium_ref=mv.reference, review_ids=getattr(v, "review_ids", None))


def _pretty_prop(prop: str) -> str:
    try:
        from veritas.common.annotation.property import PROPERTIES
        return PROPERTIES.get(prop).name if PROPERTIES.get(prop) else prop
    except Exception:
        return prop


def _print_example(ex: Example, claim_text: Optional[str] = None, review_urls: Optional[list[str]] = None) -> None:
    header = f"Claim {ex.claim_id} — {_pretty_prop(ex.property)}"
    if ex.medium_ref:
        header += f" — medium: {ex.medium_ref}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    # Print claim text at the top before explanations
    if claim_text:
        print("Claim text:")
        print(claim_text.strip())
        print()

    # Print corresponding review URLs, if provided
    if review_urls:
        print("Reviews:")
        for url in review_urls:
            print(str(url))
        print()

    tends = ", ".join(f"{t:+.3f}" for t in ex.tendencies)
    print(f"Individual tendencies ({ex.n_individuals}): [{tends}]  |  max diff = {ex.max_diff:.3f}")
    print()

    for i, lbl in enumerate(ex.label.individual_ratings, start=1):
        model = getattr(lbl, "model", f"model{i}")
        expl = getattr(lbl, "explanation", None) or "(no explanation)"
        print(f"[{i}] {model}:")
        print(expl.strip())
        print()


def _model_name(lbl, fallback: str) -> str:
    return getattr(lbl, "model", fallback) or fallback


def _round_tendency(x: float) -> float:
    """Round tendencies to 3 decimals to make equality comparisons robust.

    The labeling scheme typically uses steps of 1/3, so 3 decimals are safe.
    """
    try:
        return round(float(x), 3)
    except Exception:
        return float(x)


def _compute_summary(examples: list[Example]) -> dict:
    """Compute per-property stats with three agreement groups:
    - Full Consensus: all 4 predictions equal (after rounding)
    - Majority: some triplet (3 predictions) whose pairwise spread <= 1/3; the remaining one is tracked as the outlier
    - Disputed: all other cases (including when fewer/more than four valid predictions)

    Returns dict per property with group counts and per-model outlier counts.
    """
    per_prop_examples: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        per_prop_examples[ex.property].append(ex)

    summary: dict[str, dict] = {}

    for prop, exs in per_prop_examples.items():
        total = len(exs)
        group_counts = Counter({"full_consensus": 0, "majority": 0, "disputed": 0})
        outlier_models = Counter()

        for ex in exs:
            group, outlier = _classify_group(ex)
            if group == "Full Consensus":
                group_counts["full_consensus"] += 1
            elif group == "Majority":
                group_counts["majority"] += 1
                if outlier:
                    outlier_models[outlier] += 1
            else:
                group_counts["disputed"] += 1

        # Prepare summary entry
        summary[prop] = {
            "total": total,
            "groups": {
                "Full Consensus": group_counts["full_consensus"],
                "Majority": group_counts["majority"],
                "Disputed": group_counts["disputed"],
            },
            "outlier_counts": dict(outlier_models),  # model -> count
        }

    return summary


def _print_summary(summary: dict) -> None:
    if not summary:
        print("No data to summarize.")
        print()
        return
    print("===== Summary (agreement groups and outliers) =====")
    for prop in sorted(summary.keys()):
        s = summary[prop]
        pretty = _pretty_prop(prop)
        total = s["total"] or 0
        g = s["groups"]
        print(f"- {pretty} (n={total}): Full Consensus: {g['Full Consensus']}, Majority: {g['Majority']}, Disputed: {g['Disputed']}")
        oc = s.get("outlier_counts", {})
        if oc:
            print("    Outlier counts by model:")
            for model, count in sorted(oc.items(), key=lambda x: (-x[1], x[0])):
                print(f"      {model}: {count}")
        else:
            print("    Outlier counts by model: (none)")
    print()


def _classify_group(ex: Example) -> tuple[str, Optional[str]]:
    """Classify a single example into one of the groups and return (group, outlier_model).

    group in {"Full Consensus", "Majority", "Disputed"}. For "Majority", outlier_model is the
    model name of the unique outlier; otherwise None.
    """
    # Build index-aligned list of finite rounded tendencies
    vals_idx: list[tuple[int, float]] = []
    for i, lbl in enumerate(ex.label.individual_ratings):
        try:
            val = float(lbl.score)
        except Exception:
            continue
        if np.isfinite(val):
            vals_idx.append((i, _round_tendency(val)))

    if len(vals_idx) != 4:
        return ("Disputed", None)

    vals = [v for _, v in vals_idx]
    if max(vals) - min(vals) == 0.0:
        return ("Full Consensus", None)

    # Majority detection by removing each element and checking spread of the remaining three
    eps = 1e-9
    best_removed_idx = None
    best_spread = float("inf")
    for k in range(4):
        triplet = [vj for m, (_, vj) in enumerate(vals_idx) if m != k]
        spread = max(triplet) - min(triplet) if triplet else float("inf")
        if spread < best_spread - eps:
            best_spread = spread
            best_removed_idx = k

    if best_spread <= (1.0/3.0 + 1e-6) and best_removed_idx is not None:
        outlier_orig_index = vals_idx[best_removed_idx][0]
        lbl = ex.label.individual_ratings[outlier_orig_index]
        return ("Majority", _model_name(lbl, f"model{outlier_orig_index+1}"))
    return ("Disputed", None)


async def run(top: int = 20, prop: str | None = None, include_media: bool = True, only_four: bool = True) -> None:
    await db.connect_maybe_initialize()
    verdicts = await db.get_verdicts()

    examples = list(_iter_examples(verdicts, include_media=include_media, prop_filter=prop))
    if only_four:
        examples = [e for e in examples if e.n_individuals == 4]

    # Filter to Disputed group examples only
    disputed_examples = [e for e in examples if _classify_group(e)[0] == "Disputed"]
    # Sort disputed examples by disagreement descending (for readability)
    disputed_examples.sort(key=lambda e: e.max_diff, reverse=True)

    # Print summary across the filtered set
    _print_summary(_compute_summary(examples))

    # Print top N
    for ex in disputed_examples[: max(0, int(top))]:
        # Fetch and show the full claim text
        claim = await db.get_claim_by_id(ex.claim_id)
        claim_text = getattr(claim, "data", None) if claim is not None else None
        # Resolve review URLs for the example
        review_urls: list[str] = []
        if ex.review_ids:
            for rid in sorted(ex.review_ids):
                try:
                    rev = await db.get_review_by_id(rid)
                    if rev and getattr(rev, "url", None):
                        review_urls.append(str(rev.url))
                except Exception:
                    # Be robust: ignore failures for individual reviews
                    continue
        _print_example(ex, claim_text, review_urls if review_urls else None)


def _print_example_with_claim(ex: Example, claim_text: Optional[str]) -> None:
    # Backward compatibility if used elsewhere: delegate to new signature
    _print_example(ex, claim_text, None)


def main() -> None:
    # Obtain parameters via function with defaults; users can import and override programmatically
    asyncio.run(run(top=50))


if __name__ == "__main__":
    main()
