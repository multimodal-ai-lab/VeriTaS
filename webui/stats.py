"""Pure shaping of aggregate rows into the payloads the dashboard renders.

Kept free of I/O so the arithmetic behind every displayed number is unit-tested.
The authoritative statistics for a write-up remain the ones produced by
`scripts.gold_evidence.run_temporal_analysis`; this module only summarizes what
is currently in the database for interactive inspection.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

#: Order used whenever claim statuses are displayed as a series.
STATUS_ORDER = ("pending", "extracted", "filtered", "deferred", "accepted", "rejected")

#: Order used for inadmissibility reasons, mirroring `admissibility.REASON_ORDER`.
REASON_ORDER = (
    "fact_check_source",
    "not_filtered",
    "low_extraction_confidence",
    "inaccessible",
    "undated_source",
    "unfaithful",
    "after_fact_check",
    "later_event",
)


def series(rows: Iterable[dict], *, label_key: str = "label",
           count_key: str = "count", order: Sequence[str] = ()) -> list[dict]:
    """Turns `(label, count)` rows into a chart-ready series with shares.

    Rows whose label appears in `order` are sorted by that order; the rest follow,
    sorted by descending count. Unknown labels become the string "unknown"."""
    counted: dict[str, int] = {}
    for row in rows:
        label = row.get(label_key)
        label = "unknown" if label is None else str(label)
        counted[label] = counted.get(label, 0) + int(row.get(count_key) or 0)

    total = sum(counted.values())
    ranking = {label: index for index, label in enumerate(order)}
    ordered = sorted(
        counted.items(),
        key=lambda item: (ranking.get(item[0], len(ranking)), -item[1], item[0]),
    )
    return [
        {"label": label, "count": count, "share": (count / total) if total else None}
        for label, count in ordered
    ]


def share(numerator: int | None, denominator: int | None) -> float | None:
    """`numerator / denominator`, or None when the denominator is zero/unknown."""
    if not denominator:
        return None
    return (numerator or 0) / denominator


def describe(values: Sequence[float | None]) -> dict:
    """Count, mean, standard deviation, and the five-number summary."""
    clean = sorted(float(value) for value in values if value is not None
                   and not (isinstance(value, float) and math.isnan(value)))
    if not clean:
        return {"n": 0, "mean": None, "std": None, "min": None, "p25": None,
                "median": None, "p75": None, "max": None}

    n = len(clean)
    mean = sum(clean) / n
    variance = sum((value - mean) ** 2 for value in clean) / (n - 1) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": math.sqrt(variance),
        "min": clean[0],
        "p25": _quantile(clean, 0.25),
        "median": _quantile(clean, 0.5),
        "p75": _quantile(clean, 0.75),
        "max": clean[-1],
    }


def _quantile(ordered: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile of an already sorted sequence."""
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def histogram(values: Sequence[float | None], *, bins: int = 20,
              low: float | None = None, high: float | None = None) -> dict:
    """Equal-width histogram. Values outside `[low, high]` are clamped into the
    edge bins so nothing silently disappears from the displayed distribution."""
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"bins": [], "n": 0, "low": None, "high": None}

    low = min(clean) if low is None else low
    high = max(clean) if high is None else high
    if high <= low:
        high = low + 1.0

    width = (high - low) / bins
    counts = [0] * bins
    for value in clean:
        index = int((min(max(value, low), high) - low) / width)
        counts[min(index, bins - 1)] += 1

    return {
        "n": len(clean),
        "low": low,
        "high": high,
        "bins": [
            {"start": low + index * width, "end": low + (index + 1) * width, "count": count}
            for index, count in enumerate(counts)
        ],
    }


def recoverability(rows: Iterable[dict]) -> dict:
    """The 2x2 contingency of `E_claim` vs `E_factcheck` recoverability.

    `rows` are `(claim_close, fact_check_close, count)` triples over claims that
    have a stored result for *both* conditions. `only_E_factcheck > only_E_claim`
    is the direction indicating that evidence which appeared during the
    fact-checking period is needed to recover the gold verdict. The significance
    test on those discordant pairs is reported by the analysis script, which is
    the single source of truth for published numbers."""
    both = only_claim = only_fact_check = neither = 0
    for row in rows:
        claim_close = row.get("claim_close")
        fact_check_close = row.get("fact_check_close")
        count = int(row.get("count") or 0)
        if claim_close is None or fact_check_close is None:
            continue
        if claim_close and fact_check_close:
            both += count
        elif claim_close:
            only_claim += count
        elif fact_check_close:
            only_fact_check += count
        else:
            neither += count

    paired = both + only_claim + only_fact_check + neither
    n_claim = both + only_claim
    n_fact_check = both + only_fact_check
    return {
        "n_paired_claims": paired,
        "contingency": {
            "both": both,
            "only_E_claim": only_claim,
            "only_E_factcheck": only_fact_check,
            "neither": neither,
        },
        "recoverable_from_E_claim": n_claim,
        "recoverable_from_E_factcheck": n_fact_check,
        "rate_E_claim": share(n_claim, paired),
        "rate_E_factcheck": share(n_fact_check, paired),
        "gain_from_fact_check_period": share(n_fact_check - n_claim, paired),
    }
