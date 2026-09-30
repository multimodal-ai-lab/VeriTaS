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

#: Order used for instance rejection reasons, mirroring `pipeline`'s constants.
INSTANCE_REASON_ORDER = (
    "no_gold_verdict",
    "no_claim_time",
    "no_fact_check_time",
    "nothing_extracted",
    "key_evidence_lost",
    "sufficiency_validation_failed",
    "insufficient_evidence",
)

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


def to_log(value: float) -> float:
    """Signed logarithmic position of a day difference.

    `sign(v) * log10(1 + |v|)`, so the transform is defined at (and continuous
    through) zero while still compressing the long tails that `t_e - t_c` has.
    The `1 +` keeps sub-day differences from running off to minus infinity."""
    return math.copysign(math.log10(1.0 + abs(value)), value)


def from_log(position: float) -> float:
    """Inverse of `to_log`."""
    return math.copysign(10.0 ** abs(position) - 1.0, position)


#: Day values that get an x-axis tick, mirrored to both sides of zero.
_DECADES = (1, 10, 100, 1_000, 10_000)


def signed_log_histogram(values: Sequence[float | None], *,
                         bins_per_side: int = 24) -> dict:
    """Histogram of signed day differences on a symmetric logarithmic axis.

    Zero is a bin *edge*, never the interior of a bin: negative values are binned
    over `[-U, 0)` and positive ones over `(0, +U]`, where `U` is the shared
    upper bound in log space. Reading a bin therefore never requires deciding
    which side of the reference time it belongs to.

    A difference of exactly zero (the source became available at the very
    reference time) is counted in the innermost *negative* bin, matching the
    admissibility rule, which treats `t_e <= t_c` as "before".
    """
    clean = [float(value) for value in values if value is not None
             and not math.isnan(float(value))]
    if not clean:
        return {"n": 0, "n_negative": 0, "n_positive": 0, "n_zero": 0,
                "limit": 0.0, "max_count": 0, "bins": [], "ticks": []}

    limit = max((abs(to_log(value)) for value in clean), default=0.0)
    if limit <= 0.0:
        limit = to_log(1.0)  # every value is zero; still show a one-day span
    width = limit / bins_per_side

    negative = [0] * bins_per_side  # index 0 is the bin adjacent to zero
    positive = [0] * bins_per_side
    n_zero = 0

    for value in clean:
        position = to_log(value)
        index = min(int(abs(position) / width), bins_per_side - 1)
        if value > 0:
            positive[index] += 1
        else:
            negative[index] += 1
            if value == 0:
                n_zero += 1

    bins = []
    for index in reversed(range(bins_per_side)):  # left to right
        bins.append(_bin(-(index + 1) * width, -index * width, negative[index], "negative"))
    for index in range(bins_per_side):
        bins.append(_bin(index * width, (index + 1) * width, positive[index], "positive"))

    return {
        "n": len(clean),
        "n_negative": sum(negative),
        "n_positive": sum(positive),
        "n_zero": n_zero,
        "limit": limit,
        "max_count": max([entry["count"] for entry in bins], default=0),
        "bins": bins,
        "ticks": _log_ticks(limit),
    }


def _bin(log_start: float, log_end: float, count: int, side: str) -> dict:
    """One bin, carrying both its log-space extent (for drawing) and its extent
    in days (for the tooltip)."""
    return {
        "log_start": log_start,
        "log_end": log_end,
        "start": from_log(log_start),
        "end": from_log(log_end),
        "count": count,
        "side": side,
    }


def _log_ticks(limit: float) -> list[dict]:
    """Ticks at zero and at every power of ten that fits, on both sides."""
    ticks = [{"value": 0.0, "log": 0.0, "label": "0"}]
    for decade in _DECADES:
        position = to_log(float(decade))
        if position > limit:
            break
        for sign in (-1, 1):
            ticks.append({
                "value": sign * decade,
                "log": sign * position,
                "label": f"{'−' if sign < 0 else ''}{decade:,}".replace(",", " "),
            })
    return sorted(ticks, key=lambda tick: tick["log"])


def count_ticks(max_count: int, *, target: int = 4) -> list[int]:
    """Nice, round y-axis ticks from 0 up to at least `max_count`."""
    if max_count <= 0:
        return [0]
    raw = max_count / target
    magnitude = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    for multiple in (1, 2, 2.5, 5, 10):
        step = multiple * magnitude
        if raw <= step:
            break
    step = max(int(round(step)), 1)
    ticks = list(range(0, max_count + step, step))
    return ticks


#: The funnel the evidence passes through, mirroring `admissibility.REASON_ORDER`.
#: Each entry is `(stage label, the reason that removes items before it)`. The
#: reasons are mutually exclusive - the pipeline records the *first* violated
#: criterion - so subtracting them one after another reconstructs the funnel
#: exactly, without needing a second pass over the rows.
FUNNEL_STAGES = (
    ("Stage 2 completed", "not_filtered"),
    ("Confident enough", "low_extraction_confidence"),
    ("Source re-retrieved", "inaccessible"),
    ("Publication time known", "undated_source"),
    ("Still supports the proposition", "unfaithful"),
    ("Available before t_f", "after_fact_check"),
    ("No concurrent fact-check", "concurrent_fact_check"),
    ("No later-event contamination", "later_event"),
)


def evidence_funnel(*, n_candidates: int, n_unfiltered: int, reasons: Iterable[dict],
                    n_admissible: int, n_before_claim: int) -> list[dict]:
    """How many evidence items survive each stage of the reconstruction.

    `reasons` are `(label, count)` rows over the *inadmissible* items. The result
    starts at the Stage 1 candidates and ends at the two evidence sets the
    temporal analysis compares, `E_f` (every admissible item) and `E_c` (those
    that were already available when the claim was made)."""
    dropped = {str(row.get("label")): int(row.get("count") or 0) for row in reasons}

    stages = [{
        "label": "Candidates extracted",
        "count": n_candidates,
        "dropped": 0,
        "reason": None,
        "share": 1.0 if n_candidates else None,
    }]
    remaining = n_candidates

    for index, (label, reason) in enumerate(FUNNEL_STAGES):
        # Items whose Stage 2 never completed leave at the first step, whether
        # they were recorded as `not_filtered` or never got a verdict at all.
        loss = dropped.get(reason, 0) + (n_unfiltered if index == 0 else 0)
        remaining -= loss
        stages.append({
            "label": label,
            "count": remaining,
            "dropped": loss,
            "reason": reason,
            "share": share(remaining, n_candidates),
        })

    # The last funnel step must land exactly on the stored admissible count; if
    # it does not, the reason counts and the admissible flag disagree and the
    # discrepancy is reported rather than hidden.
    stages[-1]["label"] = "Admissible (E_f)"
    stages[-1]["discrepancy"] = remaining - n_admissible
    stages[-1]["count"] = n_admissible
    stages[-1]["share"] = share(n_admissible, n_candidates)

    stages.append({
        "label": "Already available at the claim (E_c)",
        "count": n_before_claim,
        "dropped": max(n_admissible - n_before_claim, 0),
        "reason": "after_claim",
        "share": share(n_before_claim, n_candidates),
    })
    return stages


def recoverability(rows: Iterable[dict]) -> dict:
    """The 2x2 contingency of `E_c` vs `E_f` recoverability.

    The payload keys keep the spelled-out `E_c` / `E_f` form used by
    `veritas.gold_evidence.analysis.recoverability`, so a response from this API
    and a row of the analysis export can be compared field by field. The UI
    displays them as the shorter `E_c` and `E_f`.

    `rows` are `(claim_close, fact_check_close, count)` triples over claims that
    have a stored result for *both* conditions. `only_E_f > only_E_c`
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
            "only_E_c": only_claim,
            "only_E_f": only_fact_check,
            "neither": neither,
        },
        "recoverable_from_E_c": n_claim,
        "recoverable_from_E_f": n_fact_check,
        "rate_E_c": share(n_claim, paired),
        "rate_E_f": share(n_fact_check, paired),
        "gain_from_fact_check_period": share(n_fact_check - n_claim, paired),
    }
