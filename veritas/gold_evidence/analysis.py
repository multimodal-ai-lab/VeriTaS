"""Temporal analysis of reconstructed gold evidence (Spec §5).

Pure aggregation over already-computed rows: no I/O, no LLM calls, so every
reported number is reproducible from the exported CSV/JSON alone.

The interval under study is

    t_c < t_e <= t_f

i.e. evidence that became available *after the claim was made* but *before the
professional fact-check was published*.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from veritas.gold_evidence import CONDITION_CLAIM, CONDITION_FACT_CHECK
from veritas.gold_evidence.admissibility import in_window
from veritas.gold_evidence.models import Evidence, to_naive


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------

@dataclass
class ClaimRecord:
    """One claim-level row of the analysis export."""

    claim_id: int
    t_c: datetime | None
    t_f: datetime | None
    status: str | None
    reason: str | None
    released: bool
    is_rectified: bool
    language: str | None
    n_candidates: int
    n_admissible: int
    n_sources: int
    n_in_window: int
    n_undated: int
    n_deferred: int
    #: Stage 1 output beyond the evidence itself.
    n_rationales: int
    #: Evidence items the rationale rests on, and how many lost every source.
    n_essential: int
    n_essential_lost: int
    n_evidence_claim: int
    n_evidence_fact_check: int
    gold_integrity: float | None
    gold_veracity: float | None
    gold_context_coverage: float | None
    has_media: bool
    recoverable_claim: bool | None = None
    recoverable_fact_check: bool | None = None
    max_diff_claim: float | None = None
    max_diff_fact_check: float | None = None

    @property
    def fact_check_duration(self) -> float | None:
        """(t_f - t_c) in days."""
        if self.t_c is None or self.t_f is None:
            return None
        return (self.t_f - self.t_c).total_seconds() / 86400.0

    @property
    def has_window_evidence(self) -> bool:
        return self.n_in_window > 0

    @property
    def has_rationale(self) -> bool:
        return self.n_rationales > 0

    @property
    def evidence_free(self) -> bool:
        """No admissible evidence at all - legitimate when a rationale carries the
        verdict (arithmetic, an internal contradiction, the claim's own media)."""
        return self.n_admissible == 0

    def to_dict(self) -> dict:
        data = {k: v for k, v in self.__dict__.items()}
        data["t_c"] = self.t_c.isoformat() if self.t_c else None
        data["t_f"] = self.t_f.isoformat() if self.t_f else None
        data["fact_check_duration_days"] = self.fact_check_duration
        data["has_window_evidence"] = self.has_window_evidence
        data["has_rationale"] = self.has_rationale
        data["evidence_free"] = self.evidence_free
        return data


def build_claim_record(*, claim, gold, evidence: list[Evidence],
                       t_c: datetime | None, t_f: datetime | None,
                       results: dict[str, dict] | None = None,
                       rationales: list | None = None) -> ClaimRecord:
    """Assembles the claim-level row from the stored objects."""
    from veritas.gold_evidence.admissibility import essential, in_window, lost_essential, select

    results = results or {}
    t_c, t_f = to_naive(t_c), to_naive(t_f)
    admissible = [e for e in evidence if e.admissible]
    claim_set = select(evidence, CONDITION_CLAIM)
    fact_check_set = select(evidence, CONDITION_FACT_CHECK)

    claim_result = results.get(CONDITION_CLAIM) or {}
    fact_check_result = results.get(CONDITION_FACT_CHECK) or {}

    return ClaimRecord(
        claim_id=claim.id,
        t_c=t_c,
        t_f=t_f,
        status=claim.gold_evidence_status,
        reason=claim.gold_evidence_reason,
        released=bool(claim.released),
        is_rectified=bool(claim.is_rectified),
        language=claim.language,
        n_candidates=len(evidence),
        n_admissible=len(admissible),
        n_sources=sum(len(e.sources) for e in evidence),
        n_in_window=sum(1 for e in evidence for s in e.sources
                        if s.admissible and in_window(s)),
        n_undated=sum(1 for e in evidence for s in e.sources if s.available_since is None),
        n_deferred=sum(1 for e in evidence for s in e.sources if s.deferred),
        n_rationales=len(rationales or []),
        n_essential=len(essential(evidence)),
        n_essential_lost=len(lost_essential(evidence)),
        n_evidence_claim=len(claim_set),
        n_evidence_fact_check=len(fact_check_set),
        gold_integrity=_score(_safe(lambda: gold.integrity)) if gold else None,
        gold_veracity=_score(gold.veracity) if gold else None,
        gold_context_coverage=_score(gold.context_coverage) if gold else None,
        has_media=bool(gold and gold.media_verdicts),
        recoverable_claim=claim_result.get("is_close"),
        recoverable_fact_check=fact_check_result.get("is_close"),
        max_diff_claim=claim_result.get("max_property_diff"),
        max_diff_fact_check=fact_check_result.get("max_property_diff"),
    )


def build_evidence_records(evidence: Evidence, *, claim_id: int,
                           t_c: datetime | None,
                           t_f: datetime | None) -> list[dict]:
    """One row per *source* of an evidence item.

    The rows are source-level because everything that varies within an item varies
    per source: the availability time, the accessibility, the faithfulness, the
    admissibility. The item's own fields - proposition, role, and whether it
    survived at all - are repeated on each row so the export can be grouped either
    way."""
    return [_source_record(source, evidence, claim_id=claim_id, t_c=t_c, t_f=t_f)
            for source in evidence.sources]


def _source_record(source, evidence: Evidence, *, claim_id: int,
                   t_c: datetime | None, t_f: datetime | None) -> dict:
    temporal = source.temporal_validation
    faithfulness = source.faithfulness
    return {
        "source_id": source.id,
        "evidence_id": evidence.id,
        "claim_id": claim_id,
        "review_id": evidence.review_id,
        "proposition": evidence.proposition,
        "role": evidence.role.value,
        "extraction_confidence": evidence.extraction_confidence,
        "is_multimodal": evidence.is_multimodal,
        # Whether the *item* survived; a discarded source of a surviving item costs
        # the reconstruction nothing.
        "evidence_admissible": evidence.admissible,
        "n_sources": len(evidence.sources),

        "source_name": source.name,
        "source_kind": source.kind.value,
        "source_locator": source.locator,
        "source_domain": source.domain,
        "source_proximity": source.proximity.value,
        "available_since": source.available_since.isoformat() if source.available_since else None,
        "t_e_minus_t_c_days": source.time_to_claim(t_c),
        "t_e_minus_t_f_days": source.time_to_fact_check(t_f),
        "in_window": in_window(source),
        "accessible": source.accessible,
        "faithfulness": faithfulness.assessment if faithfulness else None,
        "before_claim": temporal.before_claim if temporal else None,
        "before_fact_check": temporal.before_fact_check if temporal else None,
        # §3.3 (3) is judged for the proposition, so it is the same on every row
        # of one item.
        "later_event": evidence.relies_on_later_event,
        "admissible": source.admissible,
        "inadmissibility_reason": source.inadmissibility_reason,
        "deferred": source.deferred,
    }


# ---------------------------------------------------------------------------
# Aggregates (Spec §5)
# ---------------------------------------------------------------------------

def aggregate(claim_records: list[ClaimRecord],
              evidence_records: list[dict]) -> dict:
    """Computes every quantity §5 asks for."""
    n_claims = len(claim_records)
    processed = [c for c in claim_records if c.n_candidates > 0]

    deltas_claim = _values(evidence_records, "t_e_minus_t_c_days")
    deltas_fact_check = _values(evidence_records, "t_e_minus_t_f_days")
    durations = [c.fact_check_duration for c in claim_records
                 if c.fact_check_duration is not None]

    admissible_records = [e for e in evidence_records if e.get("admissible")]
    window_records = [e for e in admissible_records if e.get("in_window")]

    rejected = [e for e in evidence_records if e.get("admissible") is False]
    # A source can be lost without costing the reconstruction anything, as long as
    # its item kept another one. That difference is what this reports.
    fatal = [e for e in rejected if e.get("evidence_admissible") is False]

    return {
        "n_claims": n_claims,
        "n_claims_processed": len(processed),
        "n_sources": len(evidence_records),
        "n_evidence_candidates": sum(c.n_candidates for c in claim_records),
        "n_sources_admissible": len(admissible_records),
        "n_sources_in_window": len(window_records),
        "n_sources_deferred": sum(1 for e in evidence_records if e.get("deferred")),

        # Proportion of claims containing post-claim / pre-fact-check evidence
        "share_claims_with_window_evidence": _share(
            sum(1 for c in claim_records if c.has_window_evidence), n_claims),
        # Fraction of admissible sources falling into the interval
        "share_evidence_in_window": _share(len(window_records), len(admissible_records)),
        "mean_window_items_per_claim": _mean([c.n_in_window for c in claim_records]),

        # Distributions of the three time differences
        "distribution_t_e_minus_t_c": describe(deltas_claim),
        "distribution_t_e_minus_t_f": describe(deltas_fact_check),
        "distribution_t_f_minus_t_c": describe(durations),

        # Categorical distributions over the admissible evidence
        "source_kind_distribution": _counts(admissible_records, "source_kind"),
        "source_proximity_distribution": _counts(admissible_records, "source_proximity"),
        "role_distribution": _counts(admissible_records, "role"),
        "modality_composition": {
            "multimodal": sum(1 for e in admissible_records if e.get("is_multimodal")),
            "text_only": sum(1 for e in admissible_records if not e.get("is_multimodal")),
            "share_multimodal": _share(
                sum(1 for e in admissible_records if e.get("is_multimodal")),
                len(admissible_records)),
        },

        # Rejection accounting
        "share_sources_rejected": _share(len(rejected), len(evidence_records)),
        "share_sources_rejected_fatally": _share(len(fatal), len(evidence_records)),
        "rejection_reasons": _counts(rejected, "inadmissibility_reason"),
        "instance_statuses": dict(Counter(c.status for c in claim_records)),
        "instance_rejection_reasons": dict(Counter(c.reason for c in claim_records if c.reason)),
        "share_instances_rejected_insufficient": _share(
            sum(1 for c in claim_records if c.reason == "insufficient_evidence"), n_claims),
        "share_instances_rejected_essential_lost": _share(
            sum(1 for c in claim_records if c.reason == "essential_evidence_lost"), n_claims),

        # Rationale and redundancy accounting
        "share_claims_with_rationale": _share(
            sum(1 for c in claim_records if c.has_rationale), n_claims),
        "share_claims_without_admissible_evidence": _share(
            sum(1 for c in claim_records if c.evidence_free), n_claims),
        "n_essential": sum(c.n_essential for c in claim_records),
        "n_essential_lost": sum(c.n_essential_lost for c in claim_records),
        "sources_per_evidence": describe(_sources_per_evidence(evidence_records)),

        # The central question
        "recoverability": recoverability(claim_records),
    }


def recoverability(claim_records: list[ClaimRecord]) -> dict:
    """The 2x2 contingency of `E_c` vs `E_f` recoverability, plus
    McNemar's exact test on the discordant pairs.

    A significant result with `only_fact_check > only_claim` is the evidence that
    material appearing during the professional fact-checking period is *necessary*
    to reconstruct the gold verdict."""
    paired = [c for c in claim_records
              if c.recoverable_claim is not None and c.recoverable_fact_check is not None]

    both = sum(1 for c in paired if c.recoverable_claim and c.recoverable_fact_check)
    only_claim = sum(1 for c in paired if c.recoverable_claim and not c.recoverable_fact_check)
    only_fact_check = sum(1 for c in paired if not c.recoverable_claim and c.recoverable_fact_check)
    neither = sum(1 for c in paired if not c.recoverable_claim and not c.recoverable_fact_check)

    n_claim = both + only_claim
    n_fact_check = both + only_fact_check

    return {
        "n_paired_claims": len(paired),
        "contingency": {
            "both": both,
            "only_E_c": only_claim,
            "only_E_f": only_fact_check,
            "neither": neither,
        },
        "recoverable_from_E_c": n_claim,
        "recoverable_from_E_f": n_fact_check,
        "rate_E_c": _share(n_claim, len(paired)),
        "rate_E_f": _share(n_fact_check, len(paired)),
        "gain_from_fact_check_period": _share(n_fact_check - n_claim, len(paired)),
        "mcnemar_exact_p": mcnemar_exact(only_fact_check, only_claim),
    }


def mcnemar_exact(b: int, c: int) -> float | None:
    """Two-sided exact McNemar test on the discordant pair counts.

    Under H0 each discordant pair is equally likely to fall either way, so the
    count follows Binomial(b + c, 0.5). Implemented directly to avoid adding a
    SciPy dependency."""
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def describe(values: list[float]) -> dict:
    """Summary statistics of a distribution, plus the raw values for plotting."""
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "mean": _mean(ordered),
        "std": _std(ordered),
        "min": ordered[0],
        "p05": _quantile(ordered, 0.05),
        "q1": _quantile(ordered, 0.25),
        "median": _quantile(ordered, 0.5),
        "q3": _quantile(ordered, 0.75),
        "p95": _quantile(ordered, 0.95),
        "max": ordered[-1],
        "values": ordered,
    }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _sources_per_evidence(records: list[dict]) -> list[float]:
    """How many sources each evidence item has. The mean says how redundantly the
    fact-checks cite, and therefore how much link rot the reconstruction absorbs."""
    sizes: dict = {}
    for index, record in enumerate(records):
        key = record.get("evidence_id") or f"#{index}"
        sizes[key] = sizes.get(key, 0) + 1
    return [float(size) for size in sizes.values()]


def _values(records: list[dict], key: str) -> list[float]:
    return [r[key] for r in records if r.get(key) is not None]


def _counts(records: list[dict], key: str) -> dict:
    return dict(Counter(r.get(key) for r in records if r.get(key) is not None))


def _share(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mean = _mean(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def _quantile(ordered: list[float], q: float) -> float:
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _score(rating) -> float | None:
    return rating.score if rating is not None else None


def _safe(fn):
    try:
        return fn()
    except (ValueError, AttributeError, TypeError):
        return None
