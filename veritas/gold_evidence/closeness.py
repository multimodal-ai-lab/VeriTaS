"""Verdict representation and closeness function for the sufficiency validator.

Kept separate from `sufficiency` so that the decision rule behind the headline
numbers can be imported and tested without pulling in the model providers.
"""

from __future__ import annotations

from pydantic import BaseModel

from veritas.common import Verdict
from veritas.common.annotation import Rating
from veritas.common.annotation.rating import RatingAggregated
from veritas.common.verdict import MediumVerdict
from veritas.gold_evidence import proximity_threshold as default_threshold

MODE_INTEGRITY = "integrity"
MODE_FULL = "full"
ENSEMBLE_MODES = (MODE_INTEGRITY, MODE_FULL)

#: Tolerance for the threshold comparison. Ratings are sums of thirds, so a
#: distance that is conceptually exactly at the threshold can land marginally
#: above it in floating point (e.g. |-0.7 - (-1.0)| == 0.30000000000000004).
EPSILON = 1e-9


class PredictedVerdict(BaseModel):
    """A verdict predicted from evidence alone.

    In `integrity` mode only `integrity` is populated; in `full` mode the whole
    property cascade is, and integrity is derived exactly as `Verdict.integrity`
    derives it - the worst of context coverage, veracity and contextualization.
    """

    model_config = {"arbitrary_types_allowed": True}

    claim_id: int
    mode: str
    integrity: RatingAggregated | None = None
    veracity: RatingAggregated | None = None
    context_coverage: RatingAggregated | None = None
    media_verdicts: list[MediumVerdict] = []

    @property
    def contextualization(self) -> Rating | None:
        ratings = [mv.contextualization for mv in self.media_verdicts]
        return min(ratings, key=lambda r: r.score) if ratings else None

    @property
    def authenticity(self) -> Rating | None:
        ratings = [mv.authenticity for mv in self.media_verdicts]
        return min(ratings, key=lambda r: r.score) if ratings else None

    @property
    def effective_integrity(self) -> Rating | None:
        """The integrity rating, predicted directly or derived from the cascade."""
        if self.integrity is not None:
            return self.integrity
        candidates = [r for r in (self.context_coverage, self.veracity, self.contextualization)
                      if r is not None]
        return min(candidates, key=lambda r: r.score) if candidates else None

    def get_medium_verdict(self, reference: str) -> MediumVerdict | None:
        for mv in self.media_verdicts:
            if mv.reference == reference:
                return mv
        return None

    @property
    def n_ratings(self) -> int:
        rating = self.effective_integrity
        if isinstance(rating, RatingAggregated):
            return rating.n_ratings
        return 1 if rating is not None else 0


def property_diffs(predicted: PredictedVerdict, gold: Verdict) -> dict[str, float]:
    """Absolute score differences for every property present in **both** verdicts.

    Media verdicts are matched by reference, as `metric.calculate_metrics` does.
    In `integrity` mode this yields exactly one entry; in `full` mode it yields
    one per shared property."""
    diffs: dict[str, float] = {}

    predicted_integrity = predicted.effective_integrity
    gold_integrity = _safe(lambda: gold.integrity)
    if predicted_integrity is not None and gold_integrity is not None:
        diffs["integrity"] = abs(predicted_integrity.score - gold_integrity.score)

    for name in ("veracity", "context_coverage"):
        p, g = getattr(predicted, name, None), getattr(gold, name, None)
        if p is not None and g is not None:
            diffs[name] = abs(p.score - g.score)

    for gold_mv in gold.media_verdicts:
        predicted_mv = predicted.get_medium_verdict(gold_mv.reference)
        if predicted_mv is None:
            continue
        diffs[f"authenticity[{gold_mv.reference}]"] = abs(
            predicted_mv.authenticity.score - gold_mv.authenticity.score)
        diffs[f"contextualization[{gold_mv.reference}]"] = abs(
            predicted_mv.contextualization.score - gold_mv.contextualization.score)

    return diffs


def is_close(predicted: PredictedVerdict, gold: Verdict, threshold: float = None) -> bool:
    """Whether the predicted verdict is sufficiently close to the gold verdict.

    Closeness is the **maximum** absolute score distance over all shared
    properties: every property the two verdicts have in common must agree within
    `threshold`. An empty overlap is never close, because nothing was verified."""
    if threshold is None:
        threshold = default_threshold
    diffs = property_diffs(predicted, gold)
    if not diffs:
        return False
    return max(diffs.values()) <= threshold + EPSILON


def _safe(fn):
    try:
        return fn()
    except (ValueError, AttributeError, TypeError):
        return None
