"""Common utilities for human evaluation statistics scripts."""

from veritas.common.annotation.rating import Rating, RatingAggregated
from veritas.common.verdict import Verdict, MediumVerdict


def value_to_sign(prop: str, value: str | None) -> float:
    """Convert property value to polarity sign."""
    if not value:
        return 0.0
    v = str(value).strip().lower()

    if prop in ["clarity", "medium_clarity"]:
        return +1.0 if v == "clear" else (-1.0 if v == "ambiguous" else 0.0)
    elif prop == "veracity":
        return +1.0 if v == "true" else (-1.0 if v == "false" else 0.0)
    elif prop == "context_coverage":
        return +1.0 if v == "sufficient" else (-1.0 if v == "insufficient" else 0.0)
    elif prop == "intent":
        return +1.0 if v == "legitimate" else (-1.0 if v == "illegitimate" else 0.0)
    elif prop in ["authenticity", "media_authenticity"]:
        return +1.0 if v == "pristine" else (-1.0 if v == "fabricated" else 0.0)
    elif prop in ["contextualization", "media_contextualization"]:
        return +1.0 if v == "correct" else (-1.0 if v == "incorrect" else 0.0)
    return 0.0


def to_tendency(prop: str, value: str | None, confidence: int | None) -> float | None:
    """Convert value + confidence to tendency in [-1, 1]."""
    if value is None:
        return None

    sign = value_to_sign(prop, value)
    magnitude = (confidence / 3.0) if confidence is not None else 0.0
    magnitude = max(0.0, min(1.0, magnitude))

    return sign * magnitude


def to_aggregated(rating: Rating) -> RatingAggregated:
    """Wrap a single Rating into a RatingAggregated."""
    return RatingAggregated(individual_ratings=[rating], rater=rating.rater,
                            explanation=rating.explanation)


def build_human_verdict(
    claim_id: int,
    review_ids: set[int],
    claim_ratings: dict[str, Rating],
    media_ratings: dict[int, dict[str, Rating]],
    auto_verdict: Verdict,
) -> Verdict:
    """Build a Verdict object from human annotations for a single (annotator, claim) pair.

    Media references are taken from the auto_verdict so that they match during metric computation.
    """
    media_verdicts: list[MediumVerdict] = []
    for mv in auto_verdict.media_verdicts:
        ref_id = int(mv.reference.split(":")[-1].rstrip(">"))
        if ref_id not in media_ratings:
            continue
        human_media = media_ratings[ref_id]
        auth = human_media.get("authenticity")
        ctx = human_media.get("contextualization")
        if auth is None or ctx is None:
            continue
        media_verdicts.append(MediumVerdict(
            reference=mv.reference,
            authenticity=to_aggregated(auth),
            contextualization=to_aggregated(ctx),
        ))

    return Verdict(
        claim_id=claim_id,
        review_ids=review_ids,
        media_verdicts=media_verdicts,
        veracity=to_aggregated(claim_ratings["veracity"]) if "veracity" in claim_ratings else None,
        context_coverage=to_aggregated(claim_ratings["context_coverage"]) if "context_coverage" in claim_ratings else None,
        intent=to_aggregated(claim_ratings["intent"]) if "intent" in claim_ratings else None,
    )


def format_metrics(metrics: dict[str, dict[str, float]], key: str,
                   score_fmt: str = ".3f", pct_fmt: str = ".1%") -> list[str]:
    """Format a single property's metrics for table display."""
    m = metrics.get(key)
    if m is None:
        return ["N/A", "N/A", "N/A", "N/A"]
    return [
        f"{m['mse']:{score_fmt}}",
        f"{m['mae']:{score_fmt}}",
        f"{m['acc_3bin']:{pct_fmt}}",
        f"{m['acc_7bin']:{pct_fmt}}",
    ]
