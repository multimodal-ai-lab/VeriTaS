"""The closeness function that decides whether the gold verdict was recovered."""

import pytest

from tests.gold_evidence.conftest import make_gold_verdict, make_rating
from veritas.common.verdict import MediumVerdict
from veritas.gold_evidence.closeness import (
    MODE_FULL,
    MODE_INTEGRITY,
    PredictedVerdict,
    is_close,
    property_diffs,
)


def integrity_prediction(score: float, claim_id: int = 1) -> PredictedVerdict:
    return PredictedVerdict(claim_id=claim_id, mode=MODE_INTEGRITY,
                            integrity=make_rating(score))


# --- Integrity mode --------------------------------------------------------

def test_integrity_mode_compares_exactly_one_property():
    gold = make_gold_verdict(veracity=-1.0, media={"<image:1>": (1.0, -1.0)})
    diffs = property_diffs(integrity_prediction(-1.0), gold)
    assert list(diffs) == ["integrity"]
    assert diffs["integrity"] == pytest.approx(0.0)


def test_integrity_mode_uses_the_gold_compromising_property():
    """Gold integrity is the worst of context coverage, veracity, contextualization."""
    gold = make_gold_verdict(veracity=-1.0, media={"<image:1>": (1.0, 1.0)})
    assert gold.integrity.score == pytest.approx(-1.0)
    assert is_close(integrity_prediction(-1.0), gold, threshold=0.3)
    assert not is_close(integrity_prediction(1.0), gold, threshold=0.3)


@pytest.mark.parametrize("predicted_score, expected", [
    (-1.0, True),         # exact
    (-0.7, True),         # 0.3 away -> at the threshold, inclusive
    (-0.7 - 1e-12, True), # marginally inside, within float tolerance
    (-0.69, False),       # 0.31 away
    (0.0, False),
])
def test_threshold_boundary_is_inclusive(predicted_score, expected):
    gold = make_gold_verdict(veracity=-1.0)
    assert is_close(integrity_prediction(predicted_score), gold, threshold=0.3) is expected


# --- Full mode -------------------------------------------------------------

def full_prediction(*, veracity=None, context_coverage=None, media=None,
                    claim_id: int = 1) -> PredictedVerdict:
    media_verdicts = [
        MediumVerdict(reference=reference,
                      authenticity=make_rating(authenticity),
                      contextualization=make_rating(contextualization))
        for reference, (authenticity, contextualization) in (media or {}).items()
    ]
    return PredictedVerdict(
        claim_id=claim_id,
        mode=MODE_FULL,
        veracity=make_rating(veracity) if veracity is not None else None,
        context_coverage=make_rating(context_coverage) if context_coverage is not None else None,
        media_verdicts=media_verdicts,
    )


def test_full_mode_compares_every_shared_property():
    gold = make_gold_verdict(veracity=1.0, context_coverage=1.0,
                             media={"<image:1>": (1.0, 1.0)})
    predicted = full_prediction(veracity=1.0, context_coverage=1.0,
                                media={"<image:1>": (1.0, 1.0)})
    diffs = property_diffs(predicted, gold)
    assert set(diffs) == {"integrity", "veracity", "context_coverage",
                          "authenticity[<image:1>]", "contextualization[<image:1>]"}
    assert max(diffs.values()) == pytest.approx(0.0)


def test_full_mode_takes_the_maximum_not_the_mean():
    """One badly diverging property rejects the instance, even if the rest agree."""
    gold = make_gold_verdict(veracity=1.0, context_coverage=1.0,
                             media={"<image:1>": (1.0, 1.0)})
    predicted = full_prediction(veracity=1.0, context_coverage=1.0,
                                media={"<image:1>": (-1.0, 1.0)})
    diffs = property_diffs(predicted, gold)
    assert diffs["authenticity[<image:1>]"] == pytest.approx(2.0)
    assert not is_close(predicted, gold, threshold=0.3)


def test_media_are_matched_by_reference():
    gold = make_gold_verdict(veracity=1.0, media={"<image:1>": (1.0, 1.0)})
    predicted = full_prediction(veracity=1.0, media={"<image:2>": (-1.0, -1.0)})
    diffs = property_diffs(predicted, gold)
    assert not any(key.startswith("authenticity") for key in diffs)
    # The unmatched medium does not silently count as agreement either:
    assert "integrity" in diffs


def test_properties_missing_on_one_side_are_skipped():
    gold = make_gold_verdict(veracity=1.0, context_coverage=-1.0)
    predicted = full_prediction(veracity=1.0)  # context coverage not assessed
    diffs = property_diffs(predicted, gold)
    assert "context_coverage" not in diffs
    assert "veracity" in diffs


def test_derived_integrity_matches_the_verdict_definition():
    """`effective_integrity` mirrors `Verdict.integrity`: the worst property."""
    predicted = full_prediction(veracity=0.5, context_coverage=1.0,
                                media={"<image:1>": (1.0, -0.5)})
    assert predicted.effective_integrity.score == pytest.approx(-0.5)


def test_explicit_integrity_wins_over_derivation():
    predicted = PredictedVerdict(claim_id=1, mode=MODE_INTEGRITY,
                                 integrity=make_rating(0.2),
                                 veracity=make_rating(-1.0))
    assert predicted.effective_integrity.score == pytest.approx(0.2)


# --- Degenerate cases ------------------------------------------------------

def test_empty_overlap_is_never_close():
    """Nothing was verified, so nothing may be accepted."""
    gold = make_gold_verdict(veracity=1.0)
    predicted = PredictedVerdict(claim_id=1, mode=MODE_FULL)
    assert property_diffs(predicted, gold) == {}
    assert is_close(predicted, gold, threshold=1.0) is False


def test_gold_without_any_property_yields_no_diffs():
    gold = make_gold_verdict()
    assert property_diffs(integrity_prediction(0.0), gold) == {}
    assert is_close(integrity_prediction(0.0), gold) is False


def test_n_ratings_counts_ensemble_members():
    from veritas.common.annotation import Rating
    from veritas.common.annotation.rating import RatingAggregated

    rating = RatingAggregated(
        individual_ratings=[Rating(score=s, rater=f"m{i}", explanation="e")
                            for i, s in enumerate([1.0, 0.5, 1.0])],
        rater="ensemble")
    predicted = PredictedVerdict(claim_id=1, mode=MODE_INTEGRITY, integrity=rating)
    assert predicted.n_ratings == 3
    assert PredictedVerdict(claim_id=1, mode=MODE_FULL).n_ratings == 0
