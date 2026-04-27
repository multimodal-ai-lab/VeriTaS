"""Metric functions to measure model performance on the VeriTaS benchmark."""
import numpy as np

from veritas.common import Rating, Verdict
from veritas.common.verdict import MediumVerdict


def mse(predictions: list[Rating], targets: list[Rating]) -> float:
    """Returns the Mean Squared Error (MSE) for the given predictions and ground truth.
    Penalizes large errors more heavily than small ones."""
    _validate_inputs(predictions, targets)
    prediction_values = np.array([rating.score for rating in predictions])
    target_values = np.array([rating.score for rating in targets])
    return float(np.mean((prediction_values - target_values) ** 2))


def mae(predictions: list[Rating], targets: list[Rating]) -> float:
    """Returns the Mean Absolute Error (MAE) for the given predictions and ground truth.
    Penalizes errors linear in their magnitude."""
    _validate_inputs(predictions, targets)
    prediction_values = np.array([rating.score for rating in predictions])
    target_values = np.array([rating.score for rating in targets])
    return float(np.mean(np.abs(prediction_values - target_values)))


def acc_3bin(predictions: list[Rating], targets: list[Rating]) -> float:
    """Returns the 3-bin Accuracy (Acc) for the given predictions and ground truth
    by mapping the rating scores to three equally sized bins."""
    _validate_inputs(predictions, targets)
    prediction_categories = np.array([rating.as_3_bin() for rating in predictions])
    target_categories = np.array([rating.as_3_bin() for rating in targets])
    correct = prediction_categories == target_categories
    return float(np.mean(correct))


def acc_7bin(predictions: list[Rating], targets: list[Rating]) -> float:
    """Returns the 7-bin Accuracy (Acc) for the given predictions and ground truth
    by mapping the rating scores to seven bins."""
    _validate_inputs(predictions, targets)
    prediction_categories = np.array([rating.as_7_bin() for rating in predictions])
    target_categories = np.array([rating.as_7_bin() for rating in targets])
    correct = prediction_categories == target_categories
    return float(np.mean(correct))


def calculate_metrics(predictions: list[Verdict], targets: list[Verdict]) -> dict[str, dict[str, float]]:
    """Convenience function that computes MSE, MAE, 3-bin Accuracy, and 7-bin Accuracy
    for all properties and their total (exc. integrity, which is reported separately)."""
    assert len(predictions) == len(targets), "Predictions and targets must have the same length"

    results = dict()

    # Extract media verdicts matched by reference: only include media present in both
    p_authenticity: list[Rating | None] = []
    t_authenticity: list[Rating | None] = []
    p_contextualization: list[Rating | None] = []
    t_contextualization: list[Rating | None] = []

    for p, t in zip(predictions, targets):
        # Index target media verdicts by reference for efficient lookup
        t_media_by_ref: dict[str, MediumVerdict] = {mv.reference: mv for mv in t.media_verdicts}
        for p_mv in p.media_verdicts:
            t_mv = t_media_by_ref.get(p_mv.reference)
            if t_mv is None:
                # Media dismissed/missing in target — skip
                continue
            p_authenticity.append(p_mv.authenticity)
            t_authenticity.append(t_mv.authenticity)
            p_contextualization.append(p_mv.contextualization)
            t_contextualization.append(t_mv.contextualization)

    # 1. Media Authenticity
    results["authenticity"] = calculate_metrics_for_property(
        p_authenticity, t_authenticity
    )

    # 2. Media Contextualization
    results["contextualization"] = calculate_metrics_for_property(
        p_contextualization, t_contextualization
    )

    # 3. Veracity
    p_veracity = [p.veracity for p in predictions]
    t_veracity = [t.veracity for t in targets]
    results["veracity"] = calculate_metrics_for_property(
        p_veracity, t_veracity
    )

    # 4. Context Coverage
    p_context_coverage = [p.context_coverage for p in predictions]
    t_context_coverage = [t.context_coverage for t in targets]
    results["context_coverage"] = calculate_metrics_for_property(
        p_context_coverage, t_context_coverage
    )

    # 5. Integrity (may fail if no decisive property exists)
    p_integrity: list[Rating | None] = []
    t_integrity: list[Rating | None] = []
    for p, t in zip(predictions, targets):
        try:
            p_integrity.append(p.integrity)
        except (ValueError, AttributeError):
            p_integrity.append(None)
        try:
            t_integrity.append(t.integrity)
        except (ValueError, AttributeError):
            t_integrity.append(None)
    results["integrity"] = calculate_metrics_for_property(
        p_integrity, t_integrity
    )

    # 6. Total
    p_all = p_authenticity + p_contextualization + p_veracity + p_context_coverage
    t_all = t_authenticity + t_contextualization + t_veracity + t_context_coverage
    results["total"] = calculate_metrics_for_property(p_all, t_all)

    return results


def calculate_metrics_for_property(predictions: list[Rating | None], targets: list[Rating | None]) -> dict[str, float] | None:
    """Convenience function that computes MSE, MAE, 3-bin Accuracy, and 7-bin Accuracy
    for a single property from Rating lists. Pairs where either value is None are excluded.
    Returns None if no valid pairs remain."""
    filtered = _filter(predictions, targets)
    if not filtered:
        return None
    ps, ts = zip(*filtered)
    return dict(mse=mse(ps, ts),
                mae=mae(ps, ts),
                acc_3bin=acc_3bin(ps, ts),
                acc_7bin=acc_7bin(ps, ts))


def _filter(predictions: list[Rating | None], targets: list[Rating | None]) -> list[tuple[Rating, Rating]]:
    return [(p, t) for p, t in zip(predictions, targets) if t is not None and p is not None]


def _validate_inputs(predictions, targets):
    assert len(predictions) == len(targets)
    assert all(predictions)
    assert all(targets)
