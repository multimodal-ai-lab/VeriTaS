"""Evaluation metrics for fact-checking baselines."""

from .types import LabelScheme, DEFAULT_LABEL_SCHEME, LABEL_SCHEME_3, LABEL_SCHEME_7, LABELS, LABELS_3

# Backwards-compatible mapping (3-class)
VERDICT_TO_NUMERIC = LABEL_SCHEME_3.verdict_to_numeric

# Mapping from 7-class labels to coarsened 3-class labels.
# Only the "certain" and "rather certain" levels map to their respective bin;
# everything else (rather uncertain + unknown) maps to Unknown.
COARSEN_7_TO_3 = {
    "Intact (certain)": "Intact",
    "Intact (rather certain)": "Intact",
    "Intact (rather uncertain)": "Unknown",
    "Unknown": "Unknown",
    "Compromised (rather uncertain)": "Unknown",
    "Compromised (rather certain)": "Compromised",
    "Compromised (certain)": "Compromised",
}


def coarsen_label(label: str) -> str:
    """Coarsen a 7-class label to its 3-class equivalent.

    Mapping:
        Intact (certain | rather certain)         -> Intact
        Compromised (certain | rather certain)     -> Compromised
        everything else                            -> Unknown

    If the label is already 3-class it is returned unchanged.
    """
    return COARSEN_7_TO_3.get(label, label)


def classify_integrity(score: float, scheme: LabelScheme | None = None) -> str:
    """
    Classify integrity score into categories.

    Args:
        score: Integrity score from -1.0 to 1.0.
        scheme: Label scheme to use. Defaults to 3-class.

    Returns:
        Classification label string.
    """
    if scheme is None or scheme.name == "3-class":
        # 3-class: thresholds at +/- 0.33
        if score >= 0.33:
            return "Intact"
        elif score > -0.33:
            return "Unknown"
        else:
            return "Compromised"
    else:
        # 7-class: thresholds matching discretize_7_bins from veritas
        if score < -5 / 6:
            return "Compromised (certain)"
        elif score < -3 / 6:
            return "Compromised (rather certain)"
        elif score < -1 / 6:
            return "Compromised (rather uncertain)"
        elif score > 5 / 6:
            return "Intact (certain)"
        elif score > 3 / 6:
            return "Intact (rather certain)"
        elif score > 1 / 6:
            return "Intact (rather uncertain)"
        else:
            return "Unknown"


def compute_metrics(y_true: list, y_pred: list, labels: list | None = None) -> dict:
    """
    Compute classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        labels: List of label names. Defaults to 3-class LABELS.

    Returns:
        Dictionary containing accuracy, precision, recall, F1, and confusion matrix.
    """
    if labels is None:
        labels = LABELS

    if not y_true:
        return {}

    # Accuracy
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true)

    # Per-class precision, recall, F1
    metrics_per_class = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics_per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(1 for t in y_true if t == label)
        }

    # Macro averages
    macro_precision = sum(m["precision"] for m in metrics_per_class.values()) / len(labels)
    macro_recall = sum(m["recall"] for m in metrics_per_class.values()) / len(labels)
    macro_f1 = sum(m["f1"] for m in metrics_per_class.values()) / len(labels)

    # Weighted averages
    total_support = len(y_true)
    weighted_precision = sum(m["precision"] * m["support"] for m in metrics_per_class.values()) / total_support
    weighted_recall = sum(m["recall"] * m["support"] for m in metrics_per_class.values()) / total_support
    weighted_f1 = sum(m["f1"] * m["support"] for m in metrics_per_class.values()) / total_support

    # Confusion matrix
    matrix = [[sum(1 for t, p in zip(y_true, y_pred) if t == tl and p == pl) for pl in labels] for tl in labels]

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_precision": round(weighted_precision, 4),
        "weighted_recall": round(weighted_recall, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": metrics_per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": matrix
        }
    }


def compute_coarsened_metrics(y_true: list[str], y_pred: list[str]) -> dict | None:
    """Coarsen 7-class labels to 3-class and compute accuracy + confusion matrix.

    Returns None when the input labels are already 3-class (nothing to coarsen).
    """
    # Only meaningful when we are in 7-class mode
    if not y_true:
        return None

    # Check if any label is actually 7-class
    all_labels = set(y_true) | set(y_pred)
    if all_labels <= set(LABELS_3):
        return None

    y_true_c = [coarsen_label(l) for l in y_true]
    y_pred_c = [coarsen_label(l) for l in y_pred]

    return compute_metrics(y_true_c, y_pred_c, labels=LABELS_3)


def compute_regression_metrics(
    gt_integrity: list[float],
    verdicts: list[str],
    scheme: LabelScheme | None = None,
) -> dict:
    """
    Compute regression metrics (MSE, MAE) between ground truth integrity scores and verdicts.

    Args:
        gt_integrity: List of ground truth integrity scores (-1.0 to 1.0).
        verdicts: List of predicted verdict labels.
        scheme: Label scheme for verdict-to-numeric mapping. Defaults to 3-class.

    Returns:
        Dictionary with MSE and MAE metrics.
    """
    if scheme is None:
        scheme = DEFAULT_LABEL_SCHEME

    verdict_to_num = scheme.verdict_to_numeric

    if not gt_integrity or not verdicts:
        return {}

    # Filter to only valid pairs
    valid_pairs = []
    for gt, verdict in zip(gt_integrity, verdicts):
        if gt is not None and verdict in verdict_to_num:
            try:
                gt_val = float(gt)
                valid_pairs.append((gt_val, verdict_to_num[verdict]))
            except (ValueError, TypeError):
                continue

    if not valid_pairs:
        return {}

    n = len(valid_pairs)
    mse = sum((gt - pred) ** 2 for gt, pred in valid_pairs) / n
    mae = sum(abs(gt - pred) for gt, pred in valid_pairs) / n

    return {
        "mse": round(mse, 4),
        "mae": round(mae, 4),
        "n_samples": n,
    }


def _print_confusion_matrix(cm: dict, title: str = "Confusion Matrix"):
    """Print a single confusion matrix block."""
    labels = cm["labels"]
    col_width = max(len(l) for l in labels) + 2
    row_label_width = col_width

    print(f"\n  {title} (rows=true, cols=pred):")
    header = " " * (row_label_width + 4) + "".join(f"{l:>{col_width}s}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row = cm["matrix"][i]
        row_str = "".join(f"{v:>{col_width}d}" for v in row)
        print(f"    {row_label:<{row_label_width}s}{row_str}")


def print_metrics(metrics: dict):
    """Print metrics summary to console."""
    if not metrics:
        return

    print(f"\n  === Classification Metrics ===")
    print(f"  Accuracy:         {metrics['accuracy']:.2%}")

    # Show coarsened 3-bin accuracy when running in 7-class mode
    coarsened = metrics.get("coarsened_3class")
    if coarsened:
        print(f"  Accuracy (3-bin): {coarsened['accuracy']:.2%}")

    print(f"  Macro F1:         {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:      {metrics['weighted_f1']:.4f}")

    # Print regression metrics if available
    if "mse" in metrics:
        print(f"\n  === Regression Metrics ===")
        print(f"  MSE:              {metrics['mse']:.4f}")
        print(f"  MAE:              {metrics['mae']:.4f}")

    print(f"\n  Per-class F1:")
    for label, m in metrics["per_class"].items():
        print(f"    {label:30s}: F1={m['f1']:.4f} (P={m['precision']:.4f}, R={m['recall']:.4f}, n={m['support']})")

    _print_confusion_matrix(metrics["confusion_matrix"])

    # Show coarsened 3-bin confusion matrix when running in 7-class mode
    if coarsened:
        _print_confusion_matrix(coarsened["confusion_matrix"], title="Confusion Matrix (coarsened 3-bin)")
