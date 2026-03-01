"""
Evaluation Utilities
====================
Metrics printing, threshold tuning, and configuration display for the
FTS → ANFIS → BN cognitive-load classification pipeline.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def print_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                  title: str = "Evaluation") -> None:
    """Print accuracy, F1, confusion matrix, and classification report."""
    print(f"\n{'=' * 58}")
    print(f"  {title}")
    print(f"{'=' * 58}")
    print(f"  Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  F1-score : {f1_score(y_true, y_pred):.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"  {'':14s} Predicted")
    print(f"  {'':14s} rest  task")
    print(f"  Actual rest  {cm[0, 0]:4d}  {cm[0, 1]:4d}")
    print(f"  Actual task  {cm[1, 0]:4d}  {cm[1, 1]:4d}")
    print()
    print(classification_report(y_true, y_pred,
                                target_names=["rest", "task"]))


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    low: float = 0.05,
    high: float = 0.95,
    n_grid: int = 181,
) -> tuple[float, float]:
    """Sweep thresholds to maximise chosen metric on validation probabilities.

    Parameters
    ----------
    metric : str
        One of ``"accuracy"``, ``"balanced_accuracy"``, ``"f1"``.
    n_grid : int
        Number of evenly-spaced threshold candidates between *low* and *high*.
    """
    _metrics = {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0),
    }
    if metric not in _metrics:
        raise ValueError(
            f"metric must be one of {list(_metrics)}, got '{metric}'"
        )
    fn = _metrics[metric]

    best_t, best_score = 0.5, -1.0
    for t in np.linspace(low, high, n_grid):
        preds = (y_prob >= t).astype(int)
        score = fn(y_true, preds)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return round(best_t, 4), round(best_score, 4)


def print_config(config: dict) -> None:
    """Pretty-print the pipeline configuration table."""
    print("\n" + "=" * 58)
    print("  PIPELINE CONFIGURATION")
    print("=" * 58)
    for k, v in config.items():
        print(f"  {k:35s}: {v}")
    print("=" * 58 + "\n")
