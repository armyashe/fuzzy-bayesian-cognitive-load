"""
Evaluation Utilities
====================
Metrics printing, threshold tuning, and configuration display for the
FTS -> ANFIS -> BN cognitive-load classification pipeline.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


def print_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    title: str = "Evaluation",
    compact: bool = True,
) -> None:
    """Print evaluation metrics in the standard 5-metric format."""
    print(f"\n{'=' * 58}")
    print(f"  {title}")
    print(f"{'=' * 58}")
    print(f"  Accuracy          : {accuracy_score(y_true, y_pred):.4f}")

    if compact:
        return

    print(f"  Precision (task)  : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Recall    (task)  : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  F1-score  (task)  : {f1_score(y_true, y_pred, zero_division=0):.4f}")

    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
            print(f"  ROC-AUC           : {auc:.4f}")
        except ValueError:
            print(f"  ROC-AUC           : N/A")

    print(f"  Balanced Accuracy : {balanced_accuracy_score(y_true, y_pred):.4f}")
    print(f"  Macro-F1          : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"  {'':14s} Predicted")
    print(f"  {'':14s} rest  task")
    print(f"  Actual rest  {cm[0, 0]:4d}  {cm[0, 1]:4d}")
    print(f"  Actual task  {cm[1, 0]:4d}  {cm[1, 1]:4d}")

    rest_recall = cm[0, 0] / max(cm[0].sum(), 1)
    task_recall = cm[1, 1] / max(cm[1].sum(), 1)
    print(f"\n  Rest recall : {rest_recall:.4f}")
    print(f"  Task recall : {task_recall:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=["rest", "task"]))


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "balanced_accuracy",
    low: float = 0.05,
    high: float = 0.95,
    n_grid: int = 181,
    return_sweep: bool = False,
) -> tuple:
    """Sweep thresholds to maximise chosen metric on validation probabilities.

    Parameters
    ----------
    metric : str
        One of ``"accuracy"``, ``"balanced_accuracy"``, ``"f1"``, ``"macro_f1"``.
    n_grid : int
        Number of evenly-spaced threshold candidates between *low* and *high*.
    return_sweep : bool
        If True, return ``(best_t, best_score, thresholds, scores)`` so the
        caller can plot the sweep curve.  Default False keeps backward compat.
    """
    _metrics = {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0),
        "macro_f1": lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
    }
    if metric not in _metrics:
        raise ValueError(
            f"metric must be one of {list(_metrics)}, got '{metric}'"
        )
    fn = _metrics[metric]

    thresholds = np.linspace(low, high, n_grid)
    scores: list[float] = []
    best_t, best_score = 0.5, -1.0
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        score = fn(y_true, preds)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_t = float(t)
    if return_sweep:
        return round(best_t, 4), round(best_score, 4), thresholds, np.array(scores)
    return round(best_t, 4), round(best_score, 4)


def per_subject_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    subjects: np.ndarray,
) -> list[dict]:
    """Compute Acc / Balanced-Acc / Macro-F1 / F1 / AUC per subject.

    Parameters
    ----------
    y_true, y_pred : ndarray (N,)
    y_prob         : ndarray (N,)  - posterior probabilities
    subjects       : ndarray (N,)  - subject IDs (str or int)

    Returns
    -------
    rows : list[dict]  with keys  subject, n, acc, bal_acc, macro_f1, f1, auc
    """
    unique = np.unique(subjects)
    rows: list[dict] = []
    for uid in unique:
        mask = subjects == uid
        yt = y_true[mask]
        yp = y_pred[mask]
        yb = y_prob[mask]
        n = int(mask.sum())
        acc = round(float(accuracy_score(yt, yp)), 4)
        bal_acc = round(float(balanced_accuracy_score(yt, yp)), 4)
        macro_f1 = round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4)
        f1 = round(float(f1_score(yt, yp, zero_division=0)), 4)
        try:
            auc = round(float(roc_auc_score(yt, yb)), 4)
        except ValueError:
            auc = float("nan")
        rows.append({"subject": uid, "n": n, "acc": acc,
                      "bal_acc": bal_acc, "macro_f1": macro_f1,
                      "f1": f1, "auc": auc})
    return rows


def print_per_subject_table(rows: list[dict], title: str = "PER-SUBJECT TEST RESULTS") -> None:
    """Pretty-print the per-subject metrics table without Balanced Accuracy."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  {'Subject':<12s} {'N':>6s} {'Acc':>8s} {'MacF1':>8s} {'F1':>8s} {'AUC':>8s}")
    print(f"  {'-' * 56}")

    accs, macro_f1s, f1s, aucs = [], [], [], []

    for r in rows:
        auc_str = f"{r['auc']:>8.4f}" if not (r['auc'] != r['auc']) else f"{'N/A':>8s}"

        print(
            f"  {r['subject']:<12s} {r['n']:>6d} {r['acc']:>8.4f} "
            f"{r['macro_f1']:>8.4f} {r['f1']:>8.4f} {auc_str}"
        )

        accs.append(r["acc"])
        macro_f1s.append(r["macro_f1"])
        f1s.append(r["f1"])

        if r["auc"] == r["auc"]:  # not NaN
            aucs.append(r["auc"])

    print(f"  {'-' * 56}")

    accs_a = np.array(accs)
    macro_f1s_a = np.array(macro_f1s)
    f1s_a = np.array(f1s)
    aucs_a = np.array(aucs) if aucs else np.array([0.0])

    print(
        f"  {'Mean':<12s} {'':>6s} {accs_a.mean():>8.4f} "
        f"{macro_f1s_a.mean():>8.4f} {f1s_a.mean():>8.4f} {aucs_a.mean():>8.4f}"
    )

    print(
        f"  {'Std':<12s} {'':>6s} {accs_a.std():>8.4f} "
        f"{macro_f1s_a.std():>8.4f} {f1s_a.std():>8.4f} {aucs_a.std():>8.4f}"
    )

    print(f"{'=' * 70}\n")


def print_config(config: dict) -> None:
    """Pretty-print the pipeline configuration table."""
    print("\n" + "=" * 58)
    print("  PIPELINE CONFIGURATION")
    print("=" * 58)
    for k, v in config.items():
        print(f"  {k} = {v}")


def compute_oof_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute the standard metrics for out-of-fold probabilities.

    Kept as a named wrapper because callers use the OOF terminology, but the
    implementation is identical to held-out test metric calculation.
    """
    return compute_metrics(y_true, y_prob, threshold)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute the five required metrics plus legacy aliases.

    Returns
    -------
    dict with keys:
        accuracy, precision, recall, f1, auc   (required)
        acc, bal_acc, macro_f1                  (legacy / threshold selection)
    """
    y_pred = (y_prob >= threshold).astype(int)
    acc = round(float(accuracy_score(y_true, y_pred)), 4)
    prec = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
    rec = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
    f1 = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
    macro_f1 = round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4)
    bal_acc = round(float(balanced_accuracy_score(y_true, y_pred)), 4)
    try:
        auc = round(float(roc_auc_score(y_true, y_prob)), 4)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        # legacy
        "acc": acc,
        "bal_acc": bal_acc,
        "macro_f1": macro_f1,
    }
