from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, object]:
    accuracy = float(accuracy_score(y_true, y_pred))
    balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    precision_task = float(precision_score(y_true, y_pred, zero_division=0))
    recall_task = float(recall_score(y_true, y_pred, zero_division=0))
    f1_task = float(f1_score(y_true, y_pred, zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    roc_auc, undefined_reason = _safe_auc(y_true, y_prob)
    return {
        "balanced_accuracy": balanced_accuracy,
        "accuracy": accuracy,
        "precision_task": precision_task,
        "recall_task": recall_task,
        "f1_task": f1_task,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "roc_auc_undefined_reason": undefined_reason,
        "confusion_matrix": cm.astype(int).tolist(),
    }


def compute_pooled_metrics_from_predictions(frame: pd.DataFrame) -> dict[str, object]:
    ordered = frame.sort_values("sample_id").reset_index(drop=True)
    return compute_binary_metrics(
        ordered["true_label"].to_numpy(dtype=int),
        ordered["score_or_probability"].to_numpy(dtype=float),
        ordered["prediction"].to_numpy(dtype=int),
    )


def compute_per_subject_metrics_from_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["method", "subject_id"] if "method" in frame.columns else ["subject_id"]
    for keys, group in frame.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_columns, keys, strict=True))
        metrics = compute_binary_metrics(
            group["true_label"].to_numpy(dtype=int),
            group["score_or_probability"].to_numpy(dtype=float),
            group["prediction"].to_numpy(dtype=int),
        )
        label_counts = group["true_label"].value_counts().to_dict()
        base.update(
            {
                "sample_count": int(len(group)),
                "rest_count": int(label_counts.get(0, 0)),
                "task_count": int(label_counts.get(1, 0)),
                "selected_threshold": float(group["threshold"].iloc[0]) if "threshold" in group.columns else float("nan"),
                "configuration_hash": group["configuration_hash"].iloc[0] if "configuration_hash" in group.columns else "",
                **metrics,
            }
        )
        rows.append(base)
    return pd.DataFrame(rows)


def compute_subject_metric_summary(per_subject_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in ("balanced_accuracy", "accuracy", "precision_task", "recall_task", "f1_task", "macro_f1", "roc_auc")
        if column in per_subject_metrics.columns
    ]
    rows: list[dict[str, object]] = []
    for metric in metric_columns:
        values = per_subject_metrics[metric].dropna().to_numpy(dtype=float)
        if values.size == 0:
            continue
        q1 = float(np.quantile(values, 0.25))
        q3 = float(np.quantile(values, 0.75))
        rows.extend(
            [
                {"metric": metric, "statistic": "mean", "value": float(np.mean(values))},
                {"metric": metric, "statistic": "std", "value": float(np.std(values))},
                {"metric": metric, "statistic": "median", "value": float(np.median(values))},
                {"metric": metric, "statistic": "iqr", "value": float(q3 - q1)},
                {"metric": metric, "statistic": "min", "value": float(np.min(values))},
                {"metric": metric, "statistic": "max", "value": float(np.max(values))},
            ]
        )
    return pd.DataFrame(rows)


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, str | None]:
    unique = np.unique(y_true)
    if unique.size < 2:
        return float("nan"), "single_class_true_labels"
    return float(roc_auc_score(y_true, y_prob)), None
