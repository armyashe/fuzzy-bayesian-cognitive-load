from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .publication_metrics import compute_pooled_metrics_from_predictions


def compute_bootstrap_confidence_intervals(
    frame: pd.DataFrame,
    *,
    metric_names: tuple[str, ...],
    n_bootstrap: int,
    seed: int,
    method: str,
) -> pd.DataFrame:
    method_frame = frame.loc[frame["method"] == method].copy() if "method" in frame.columns else frame.copy()
    unique_subjects = sorted(method_frame["subject_id"].astype(str).unique().tolist())
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {metric: [] for metric in metric_names}

    for _ in range(int(n_bootstrap)):
        chosen_subjects = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        resampled = pd.concat(
            [method_frame.loc[method_frame["subject_id"].astype(str) == subject] for subject in chosen_subjects],
            ignore_index=True,
        )
        metrics = compute_pooled_metrics_from_predictions(resampled)
        for metric in metric_names:
            value = metrics[metric]
            if not (isinstance(value, float) and math.isnan(value)):
                samples[metric].append(float(value))

    rows: list[dict[str, object]] = []
    for metric in metric_names:
        metric_values = np.asarray(samples[metric], dtype=float)
        rows.append(
            {
                "method": method,
                "metric": metric,
                "lower": float(np.quantile(metric_values, 0.025)),
                "upper": float(np.quantile(metric_values, 0.975)),
                "point_estimate": float(compute_pooled_metrics_from_predictions(method_frame)[metric]),
                "bootstrap_repetitions": int(n_bootstrap),
                "resample_unit": "subject_cluster",
                "interval_method": "percentile",
            }
        )
    return pd.DataFrame(rows)


def run_paired_wilcoxon_by_subject(
    subject_metrics: pd.DataFrame,
    *,
    method_a: str,
    method_b: str,
    metric: str,
) -> dict[str, object]:
    subset = subject_metrics.loc[subject_metrics["method"].isin([method_a, method_b]), ["method", "subject_id", metric]].copy()
    pivot = subset.pivot(index="subject_id", columns="method", values=metric).dropna()
    a_values = pivot[method_a].to_numpy(dtype=float)
    b_values = pivot[method_b].to_numpy(dtype=float)
    differences = a_values - b_values
    zero_count = int(np.sum(np.isclose(differences, 0.0)))
    statistic, p_value = wilcoxon(a_values, b_values, zero_method="pratt", alternative="two-sided", method="auto")

    non_zero = differences[~np.isclose(differences, 0.0)]
    if non_zero.size == 0:
        effect_size = 0.0
    else:
        positive = int(np.sum(non_zero > 0.0))
        negative = int(np.sum(non_zero < 0.0))
        effect_size = float((positive - negative) / non_zero.size)

    return {
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "paired_subject_count": int(len(pivot)),
        "zero_difference_count": zero_count,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_size_rank_biserial": effect_size,
        "zero_method": "pratt",
        "test": "wilcoxon_signed_rank",
    }


def holm_correction(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    total = len(p_values)
    for rank, (index, p_value) in enumerate(indexed, start=1):
        corrected = min(1.0, (total - rank + 1) * float(p_value))
        running_max = max(running_max, corrected)
        adjusted[index] = running_max
    return adjusted
