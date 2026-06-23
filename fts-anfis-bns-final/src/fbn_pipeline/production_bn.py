from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.model_selection import GroupKFold

from .nested_evaluation import LeakageError, NestedDataset, assert_no_subject_overlap
from .models.bn_fusion import BNFusion


SignalPredictor = Callable[[NestedDataset, NestedDataset, int], np.ndarray]


@dataclass(slots=True)
class CrossFittedEvidenceResult:
    evidence: np.ndarray
    sample_ids: list[str]
    subject_ids: np.ndarray
    labels: np.ndarray
    coverage_counts: np.ndarray
    rows: list[dict[str, object]]


def generate_cross_fitted_signal_evidence(
    *,
    dataset: NestedDataset,
    n_splits: int,
    seed: int,
    predictor: SignalPredictor,
    metadata: dict[str, object] | None = None,
) -> CrossFittedEvidenceResult:
    unique_subjects = np.unique(dataset.subject_ids.astype(str))
    if len(unique_subjects) < 2:
        raise ValueError("At least two subject groups are required for cross-fitting")
    if n_splits > len(unique_subjects):
        raise ValueError("n_splits cannot exceed the available subject groups")

    splitter = GroupKFold(n_splits=n_splits)
    evidence = np.full((len(dataset.y), 4), np.nan, dtype=float)
    coverage_counts = np.zeros(len(dataset.y), dtype=int)
    row_buffer: list[dict[str, object] | None] = [None] * len(dataset.y)
    shared_metadata = dict(metadata or {})

    for fold_index, (train_idx, valid_idx) in enumerate(
        splitter.split(dataset.X, dataset.y, dataset.subject_ids)
    ):
        train_data = dataset.subset(np.asarray(train_idx, dtype=int))
        valid_data = dataset.subset(np.asarray(valid_idx, dtype=int))
        assert_no_subject_overlap(train_data.subject_ids, valid_data.subject_ids, "bn cross-fit")
        fold_evidence = np.asarray(predictor(train_data, valid_data, seed + fold_index), dtype=float)
        if fold_evidence.shape != (len(valid_data.y), 4):
            raise LeakageError("Cross-fitted signal evidence must have shape (n_valid, 4)")
        if not np.all(np.isfinite(fold_evidence)):
            raise LeakageError("Cross-fitted signal evidence contains non-finite values")

        evidence[valid_idx] = fold_evidence
        coverage_counts[valid_idx] += 1
        for local_offset, global_index in enumerate(valid_idx.tolist()):
            row_buffer[global_index] = {
                "crossfit_fold_index": int(fold_index),
                "sample_id": dataset.sample_ids[global_index],
                "subject_id": str(dataset.subject_ids[global_index]),
                "true_label": int(dataset.y[global_index]),
                "temp_evidence": float(fold_evidence[local_offset, 0]),
                "hr_evidence": float(fold_evidence[local_offset, 1]),
                "gsr_evidence": float(fold_evidence[local_offset, 2]),
                "rr_evidence": float(fold_evidence[local_offset, 3]),
                **shared_metadata,
            }

    if np.any(np.isnan(evidence)):
        raise LeakageError("Each BN-training sample must receive exactly one evidence vector")
    if np.any(coverage_counts != 1):
        raise LeakageError("Cross-fitted BN evidence coverage must be exactly one per sample")

    rows = [row for row in row_buffer if row is not None]
    return CrossFittedEvidenceResult(
        evidence=evidence,
        sample_ids=list(dataset.sample_ids),
        subject_ids=dataset.subject_ids.copy(),
        labels=dataset.y.copy(),
        coverage_counts=coverage_counts,
        rows=rows,
    )


def fit_bn_from_cross_fitted_evidence(
    *,
    evidence_result: CrossFittedEvidenceResult,
    y_true: np.ndarray,
    n_bins: int,
    alpha: float,
    soft_evidence: bool,
    signal_weights: np.ndarray | None = None,
) -> BNFusion:
    if len(y_true) != len(evidence_result.sample_ids):
        raise LeakageError("Cross-fitted evidence length must match y_true")
    if np.any(evidence_result.coverage_counts != 1):
        raise LeakageError("BN fitting requires complete cross-fitted evidence coverage")

    bn = BNFusion(n_bins=n_bins, alpha=alpha, soft=soft_evidence)
    bn.fit(evidence_result.evidence, np.asarray(y_true, dtype=int), signal_weights=signal_weights)
    bn.fit_source_ = "cross_fitted"
    bn.fit_sample_ids_ = list(evidence_result.sample_ids)
    bn.fit_subject_ids_ = evidence_result.subject_ids.astype(str).tolist()
    return bn


def save_cross_fitted_evidence_csv(
    evidence_result: CrossFittedEvidenceResult,
    output_path: Path | str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not evidence_result.rows:
        return
    fieldnames = list(evidence_result.rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(evidence_result.rows)
