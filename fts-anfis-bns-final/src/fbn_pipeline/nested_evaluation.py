from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .determinism import configure_determinism
from .metrics import per_subject_metrics
from .nested_results import CandidateResult, NestedEvaluationResult, OuterFoldResult
from .structured_config import EvaluationConfig, validate_candidate_ids


class LeakageError(ValueError):
    pass


@dataclass(slots=True)
class NestedDataset:
    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    sample_ids: list[str]

    def __post_init__(self) -> None:
        n_samples = len(self.y)
        if self.X.shape[0] != n_samples:
            raise ValueError("X and y must have the same sample count")
        if len(self.subject_ids) != n_samples:
            raise ValueError("subject_ids length must match y")
        if len(self.sample_ids) != n_samples:
            raise ValueError("sample_ids length must match y")

    def subset(self, indices: np.ndarray) -> "NestedDataset":
        return NestedDataset(
            X=self.X[indices],
            y=self.y[indices],
            subject_ids=self.subject_ids[indices],
            sample_ids=[self.sample_ids[i] for i in indices.tolist()],
        )


@dataclass(slots=True)
class CandidateSpec:
    candidate_id: str
    hyperparameters: dict[str, object]
    complexity: float


class NestedMethodAdapter(Protocol):
    method_name: str

    def candidate_specs(self) -> list[CandidateSpec]:
        ...

    def fit_predict_inner_fold(
        self,
        train_data: NestedDataset,
        valid_data: NestedDataset,
        candidate: CandidateSpec,
        *,
        seed: int,
    ) -> np.ndarray:
        ...

    def fit_predict_outer_fold(
        self,
        train_data: NestedDataset,
        test_data: NestedDataset,
        candidate: CandidateSpec,
        *,
        seed: int,
    ) -> np.ndarray:
        ...


def assert_no_subject_overlap(
    train_subjects: list[str] | np.ndarray,
    other_subjects: list[str] | np.ndarray,
    context: str,
) -> None:
    overlap = set(np.asarray(train_subjects).tolist()) & set(np.asarray(other_subjects).tolist())
    if overlap:
        raise LeakageError(f"Subject overlap detected in {context}: {sorted(overlap)}")


def run_nested_subject_loso(
    *,
    dataset: NestedDataset,
    adapter: NestedMethodAdapter,
    evaluation: EvaluationConfig,
) -> NestedEvaluationResult:
    runtime_metadata = configure_determinism(
        evaluation.seed,
        deterministic_threads=evaluation.deterministic_threads,
    )
    unique_subjects = sorted(np.unique(dataset.subject_ids).tolist())
    if len(unique_subjects) < 2:
        raise ValueError("At least two subject groups are required")

    candidate_specs = adapter.candidate_specs()
    validate_candidate_ids(tuple(spec.candidate_id for spec in candidate_specs))
    if not candidate_specs:
        raise ValueError("At least one candidate is required")

    outer_fold_results: list[OuterFoldResult] = []
    aggregated_probabilities = np.full(len(dataset.y), np.nan, dtype=float)
    aggregated_predictions = np.full(len(dataset.y), -1, dtype=int)
    prediction_counts = np.zeros(len(dataset.y), dtype=int)

    for outer_fold_index, outer_subject in enumerate(unique_subjects):
        outer_test_mask = dataset.subject_ids == outer_subject
        outer_train_mask = ~outer_test_mask
        outer_test_indices = np.flatnonzero(outer_test_mask)
        outer_train_indices = np.flatnonzero(outer_train_mask)
        outer_train_subjects = dataset.subject_ids[outer_train_indices]
        outer_test_subjects = dataset.subject_ids[outer_test_indices]

        assert_no_subject_overlap(outer_train_subjects, outer_test_subjects, "outer split")
        evaluation.validate_available_groups(sorted(np.unique(outer_train_subjects).tolist()))

        outer_train_data = dataset.subset(outer_train_indices)
        outer_test_data = dataset.subset(outer_test_indices)
        candidate_results = _evaluate_candidates(
            outer_train_data=outer_train_data,
            outer_test_subject=outer_subject,
            adapter=adapter,
            evaluation=evaluation,
            outer_fold_index=outer_fold_index,
        )
        selected_candidate = _select_candidate(candidate_results)

        outer_probabilities = np.asarray(
            adapter.fit_predict_outer_fold(
                outer_train_data,
                outer_test_data,
                selected_candidate["spec"],
                seed=evaluation.seed + outer_fold_index,
            ),
            dtype=float,
        )
        if len(outer_probabilities) != len(outer_test_indices):
            raise LeakageError("Outer prediction length does not match outer-test samples")

        outer_predictions = (outer_probabilities >= selected_candidate["threshold"]).astype(int)
        aggregated_probabilities[outer_test_indices] = outer_probabilities
        aggregated_predictions[outer_test_indices] = outer_predictions
        prediction_counts[outer_test_indices] += 1

        outer_fold_results.append(
            OuterFoldResult(
                outer_fold_index=outer_fold_index,
                outer_test_subject=outer_subject,
                outer_training_subjects=sorted(np.unique(outer_train_subjects).tolist()),
                selected_candidate_id=selected_candidate["spec"].candidate_id,
                selected_hyperparameters=dict(selected_candidate["spec"].hyperparameters),
                selected_threshold=float(selected_candidate["threshold"]),
                inner_selection_metrics=dict(selected_candidate["metrics"]),
                outer_test_sample_ids=list(outer_test_data.sample_ids),
                outer_test_true_labels=outer_test_data.y.copy(),
                outer_test_probabilities=outer_probabilities,
                outer_test_predictions=outer_predictions,
                seed=evaluation.seed,
                configuration_hash=_configuration_hash(evaluation, selected_candidate["spec"]),
                candidate_results=[item["result"] for item in candidate_results],
            )
        )

    if np.any(np.isnan(aggregated_probabilities)):
        raise LeakageError("An outer sample received zero predictions")
    if np.any(prediction_counts != 1):
        raise LeakageError("Each outer sample must receive exactly one prediction")
    if dataset.sample_ids != [dataset.sample_ids[i] for i in range(len(dataset.sample_ids))]:
        raise LeakageError("Sample ordering changed silently")

    overall_metrics = _compute_metrics_from_predictions(
        dataset.y,
        aggregated_probabilities,
        aggregated_predictions,
    )
    subject_rows = per_subject_metrics(
        dataset.y,
        aggregated_predictions,
        aggregated_probabilities,
        dataset.subject_ids,
    )
    return NestedEvaluationResult(
        method_name=adapter.method_name,
        outer_fold_results=outer_fold_results,
        aggregated_sample_ids=list(dataset.sample_ids),
        aggregated_subject_ids=dataset.subject_ids.tolist(),
        aggregated_labels=dataset.y.copy(),
        aggregated_probabilities=aggregated_probabilities,
        aggregated_predictions=aggregated_predictions,
        overall_metrics=overall_metrics,
        per_subject_metrics=subject_rows,
        evaluation_protocol="outer LOSO + inner GroupKFold",
        runtime_metadata=runtime_metadata.as_dict(),
    )


def _evaluate_candidates(
    *,
    outer_train_data: NestedDataset,
    outer_test_subject: str,
    adapter: NestedMethodAdapter,
    evaluation: EvaluationConfig,
    outer_fold_index: int,
) -> list[dict[str, object]]:
    gkf = GroupKFold(n_splits=evaluation.inner_folds)
    results: list[dict[str, object]] = []
    split_groups = outer_train_data.subject_ids

    for candidate_rank, spec in enumerate(adapter.candidate_specs()):
        oof_probabilities = np.full(len(outer_train_data.y), np.nan, dtype=float)
        prediction_counts = np.zeros(len(outer_train_data.y), dtype=int)

        for inner_fold_index, (tr_idx, va_idx) in enumerate(
            gkf.split(outer_train_data.X, outer_train_data.y, split_groups)
        ):
            train_data = outer_train_data.subset(np.asarray(tr_idx))
            valid_data = outer_train_data.subset(np.asarray(va_idx))
            assert_no_subject_overlap(train_data.subject_ids, valid_data.subject_ids, "inner split")
            assert_no_subject_overlap(valid_data.subject_ids, [outer_test_subject], "outer-vs-inner validation")
            assert_no_subject_overlap(train_data.subject_ids, [outer_test_subject], "outer-vs-inner train")

            valid_probabilities = np.asarray(
                adapter.fit_predict_inner_fold(
                    train_data,
                    valid_data,
                    spec,
                    seed=evaluation.seed + outer_fold_index * 100 + inner_fold_index + candidate_rank,
                ),
                dtype=float,
            )
            if len(valid_probabilities) != len(va_idx):
                raise LeakageError("Inner validation prediction length does not match validation samples")
            oof_probabilities[va_idx] = valid_probabilities
            prediction_counts[va_idx] += 1

        if np.any(np.isnan(oof_probabilities)):
            raise LeakageError("An outer-training sample received zero inner OOF predictions")
        if np.any(prediction_counts != 1):
            raise LeakageError("An outer-training sample received more than one inner OOF prediction")

        ranking_metrics = _compute_metrics_from_predictions(
            outer_train_data.y,
            oof_probabilities,
            (oof_probabilities >= 0.5).astype(int),
        )
        threshold = _select_threshold(outer_train_data.y, oof_probabilities, evaluation.threshold_grid)
        result = CandidateResult(
            method_name=adapter.method_name,
            candidate_id=spec.candidate_id,
            hyperparameters=dict(spec.hyperparameters),
            inner_oof_probabilities=oof_probabilities,
            inner_oof_predictions=(oof_probabilities >= threshold).astype(int),
            inner_oof_metrics=ranking_metrics,
            selected_threshold=float(threshold),
            model_complexity=float(spec.complexity),
            status="ok",
            failure_reason=None,
        )
        if len(result.inner_oof_probabilities) != len(outer_train_data.y):
            raise LeakageError("Candidate result length does not match outer-training sample count")
        results.append({"spec": spec, "threshold": threshold, "metrics": ranking_metrics, "result": result})

    return results


def _select_candidate(candidate_results: list[dict[str, object]]) -> dict[str, object]:
    def candidate_key(item: dict[str, object]) -> tuple[float, float, float, str]:
        metrics = item["metrics"]
        spec = item["spec"]
        auc = float(metrics["auc"])
        if np.isnan(auc):
            auc = float("-inf")
        return (
            -float(metrics["bal_acc"]),
            -auc,
            float(spec.complexity),
            str(spec.candidate_id),
        )

    return sorted(candidate_results, key=candidate_key)[0]


def _select_threshold(y_true: np.ndarray, probabilities: np.ndarray, grid: tuple[float, ...]) -> float:
    best_threshold = float(grid[0])
    best_score = float("-inf")
    for threshold in grid:
        score = balanced_accuracy_score(y_true, (probabilities >= threshold).astype(int))
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def _configuration_hash(evaluation: EvaluationConfig, spec: CandidateSpec) -> str:
    payload = json.dumps(
        {
            "evaluation": evaluation.as_dict(),
            "candidate_id": spec.candidate_id,
            "hyperparameters": spec.hyperparameters,
            "complexity": spec.complexity,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _compute_metrics_from_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    acc = round(float(accuracy_score(y_true, predictions)), 4)
    prec = round(float(precision_score(y_true, predictions, zero_division=0)), 4)
    rec = round(float(recall_score(y_true, predictions, zero_division=0)), 4)
    f1 = round(float(f1_score(y_true, predictions, zero_division=0)), 4)
    bal_acc = round(float(balanced_accuracy_score(y_true, predictions)), 4)
    macro_f1 = round(float(f1_score(y_true, predictions, average="macro", zero_division=0)), 4)
    try:
        auc = round(float(roc_auc_score(y_true, probabilities)), 4)
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "acc": acc,
        "bal_acc": bal_acc,
        "macro_f1": macro_f1,
    }
