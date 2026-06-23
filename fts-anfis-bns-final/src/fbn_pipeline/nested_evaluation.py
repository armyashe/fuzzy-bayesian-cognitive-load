from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from .determinism import configure_determinism
from .fold_manifest import FoldManifest, OuterFoldManifest, build_subject_loso_manifest, validate_manifest_against_dataset
from .metrics import per_subject_metrics
from .nested_results import CandidateResult, NestedEvaluationResult, OuterFoldResult
from .structured_config import EvaluationConfig, validate_candidate_ids


class LeakageError(ValueError):
    pass


EXPECTED_CANDIDATE_ERRORS = (
    ValueError,
    RuntimeError,
    FloatingPointError,
    OverflowError,
    np.linalg.LinAlgError,
)


@dataclass(slots=True)
class NestedDataset:
    X: np.ndarray
    y: np.ndarray
    subject_ids: np.ndarray
    sample_ids: list[str]
    sample_indices: np.ndarray | None = None

    def __post_init__(self) -> None:
        n_samples = len(self.y)
        if self.X.shape[0] != n_samples:
            raise ValueError("X and y must have the same sample count")
        if len(self.subject_ids) != n_samples:
            raise ValueError("subject_ids length must match y")
        if len(self.sample_ids) != n_samples:
            raise ValueError("sample_ids length must match y")
        if self.sample_indices is None:
            self.sample_indices = np.arange(n_samples, dtype=int)
        if len(self.sample_indices) != n_samples:
            raise ValueError("sample_indices length must match y")

    def subset(self, indices: np.ndarray) -> "NestedDataset":
        return NestedDataset(
            X=self.X[indices],
            y=self.y[indices],
            subject_ids=self.subject_ids[indices],
            sample_ids=[self.sample_ids[i] for i in indices.tolist()],
            sample_indices=self.sample_indices[indices],
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
    manifest: FoldManifest | None = None,
    allow_partial_outer_folds: bool = False,
) -> NestedEvaluationResult:
    runtime_metadata = configure_determinism(
        evaluation.seed,
        deterministic_threads=evaluation.deterministic_threads,
    ).as_dict()
    if manifest is None:
        manifest = build_subject_loso_manifest(dataset=dataset, evaluation=evaluation)
    validate_manifest_against_dataset(manifest, dataset)
    _validate_outer_fold_subject_coverage(
        dataset=dataset,
        manifest=manifest,
        allow_partial_outer_folds=allow_partial_outer_folds,
    )

    candidate_specs = adapter.candidate_specs()
    validate_candidate_ids(tuple(spec.candidate_id for spec in candidate_specs))
    if not candidate_specs:
        raise ValueError("At least one candidate is required")

    outer_fold_results: list[OuterFoldResult] = []
    aggregated_probabilities = np.full(len(dataset.y), np.nan, dtype=float)
    aggregated_predictions = np.full(len(dataset.y), -1, dtype=int)
    prediction_counts = np.zeros(len(dataset.y), dtype=int)
    sample_index_to_dataset_local = {
        int(sample_index): local_index
        for local_index, sample_index in enumerate(dataset.sample_indices.tolist())
    }

    for outer_fold in manifest.outer_folds:
        if hasattr(adapter, "set_outer_fold_context"):
            adapter.set_outer_fold_context(outer_fold)
        outer_test_indices = np.array(
            [sample_index_to_dataset_local[int(i)] for i in outer_fold.outer_test_indices],
            dtype=int,
        )
        outer_train_indices = np.array(
            [sample_index_to_dataset_local[int(i)] for i in outer_fold.outer_train_indices],
            dtype=int,
        )
        outer_train_data = dataset.subset(outer_train_indices)
        outer_test_data = dataset.subset(outer_test_indices)
        candidate_results = _evaluate_candidates(
            outer_train_data=outer_train_data,
            outer_test_subject=outer_fold.outer_test_subject,
            adapter=adapter,
            evaluation=evaluation,
            outer_fold_manifest=outer_fold,
        )
        selected_candidate = _select_candidate(candidate_results)

        outer_probabilities = np.asarray(
            adapter.fit_predict_outer_fold(
                outer_train_data,
                outer_test_data,
                selected_candidate["spec"],
                seed=outer_fold.seed,
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
                outer_fold_index=outer_fold.outer_fold_index,
                outer_test_subject=outer_fold.outer_test_subject,
                outer_training_subjects=list(outer_fold.outer_train_subjects),
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

    covered_indices = _validate_outer_prediction_coverage(
        aggregated_probabilities=aggregated_probabilities,
        prediction_counts=prediction_counts,
        allow_partial_outer_folds=allow_partial_outer_folds,
    )
    aggregated_sample_ids = [dataset.sample_ids[i] for i in covered_indices.tolist()]
    aggregated_subject_ids = dataset.subject_ids[covered_indices].astype(str).tolist()
    aggregated_labels = dataset.y[covered_indices].copy()
    aggregated_probabilities = aggregated_probabilities[covered_indices]
    aggregated_predictions = aggregated_predictions[covered_indices]

    overall_metrics = _compute_metrics_from_predictions(
        aggregated_labels,
        aggregated_probabilities,
        aggregated_predictions,
    )
    result = NestedEvaluationResult(
        method_name=adapter.method_name,
        outer_fold_results=outer_fold_results,
        aggregated_sample_ids=aggregated_sample_ids,
        aggregated_subject_ids=aggregated_subject_ids,
        aggregated_labels=aggregated_labels,
        aggregated_probabilities=aggregated_probabilities,
        aggregated_predictions=aggregated_predictions,
        overall_metrics=overall_metrics,
        per_subject_metrics=per_subject_metrics(
            aggregated_labels,
            aggregated_predictions,
            aggregated_probabilities,
            dataset.subject_ids[covered_indices],
        ),
        evaluation_protocol="outer LOSO + inner GroupKFold",
        runtime_metadata={
            **runtime_metadata,
            "manifest_hash": manifest.manifest_hash,
            "dataset_order_hash": manifest.dataset_order_hash,
        },
    )
    return result


def _validate_outer_fold_subject_coverage(
    *,
    dataset: NestedDataset,
    manifest: FoldManifest,
    allow_partial_outer_folds: bool,
) -> None:
    expected_subjects = set(dataset.subject_ids.astype(str).tolist())
    seen_counts: dict[str, int] = {}
    for outer_fold in manifest.outer_folds:
        subject = str(outer_fold.outer_test_subject)
        seen_counts[subject] = seen_counts.get(subject, 0) + 1

    duplicate_subjects = sorted(subject for subject, count in seen_counts.items() if count > 1)
    if duplicate_subjects:
        raise LeakageError(f"Outer-test subjects must be unique; duplicates: {duplicate_subjects}")

    seen_subjects = set(seen_counts)
    missing_subjects = sorted(expected_subjects - seen_subjects)
    unexpected_subjects = sorted(seen_subjects - expected_subjects)
    if unexpected_subjects:
        raise LeakageError(f"Manifest outer-test subjects are not present in the dataset: {unexpected_subjects}")
    if missing_subjects and not allow_partial_outer_folds:
        raise LeakageError(
            "Publication coverage requires every subject exactly once as the outer test fold; "
            f"missing subjects: {missing_subjects}"
        )


def _validate_outer_prediction_coverage(
    *,
    aggregated_probabilities: np.ndarray,
    prediction_counts: np.ndarray,
    allow_partial_outer_folds: bool,
) -> np.ndarray:
    covered_indices = np.flatnonzero(prediction_counts > 0)
    if covered_indices.size == 0:
        raise LeakageError("The supplied manifest produced zero outer predictions")
    if np.any(np.isnan(aggregated_probabilities[covered_indices])):
        raise LeakageError("A covered outer sample received zero predictions")
    duplicate_indices = np.flatnonzero(prediction_counts > 1)
    if duplicate_indices.size > 0:
        raise LeakageError("Each outer sample must receive exactly one prediction")

    missing_indices = np.flatnonzero(prediction_counts == 0)
    if missing_indices.size > 0 and not allow_partial_outer_folds:
        raise LeakageError(
            "Publication coverage requires exactly one outer prediction for every sample; "
            f"missing predictions for {int(missing_indices.size)} samples"
        )
    return covered_indices


def _evaluate_candidates(
    *,
    outer_train_data: NestedDataset,
    outer_test_subject: str,
    adapter: NestedMethodAdapter,
    evaluation: EvaluationConfig,
    outer_fold_manifest: OuterFoldManifest,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    sample_index_to_local = {
        int(sample_index): local_index
        for local_index, sample_index in enumerate(outer_train_data.sample_indices.tolist())
    }

    for candidate_rank, spec in enumerate(adapter.candidate_specs()):
        oof_probabilities = np.full(len(outer_train_data.y), np.nan, dtype=float)
        prediction_counts = np.zeros(len(outer_train_data.y), dtype=int)
        failure_reason: str | None = None
        failed_inner_fold_index: int | None = None

        try:
            for inner_fold in outer_fold_manifest.inner_folds:
                if hasattr(adapter, "set_inner_fold_context"):
                    adapter.set_inner_fold_context(
                        outer_fold_manifest.outer_fold_index,
                        inner_fold.inner_fold_index,
                    )
                tr_idx = np.array([sample_index_to_local[int(i)] for i in inner_fold.train_indices], dtype=int)
                va_idx = np.array([sample_index_to_local[int(i)] for i in inner_fold.valid_indices], dtype=int)
                train_data = outer_train_data.subset(tr_idx)
                valid_data = outer_train_data.subset(va_idx)
                assert_no_subject_overlap(train_data.subject_ids, valid_data.subject_ids, "inner split")
                assert_no_subject_overlap(valid_data.subject_ids, [outer_test_subject], "outer-vs-inner validation")
                assert_no_subject_overlap(train_data.subject_ids, [outer_test_subject], "outer-vs-inner train")

                valid_probabilities = np.asarray(
                    adapter.fit_predict_inner_fold(
                        train_data,
                        valid_data,
                        spec,
                        seed=outer_fold_manifest.seed + inner_fold.inner_fold_index + candidate_rank,
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
        except EXPECTED_CANDIDATE_ERRORS as exc:
            failure_reason = f"{type(exc).__name__}: {exc}"
            failed_inner_fold_index = inner_fold.inner_fold_index if "inner_fold" in locals() else None

        if failure_reason is not None:
            result = CandidateResult(
                method_name=adapter.method_name,
                candidate_id=spec.candidate_id,
                hyperparameters=dict(spec.hyperparameters),
                inner_oof_probabilities=np.array([], dtype=float),
                inner_oof_predictions=np.array([], dtype=int),
                inner_oof_metrics={"bal_acc": float("nan"), "auc": float("nan")},
                selected_threshold=float("nan"),
                model_complexity=float(spec.complexity),
                status="failed",
                failure_reason=failure_reason,
                outer_fold_index=outer_fold_manifest.outer_fold_index,
                failed_inner_fold_index=failed_inner_fold_index,
            )
            results.append(
                {"spec": spec, "threshold": float("nan"), "metrics": result.inner_oof_metrics, "result": result}
            )
            continue

        threshold = _select_threshold(outer_train_data.y, oof_probabilities, evaluation.threshold_grid)
        ranking_metrics = _compute_metrics_from_predictions(
            outer_train_data.y,
            oof_probabilities,
            (oof_probabilities >= threshold).astype(int),
        )
        result = CandidateResult(
            method_name=adapter.method_name,
            candidate_id=spec.candidate_id,
            hyperparameters=dict(spec.hyperparameters),
            inner_oof_probabilities=oof_probabilities,
            inner_oof_predictions=(oof_probabilities >= threshold).astype(int),
            inner_oof_metrics=ranking_metrics,
            selected_threshold=float(threshold),
            model_complexity=float(spec.complexity),
            status="success",
            failure_reason=None,
            outer_fold_index=outer_fold_manifest.outer_fold_index,
            failed_inner_fold_index=None,
        )
        if len(result.inner_oof_probabilities) != len(outer_train_data.y):
            raise LeakageError("Candidate result length does not match outer-training sample count")
        results.append({"spec": spec, "threshold": threshold, "metrics": ranking_metrics, "result": result})

    return results


def _select_candidate(candidate_results: list[dict[str, object]]) -> dict[str, object]:
    valid_results = [item for item in candidate_results if item["result"].status == "success"]
    if not valid_results:
        reasons = [item["result"].failure_reason for item in candidate_results]
        raise RuntimeError(f"No valid candidates remain for this outer fold. Failures: {reasons}")

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

    return sorted(valid_results, key=candidate_key)[0]


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
