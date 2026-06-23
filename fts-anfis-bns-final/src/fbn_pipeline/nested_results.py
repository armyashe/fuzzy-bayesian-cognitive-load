from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class CandidateResult:
    method_name: str
    candidate_id: str
    hyperparameters: dict[str, Any]
    inner_oof_probabilities: np.ndarray
    inner_oof_predictions: np.ndarray
    inner_oof_metrics: dict[str, float]
    selected_threshold: float
    model_complexity: float
    status: str
    failure_reason: str | None
    outer_fold_index: int | None = None
    failed_inner_fold_index: int | None = None

    def to_metadata_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("inner_oof_probabilities")
        data.pop("inner_oof_predictions")
        return data


@dataclass(slots=True)
class OuterFoldResult:
    outer_fold_index: int
    outer_test_subject: str
    outer_training_subjects: list[str]
    selected_candidate_id: str
    selected_hyperparameters: dict[str, Any]
    selected_threshold: float
    inner_selection_metrics: dict[str, float]
    outer_test_sample_ids: list[str]
    outer_test_true_labels: np.ndarray
    outer_test_probabilities: np.ndarray
    outer_test_predictions: np.ndarray
    seed: int
    configuration_hash: str
    candidate_results: list[CandidateResult]

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "outer_fold_index": self.outer_fold_index,
            "outer_test_subject": self.outer_test_subject,
            "outer_training_subjects": list(self.outer_training_subjects),
            "selected_candidate_id": self.selected_candidate_id,
            "selected_hyperparameters": dict(self.selected_hyperparameters),
            "selected_threshold": self.selected_threshold,
            "inner_selection_metrics": dict(self.inner_selection_metrics),
            "outer_test_sample_ids": list(self.outer_test_sample_ids),
            "seed": self.seed,
            "configuration_hash": self.configuration_hash,
            "candidate_results": [item.to_metadata_dict() for item in self.candidate_results],
        }


@dataclass(slots=True)
class NestedEvaluationResult:
    method_name: str
    outer_fold_results: list[OuterFoldResult]
    aggregated_sample_ids: list[str]
    aggregated_subject_ids: list[str]
    aggregated_labels: np.ndarray
    aggregated_probabilities: np.ndarray
    aggregated_predictions: np.ndarray
    overall_metrics: dict[str, float]
    per_subject_metrics: list[dict[str, Any]]
    evaluation_protocol: str = "loso+groupkfold"
    runtime_metadata: dict[str, Any] | None = None

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "outer_fold_results": [item.to_metadata_dict() for item in self.outer_fold_results],
            "aggregated_sample_ids": list(self.aggregated_sample_ids),
            "aggregated_subject_ids": list(self.aggregated_subject_ids),
            "overall_metrics": dict(self.overall_metrics),
            "per_subject_metrics": list(self.per_subject_metrics),
            "evaluation_protocol": self.evaluation_protocol,
            "runtime_metadata": self.runtime_metadata,
        }


def save_nested_evaluation_result(result: NestedEvaluationResult, output_dir: Path | str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata = result.to_metadata_dict()
    (output_path / "nested_result.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    arrays: dict[str, np.ndarray] = {
        "aggregated_labels": np.asarray(result.aggregated_labels),
        "aggregated_probabilities": np.asarray(result.aggregated_probabilities),
        "aggregated_predictions": np.asarray(result.aggregated_predictions),
    }

    for fold in result.outer_fold_results:
        prefix = f"outer_{fold.outer_fold_index}"
        arrays[f"{prefix}_true_labels"] = np.asarray(fold.outer_test_true_labels)
        arrays[f"{prefix}_probabilities"] = np.asarray(fold.outer_test_probabilities)
        arrays[f"{prefix}_predictions"] = np.asarray(fold.outer_test_predictions)
        for idx, candidate in enumerate(fold.candidate_results):
            candidate_prefix = f"{prefix}_candidate_{idx}"
            arrays[f"{candidate_prefix}_oof_probabilities"] = np.asarray(candidate.inner_oof_probabilities)
            arrays[f"{candidate_prefix}_oof_predictions"] = np.asarray(candidate.inner_oof_predictions)

    np.savez_compressed(output_path / "nested_arrays.npz", **arrays)


def load_nested_evaluation_result(output_dir: Path | str) -> NestedEvaluationResult:
    output_path = Path(output_dir)
    metadata = json.loads((output_path / "nested_result.json").read_text(encoding="utf-8"))
    arrays = np.load(output_path / "nested_arrays.npz", allow_pickle=False)

    outer_fold_results: list[OuterFoldResult] = []
    for outer_meta in metadata["outer_fold_results"]:
        outer_index = outer_meta["outer_fold_index"]
        prefix = f"outer_{outer_index}"
        candidates: list[CandidateResult] = []
        for idx, candidate_meta in enumerate(outer_meta["candidate_results"]):
            candidate_prefix = f"{prefix}_candidate_{idx}"
            candidates.append(
                CandidateResult(
                    method_name=candidate_meta["method_name"],
                    candidate_id=candidate_meta["candidate_id"],
                    hyperparameters=dict(candidate_meta["hyperparameters"]),
                    inner_oof_probabilities=np.asarray(arrays[f"{candidate_prefix}_oof_probabilities"]),
                    inner_oof_predictions=np.asarray(arrays[f"{candidate_prefix}_oof_predictions"]),
                    inner_oof_metrics=dict(candidate_meta["inner_oof_metrics"]),
                    selected_threshold=float(candidate_meta["selected_threshold"]),
                    model_complexity=float(candidate_meta["model_complexity"]),
                    status=candidate_meta["status"],
                    failure_reason=candidate_meta["failure_reason"],
                    outer_fold_index=candidate_meta.get("outer_fold_index"),
                    failed_inner_fold_index=candidate_meta.get("failed_inner_fold_index"),
                )
            )

        outer_fold_results.append(
            OuterFoldResult(
                outer_fold_index=outer_index,
                outer_test_subject=outer_meta["outer_test_subject"],
                outer_training_subjects=list(outer_meta["outer_training_subjects"]),
                selected_candidate_id=outer_meta["selected_candidate_id"],
                selected_hyperparameters=dict(outer_meta["selected_hyperparameters"]),
                selected_threshold=float(outer_meta["selected_threshold"]),
                inner_selection_metrics=dict(outer_meta["inner_selection_metrics"]),
                outer_test_sample_ids=list(outer_meta["outer_test_sample_ids"]),
                outer_test_true_labels=np.asarray(arrays[f"{prefix}_true_labels"]),
                outer_test_probabilities=np.asarray(arrays[f"{prefix}_probabilities"]),
                outer_test_predictions=np.asarray(arrays[f"{prefix}_predictions"]),
                seed=int(outer_meta["seed"]),
                configuration_hash=outer_meta["configuration_hash"],
                candidate_results=candidates,
            )
        )

    return NestedEvaluationResult(
        method_name=metadata["method_name"],
        outer_fold_results=outer_fold_results,
        aggregated_sample_ids=list(metadata["aggregated_sample_ids"]),
        aggregated_subject_ids=list(metadata["aggregated_subject_ids"]),
        aggregated_labels=np.asarray(arrays["aggregated_labels"]),
        aggregated_probabilities=np.asarray(arrays["aggregated_probabilities"]),
        aggregated_predictions=np.asarray(arrays["aggregated_predictions"]),
        overall_metrics=dict(metadata["overall_metrics"]),
        per_subject_metrics=list(metadata["per_subject_metrics"]),
        evaluation_protocol=metadata.get("evaluation_protocol", "loso+groupkfold"),
        runtime_metadata=metadata.get("runtime_metadata"),
    )
