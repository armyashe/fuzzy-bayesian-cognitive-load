from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .fold_manifest import FoldManifest, build_subject_loso_manifest, load_fold_manifest, validate_manifest_against_dataset
from .nested_evaluation import NestedDataset, run_nested_subject_loso
from .nested_results import NestedEvaluationResult, save_nested_evaluation_result
from .structured_config import EvaluationConfig


def ensure_shared_manifest(
    *,
    dataset: NestedDataset,
    evaluation: EvaluationConfig,
    output_directory: Path | str,
) -> FoldManifest:
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "fold_manifest.json"
    if manifest_path.exists():
        manifest = load_fold_manifest(manifest_path)
        try:
            validate_manifest_against_dataset(manifest, dataset)
        except ValueError:
            manifest_path.unlink()
        else:
            return manifest
    return build_subject_loso_manifest(
        dataset=dataset,
        evaluation=evaluation,
        output_path=manifest_path,
    )


def run_and_persist_method(
    *,
    dataset: NestedDataset,
    adapter,
    evaluation: EvaluationConfig,
    output_directory: Path | str,
    manifest: FoldManifest | None = None,
    allow_partial_outer_folds: bool = False,
) -> NestedEvaluationResult:
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_manifest = manifest or ensure_shared_manifest(
        dataset=dataset,
        evaluation=evaluation,
        output_directory=output_dir,
    )
    result = run_nested_subject_loso(
        dataset=dataset,
        adapter=adapter,
        evaluation=evaluation,
        manifest=resolved_manifest,
        allow_partial_outer_folds=allow_partial_outer_folds,
    )
    method_dir = output_dir / adapter.method_name
    method_dir.mkdir(parents=True, exist_ok=True)
    save_nested_evaluation_result(result, method_dir)
    _save_predictions_csv(result, method_dir / "outer_predictions.csv")
    _save_candidate_scores_csv(result, method_dir / "candidate_scores.csv")
    return result


def _save_predictions_csv(result: NestedEvaluationResult, output_path: Path) -> None:
    rows: list[dict[str, object]] = []
    sample_to_subject = dict(zip(result.aggregated_sample_ids, result.aggregated_subject_ids, strict=True))
    sample_to_label = dict(zip(result.aggregated_sample_ids, result.aggregated_labels.tolist(), strict=True))
    for fold in result.outer_fold_results:
        probability_by_sample = dict(zip(fold.outer_test_sample_ids, fold.outer_test_probabilities.tolist(), strict=True))
        prediction_by_sample = dict(zip(fold.outer_test_sample_ids, fold.outer_test_predictions.tolist(), strict=True))
        for sample_id in fold.outer_test_sample_ids:
            rows.append(
                {
                    "method": result.method_name,
                    "outer_fold": int(fold.outer_fold_index),
                    "sample_id": sample_id,
                    "subject_id": sample_to_subject[sample_id],
                    "true_label": int(sample_to_label[sample_id]),
                    "score_or_probability": float(probability_by_sample[sample_id]),
                    "prediction": int(prediction_by_sample[sample_id]),
                    "selected_threshold": float(fold.selected_threshold),
                }
            )

    _write_rows(output_path, rows)


def _save_candidate_scores_csv(result: NestedEvaluationResult, output_path: Path) -> None:
    rows: list[dict[str, object]] = []
    for fold in result.outer_fold_results:
        for candidate in fold.candidate_results:
            rows.append(
                {
                    "method": result.method_name,
                    "outer_fold": int(fold.outer_fold_index),
                    "outer_test_subject": fold.outer_test_subject,
                    "candidate_id": candidate.candidate_id,
                    "hyperparameters_json": json.dumps(candidate.hyperparameters, sort_keys=True),
                    "selected_threshold": float(candidate.selected_threshold),
                    "balanced_accuracy": float(candidate.inner_oof_metrics.get("bal_acc", float("nan"))),
                    "auc": float(candidate.inner_oof_metrics.get("auc", float("nan"))),
                    "complexity": float(candidate.model_complexity),
                    "status": candidate.status,
                    "failure_reason": candidate.failure_reason,
                    "outer_fold_index": candidate.outer_fold_index,
                    "failed_inner_fold_index": candidate.failed_inner_fold_index,
                }
            )
    _write_rows(output_path, rows)


def _write_rows(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
