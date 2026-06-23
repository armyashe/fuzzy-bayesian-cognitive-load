from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .fold_manifest import build_subject_loso_manifest
from .nested_results import NestedEvaluationResult
from .official_holdout import (
    OFFICIAL_PROTOCOL,
    OFFICIAL_SMOKE_PROTOCOL,
    OfficialHoldoutResult,
    build_official_group_manifest,
    load_official_holdout_split,
    run_official_holdout_evaluation,
)
from .publication_ablations import run_frozen_ablation, run_frozen_official_ablation
from .production_adapters import ProposedFTSANFISBNAdapter
from .production_data import load_full_subject_dataset
from .production_runner import run_and_persist_method
from .publication_artifacts import (
    PublicationRunLayout,
    collect_runtime_metadata,
    validate_official_holdout_predictions,
    validate_publication_predictions,
    write_runtime_metadata,
)
from .publication_metrics import compute_per_subject_metrics_from_predictions, compute_pooled_metrics_from_predictions
from .publication_statistics import compute_bootstrap_confidence_intervals, holm_correction, run_paired_wilcoxon_by_subject
from .structured_config import EvaluationConfig


CONTROLLED_BASELINE_METHODS = {
    "fts_logistic_regression",
    "fts_rbf_svm",
    "fts_xgboost",
    "raw_1d_cnn",
}

ABLATION_METHODS = {
    "fts_anfis_mean",
    "fts_summary_bn",
}


def load_subject_dataset(data_dir: Path | str, *, subject_limit: int | None = None):
    dataset = load_full_subject_dataset(data_dir)
    if subject_limit is None:
        return dataset
    selected = sorted(pd.unique(dataset.subject_ids.astype(str)).tolist())[: int(subject_limit)]
    indices = [idx for idx, subject_id in enumerate(dataset.subject_ids.astype(str).tolist()) if subject_id in set(selected)]
    return dataset.subset(pd.Series(indices, dtype=int).to_numpy())


def build_publication_evaluation(
    *,
    output_directory: str,
    inner_folds: int,
    seed: int,
) -> EvaluationConfig:
    return EvaluationConfig(
        outer_protocol="loso",
        inner_folds=inner_folds,
        selection_metric="balanced_accuracy",
        threshold_metric="balanced_accuracy",
        threshold_grid=(0.3, 0.4, 0.5, 0.6, 0.7),
        seed=seed,
        deterministic_threads=1,
        output_directory=output_directory,
    )


def run_publication_workflow(
    *,
    data_dir: Path | str,
    layout: PublicationRunLayout,
    evaluation: EvaluationConfig,
    adapters: list,
    command: str,
    protocol: str = "official_holdout",
    allow_partial_outer_folds: bool = False,
    ablation_names: tuple[str, ...] = (),
    train_subject_limit: int | None = None,
    test_subject_limit: int | None = None,
) -> dict[str, object]:
    if protocol == "official_holdout":
        split = load_official_holdout_split(
            data_dir,
            protocol_name=OFFICIAL_SMOKE_PROTOCOL if layout.mode == "smoke" else OFFICIAL_PROTOCOL,
            train_subject_limit=train_subject_limit if layout.mode == "smoke" else None,
            test_subject_limit=test_subject_limit if layout.mode == "smoke" else None,
        )
        manifest = build_official_group_manifest(
            split=split,
            evaluation=evaluation,
            n_splits=evaluation.inner_folds,
            protocol_name=split.protocol_name,
            output_path=layout.metadata_dir / "fold_manifest.json",
        )
        return run_official_holdout_methods(
            split=split,
            manifest=manifest,
            layout=layout,
            evaluation=evaluation,
            adapters=adapters,
            command=command,
            complete_coverage=layout.mode == "publication",
            ablation_names=ablation_names,
        )
    if protocol == "nested_loso_secondary":
        dataset = load_subject_dataset(data_dir, subject_limit=train_subject_limit)
        return run_publication_methods(
            dataset=dataset,
            layout=layout,
            evaluation=evaluation,
            adapters=adapters,
            command=command,
            allow_partial_outer_folds=allow_partial_outer_folds,
            ablation_names=ablation_names,
        )
    raise ValueError(f"Unsupported protocol: {protocol!r}")


def run_publication_methods(
    *,
    dataset,
    layout: PublicationRunLayout,
    evaluation: EvaluationConfig,
    adapters: list,
    command: str,
    overwrite: bool = False,
    allow_partial_outer_folds: bool = False,
    ablation_names: tuple[str, ...] = (),
) -> dict[str, NestedEvaluationResult]:
    raw_root = layout.metadata_dir / "raw_results"
    raw_root.mkdir(parents=True, exist_ok=True)
    manifest = build_subject_loso_manifest(
        dataset=dataset,
        evaluation=evaluation,
        output_path=layout.metadata_dir / "fold_manifest.json",
    )

    results: dict[str, NestedEvaluationResult] = {}
    prediction_frames: list[pd.DataFrame] = []
    config_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    failure_frames: list[pd.DataFrame] = []
    anfis_frames: list[pd.DataFrame] = []
    bn_frames: list[pd.DataFrame] = []
    crossfit_frames: list[pd.DataFrame] = []

    proposed_result: NestedEvaluationResult | None = None
    for adapter in adapters:
        result = run_and_persist_method(
            dataset=dataset,
            adapter=adapter,
            evaluation=evaluation,
            output_directory=raw_root,
            manifest=manifest,
            allow_partial_outer_folds=allow_partial_outer_folds,
        )
        results[result.method_name] = result
        if result.method_name == "proposed_fts_anfis_bn":
            proposed_result = result
        method_predictions, method_configs, method_candidates, method_failures = flatten_result_artifacts(
            result=result,
            dataset_order_hash=manifest.dataset_order_hash,
            prediction_role="secondary_test",
        )
        prediction_frames.append(method_predictions)
        config_frames.append(method_configs)
        candidate_frames.append(method_candidates)
        if not method_failures.empty:
            failure_frames.append(method_failures)

        if isinstance(adapter, ProposedFTSANFISBNAdapter) and adapter.outer_artifacts_by_fold:
            fold_thresholds = {int(fold.outer_fold_index): float(fold.selected_threshold) for fold in result.outer_fold_results}
            for fold_index, artifacts in sorted(adapter.outer_artifacts_by_fold.items()):
                crossfit_rows = artifacts.get("crossfit_rows", [])
                if crossfit_rows:
                    crossfit_frames.append(pd.DataFrame(crossfit_rows).assign(method=result.method_name))
                for row in artifacts.get("anfis_rows", []):
                    row["method"] = result.method_name
                    row["selected_threshold"] = fold_thresholds[fold_index]
                for row in artifacts.get("bn_rows", []):
                    row["method"] = result.method_name
                    row["selected_threshold"] = fold_thresholds[fold_index]
                    row["predicted_class"] = int(row["posterior_task_probability"] >= fold_thresholds[fold_index])
                anfis_rows = artifacts.get("anfis_rows", [])
                bn_rows = artifacts.get("bn_rows", [])
                if anfis_rows:
                    anfis_frames.append(pd.DataFrame(anfis_rows))
                if bn_rows:
                    bn_frames.append(pd.DataFrame(bn_rows))

    for ablation_name in ablation_names:
        if proposed_result is None:
            raise ValueError("Ablation execution requires the proposed_fts_anfis_bn result in the same run")
        ablation_result = run_frozen_ablation(
            dataset=dataset,
            evaluation=evaluation,
            manifest=manifest,
            proposed_result=proposed_result,
            ablation_name=ablation_name,
            allow_partial_outer_folds=allow_partial_outer_folds,
        )
        results[ablation_result.method_name] = ablation_result
        method_predictions, method_configs, method_candidates, method_failures = flatten_result_artifacts(
            result=ablation_result,
            dataset_order_hash=manifest.dataset_order_hash,
            prediction_role="secondary_test",
        )
        prediction_frames.append(method_predictions)
        config_frames.append(method_configs)
        candidate_frames.append(method_candidates)
        if not method_failures.empty:
            failure_frames.append(method_failures)

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    validate_publication_predictions(
        predictions.loc[predictions["method"] == predictions["method"].iloc[0]].copy() if not predictions.empty else predictions,
        expected_subject_ids=dataset.subject_ids.astype(str).tolist(),
        expected_sample_ids=list(dataset.sample_ids),
        mode=layout.mode,
        allow_partial_outer_folds=allow_partial_outer_folds,
    )

    predictions.to_csv(layout.predictions_dir / "outer_fold_predictions.csv", index=False)
    pd.concat(config_frames, ignore_index=True).to_csv(layout.metadata_dir / "outer_fold_configs.csv", index=False)
    pd.concat(candidate_frames, ignore_index=True).to_csv(layout.metrics_dir / "candidate_scores.csv", index=False)
    failures = pd.concat(failure_frames, ignore_index=True) if failure_frames else pd.DataFrame(columns=pd.concat(candidate_frames, ignore_index=True).columns)
    failures.to_csv(layout.metrics_dir / "candidate_failures.csv", index=False)
    if crossfit_frames:
        pd.concat(crossfit_frames, ignore_index=True).to_csv(layout.interpretability_dir / "cross_fitted_anfis_evidence.csv", index=False)
    if anfis_frames:
        pd.concat(anfis_frames, ignore_index=True).to_csv(layout.interpretability_dir / "anfis_interpretability.csv", index=False)
    if bn_frames:
        pd.concat(bn_frames, ignore_index=True).to_csv(layout.interpretability_dir / "bn_explanations.csv", index=False)

    first_result = next(iter(results.values()))
    metadata = collect_runtime_metadata(
        run_name=layout.run_name,
        mode=layout.mode,
        allow_partial_outer_folds=allow_partial_outer_folds,
        seed=evaluation.seed,
        configuration_hash=first_result.outer_fold_results[0].configuration_hash if first_result.outer_fold_results else "",
        dataset_order_hash=manifest.dataset_order_hash,
        command=command,
        protocol="nested_loso_secondary",
        complete_coverage=not allow_partial_outer_folds,
        groupkfold_split_count=evaluation.inner_folds,
        model_selection_metric=evaluation.selection_metric,
        threshold_selection_metric=evaluation.threshold_metric,
    )
    write_runtime_metadata(layout, metadata)
    derive_publication_outputs(
        layout=layout,
        bootstrap_repetitions=200 if layout.mode == "smoke" else 2000,
    )
    return results


def run_official_holdout_methods(
    *,
    split,
    manifest,
    layout: PublicationRunLayout,
    evaluation: EvaluationConfig,
    adapters: list,
    command: str,
    complete_coverage: bool,
    ablation_names: tuple[str, ...] = (),
) -> dict[str, OfficialHoldoutResult]:
    results: dict[str, OfficialHoldoutResult] = {}
    prediction_frames: list[pd.DataFrame] = []
    config_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    failure_frames: list[pd.DataFrame] = []
    anfis_frames: list[pd.DataFrame] = []
    bn_frames: list[pd.DataFrame] = []
    crossfit_frames: list[pd.DataFrame] = []
    proposed_result: OfficialHoldoutResult | None = None

    for adapter in adapters:
        result = run_official_holdout_evaluation(
            split=split,
            adapter=adapter,
            evaluation=evaluation,
            manifest=manifest,
        )
        results[result.method_name] = result
        method_predictions, method_configs, method_candidates, method_failures = flatten_official_holdout_artifacts(
            result=result,
            dataset_order_hash=manifest.dataset_order_hash,
            manifest_hash=manifest.manifest_hash,
        )
        prediction_frames.append(method_predictions)
        config_frames.append(method_configs)
        candidate_frames.append(method_candidates)
        if not method_failures.empty:
            failure_frames.append(method_failures)
        if isinstance(adapter, ProposedFTSANFISBNAdapter) and adapter.outer_artifacts_by_fold:
            for _, artifacts in sorted(adapter.outer_artifacts_by_fold.items()):
                crossfit_rows = artifacts.get("crossfit_rows", [])
                if crossfit_rows:
                    crossfit_frames.append(pd.DataFrame(crossfit_rows).assign(method=result.method_name, prediction_role="train_oof"))
                anfis_rows = artifacts.get("anfis_rows", [])
                bn_rows = artifacts.get("bn_rows", [])
                if anfis_rows:
                    anfis_frames.append(pd.DataFrame(anfis_rows).assign(method=result.method_name))
                if bn_rows:
                    bn_frames.append(pd.DataFrame(bn_rows).assign(method=result.method_name))
            proposed_result = result

    for ablation_name in ablation_names:
        if proposed_result is None:
            raise ValueError("Ablation execution requires the proposed method in the same official_holdout run")
        ablation_result = run_frozen_official_ablation(
            split=split,
            evaluation=evaluation,
            manifest=manifest,
            proposed_result=proposed_result,
            ablation_name=ablation_name,
        )
        results[ablation_result.method_name] = ablation_result
        method_predictions, method_configs, method_candidates, method_failures = flatten_official_holdout_artifacts(
            result=ablation_result,
            dataset_order_hash=manifest.dataset_order_hash,
            manifest_hash=manifest.manifest_hash,
        )
        prediction_frames.append(method_predictions)
        config_frames.append(method_configs)
        candidate_frames.append(method_candidates)
        if not method_failures.empty:
            failure_frames.append(method_failures)

    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    validate_official_holdout_predictions(
        predictions,
        expected_train_sample_ids=manifest.train_sample_ids,
        expected_test_sample_ids=manifest.test_sample_ids,
        expected_test_subject_ids=manifest.test_subject_ids,
        mode=layout.mode,
        protocol=manifest.protocol_name,
        complete_coverage=complete_coverage,
    )
    predictions.to_csv(layout.predictions_dir / "outer_fold_predictions.csv", index=False)
    pd.concat(config_frames, ignore_index=True).to_csv(layout.metadata_dir / "outer_fold_configs.csv", index=False)
    pd.concat(candidate_frames, ignore_index=True).to_csv(layout.metrics_dir / "candidate_scores.csv", index=False)
    failures = pd.concat(failure_frames, ignore_index=True) if failure_frames else pd.DataFrame(columns=pd.concat(candidate_frames, ignore_index=True).columns)
    failures.to_csv(layout.metrics_dir / "candidate_failures.csv", index=False)
    if crossfit_frames:
        pd.concat(crossfit_frames, ignore_index=True).to_csv(layout.interpretability_dir / "cross_fitted_anfis_evidence.csv", index=False)
    if anfis_frames:
        pd.concat(anfis_frames, ignore_index=True).to_csv(layout.interpretability_dir / "anfis_interpretability.csv", index=False)
    if bn_frames:
        pd.concat(bn_frames, ignore_index=True).to_csv(layout.interpretability_dir / "bn_explanations.csv", index=False)

    first_result = next(iter(results.values()))
    metadata = collect_runtime_metadata(
        run_name=layout.run_name,
        mode=layout.mode,
        allow_partial_outer_folds=layout.mode == "smoke",
        seed=evaluation.seed,
        configuration_hash=first_result.selected_candidate_id,
        dataset_order_hash=manifest.dataset_order_hash,
        command=command,
        protocol=manifest.protocol_name,
        complete_coverage=complete_coverage,
        train_subject_ids=manifest.train_subject_ids,
        test_subject_ids=manifest.test_subject_ids,
        train_sample_count=manifest.train_sample_count,
        test_sample_count=manifest.test_sample_count,
        groupkfold_split_count=manifest.inner_folds,
        model_selection_metric=evaluation.selection_metric,
        threshold_selection_metric=evaluation.threshold_metric,
    )
    write_runtime_metadata(layout, metadata)
    derive_publication_outputs(
        layout=layout,
        bootstrap_repetitions=200 if layout.mode == "smoke" else 2000,
    )
    return results


def flatten_result_artifacts(
    *,
    result: NestedEvaluationResult,
    dataset_order_hash: str,
    prediction_role: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, object]] = []
    config_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    for fold in result.outer_fold_results:
        config_rows.append(
            {
                "method": result.method_name,
                "outer_fold_index": int(fold.outer_fold_index),
                "outer_test_subject": fold.outer_test_subject,
                "selected_candidate_id": fold.selected_candidate_id,
                "selected_threshold": float(fold.selected_threshold),
                "selected_hyperparameters_json": json.dumps(fold.selected_hyperparameters, sort_keys=True),
                "seed": int(fold.seed),
                "configuration_hash": fold.configuration_hash,
                "dataset_order_hash": dataset_order_hash,
            }
        )
        for sample_id, label, prob, pred in zip(
            fold.outer_test_sample_ids,
            fold.outer_test_true_labels.tolist(),
            fold.outer_test_probabilities.tolist(),
            fold.outer_test_predictions.tolist(),
            strict=True,
        ):
            prediction_rows.append(
                {
                    "method": result.method_name,
                    "outer_fold_index": int(fold.outer_fold_index),
                    "subject_id": fold.outer_test_subject,
                    "sample_id": sample_id,
                    "true_label": int(label),
                    "score_or_probability": float(prob),
                    "prediction": int(pred),
                    "threshold": float(fold.selected_threshold),
                    "prediction_role": prediction_role,
                    "seed": int(fold.seed),
                    "configuration_hash": fold.configuration_hash,
                    "dataset_order_hash": dataset_order_hash,
                }
            )
        for rank, candidate in enumerate(fold.candidate_results, start=1):
            row = {
                "method": result.method_name,
                "outer_fold_index": int(fold.outer_fold_index),
                "outer_test_subject": fold.outer_test_subject,
                "candidate_id": candidate.candidate_id,
                "candidate_rank": int(rank),
                "selected_candidate_id": fold.selected_candidate_id,
                "hyperparameters_json": json.dumps(candidate.hyperparameters, sort_keys=True),
                "selected_threshold": float(candidate.selected_threshold),
                "balanced_accuracy": float(candidate.inner_oof_metrics.get("bal_acc", float("nan"))),
                "accuracy": float(candidate.inner_oof_metrics.get("accuracy", candidate.inner_oof_metrics.get("acc", float("nan")))),
                "macro_f1": float(candidate.inner_oof_metrics.get("macro_f1", float("nan"))),
                "auc": float(candidate.inner_oof_metrics.get("auc", float("nan"))),
                "complexity": float(candidate.model_complexity),
                "status": candidate.status,
                "failure_reason": candidate.failure_reason,
                "configuration_hash": fold.configuration_hash,
                "dataset_order_hash": dataset_order_hash,
            }
            candidate_rows.append(row)
            if candidate.status != "success":
                failure_rows.append(row)

    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(config_rows),
        pd.DataFrame(candidate_rows),
        pd.DataFrame(failure_rows),
    )


def flatten_official_holdout_artifacts(
    *,
    result: OfficialHoldoutResult,
    dataset_order_hash: str,
    manifest_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, object]] = []
    config_rows = [
        {
            "method": result.method_name,
            "protocol": result.protocol_name,
            "selected_candidate_id": result.selected_candidate_id,
            "selected_threshold": float(result.selected_threshold),
            "selected_hyperparameters_json": json.dumps(result.selected_hyperparameters, sort_keys=True),
            "configuration_hash": result.runtime_metadata.get("configuration_hash", result.selected_candidate_id),
            "dataset_order_hash": dataset_order_hash,
            "manifest_hash": manifest_hash,
        }
    ]
    for sample_id, subject_id, label, prob, pred in zip(
        result.train_oof_sample_ids,
        result.train_oof_subject_ids,
        result.train_oof_labels.tolist(),
        result.train_oof_probabilities.tolist(),
        result.train_oof_predictions.tolist(),
        strict=True,
    ):
        prediction_rows.append(
            {
                "method": result.method_name,
                "subject_id": subject_id,
                "sample_id": sample_id,
                "true_label": int(label),
                "score_or_probability": float(prob),
                "prediction": int(pred),
                "threshold": float(result.selected_threshold),
                "prediction_role": "train_oof",
                "protocol": result.protocol_name,
                "configuration_hash": result.runtime_metadata.get("configuration_hash", result.selected_candidate_id),
                "dataset_order_hash": dataset_order_hash,
                "manifest_hash": manifest_hash,
            }
        )
    for sample_id, subject_id, label, prob, pred in zip(
        result.final_test_sample_ids,
        result.final_test_subject_ids,
        result.final_test_labels.tolist(),
        result.final_test_probabilities.tolist(),
        result.final_test_predictions.tolist(),
        strict=True,
    ):
        prediction_rows.append(
            {
                "method": result.method_name,
                "subject_id": subject_id,
                "sample_id": sample_id,
                "true_label": int(label),
                "score_or_probability": float(prob),
                "prediction": int(pred),
                "threshold": float(result.selected_threshold),
                "prediction_role": "official_test",
                "protocol": result.protocol_name,
                "configuration_hash": result.runtime_metadata.get("configuration_hash", result.selected_candidate_id),
                "dataset_order_hash": dataset_order_hash,
                "manifest_hash": manifest_hash,
            }
        )
    candidate_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    for rank, candidate in enumerate(result.candidate_results, start=1):
        row = {
            "method": result.method_name,
            "protocol": result.protocol_name,
            "candidate_id": candidate.candidate_id,
            "candidate_rank": int(rank),
            "selected_candidate_id": result.selected_candidate_id,
            "hyperparameters_json": json.dumps(candidate.hyperparameters, sort_keys=True),
            "selected_threshold": float(candidate.selected_threshold),
            "balanced_accuracy": float(candidate.inner_oof_metrics.get("bal_acc", float("nan"))),
            "accuracy": float(candidate.inner_oof_metrics.get("accuracy", candidate.inner_oof_metrics.get("acc", float("nan")))),
            "macro_f1": float(candidate.inner_oof_metrics.get("macro_f1", float("nan"))),
            "auc": float(candidate.inner_oof_metrics.get("auc", float("nan"))),
            "complexity": float(candidate.model_complexity),
            "status": candidate.status,
            "failure_reason": candidate.failure_reason,
            "dataset_order_hash": dataset_order_hash,
            "manifest_hash": manifest_hash,
        }
        candidate_rows.append(row)
        if candidate.status != "success":
            failure_rows.append(row)
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(config_rows),
        pd.DataFrame(candidate_rows),
        pd.DataFrame(failure_rows),
    )


def derive_publication_outputs(
    *,
    layout: PublicationRunLayout,
    bootstrap_repetitions: int,
) -> None:
    predictions = pd.read_csv(layout.predictions_dir / "outer_fold_predictions.csv")
    metadata = json.loads((layout.metadata_dir / "runtime_metadata.json").read_text(encoding="utf-8"))
    protocol = str(metadata.get("protocol") or "")
    if protocol == OFFICIAL_PROTOCOL:
        final_rows = predictions.loc[predictions["prediction_role"] == "official_test"].copy()
    else:
        final_rows = predictions.loc[predictions["prediction_role"] != "train_oof"].copy() if "prediction_role" in predictions.columns else predictions.copy()
    overall_rows: list[dict[str, object]] = []
    subject_frames: list[pd.DataFrame] = []
    ci_frames: list[pd.DataFrame] = []
    config_frame = pd.read_csv(layout.metadata_dir / "outer_fold_configs.csv")

    for method in sorted(final_rows["method"].unique().tolist()):
        method_frame = final_rows.loc[final_rows["method"] == method].copy()
        pooled = compute_pooled_metrics_from_predictions(method_frame)
        overall_rows.append({"method": method, **pooled})
        subject_frame = compute_per_subject_metrics_from_predictions(method_frame)
        subject_frames.append(subject_frame)
        ci_frames.append(
            compute_bootstrap_confidence_intervals(
                final_rows,
                metric_names=("balanced_accuracy", "accuracy", "f1_task", "macro_f1", "roc_auc"),
                n_bootstrap=bootstrap_repetitions,
                seed=7,
                method=method,
            )
        )

    overall = pd.DataFrame(overall_rows)
    overall.to_csv(layout.metrics_dir / "overall_metrics.csv", index=False)
    per_subject = pd.concat(subject_frames, ignore_index=True) if subject_frames else pd.DataFrame()
    per_subject.to_csv(layout.metrics_dir / "per_subject_metrics.csv", index=False)
    pd.concat(ci_frames, ignore_index=True).to_csv(layout.statistics_dir / "confidence_intervals.csv", index=False)

    selection_rows: list[dict[str, object]] = []
    for method, method_configs in config_frame.groupby("method", sort=True):
        expanded = method_configs["selected_hyperparameters_json"].apply(json.loads)
        key_counts: dict[tuple[str, str], int] = {}
        for payload in expanded:
            for key, value in payload.items():
                item = (key, str(value))
                key_counts[item] = key_counts.get(item, 0) + 1
        total = len(method_configs)
        for (key, value), count in sorted(key_counts.items()):
            selection_rows.append(
                {
                    "method": method,
                    "hyperparameter": key,
                    "value": value,
                    "selection_count": int(count),
                    "selection_rate": float(count / total),
                }
            )
    pd.DataFrame(selection_rows).to_csv(layout.statistics_dir / "selection_frequencies.csv", index=False)

    baseline_rows = overall.loc[overall["method"].isin(CONTROLLED_BASELINE_METHODS)].copy()
    if not baseline_rows.empty and not per_subject.empty and "balanced_accuracy" in per_subject.columns:
        baseline_rows = baseline_rows.sort_values("balanced_accuracy", ascending=False)
        strongest_baseline = str(baseline_rows.iloc[0]["method"])
        statistical_rows: list[dict[str, object]] = []
        comparisons = []
        for baseline_method in baseline_rows["method"].astype(str).tolist():
            stat = run_paired_wilcoxon_by_subject(
                per_subject,
                method_a="proposed_fts_anfis_bn",
                method_b=baseline_method,
                metric="balanced_accuracy",
            )
            stat["is_strongest_baseline"] = baseline_method == strongest_baseline
            comparisons.append(stat)
        corrected = holm_correction([row["p_value"] for row in comparisons])
        for row, corrected_p in zip(comparisons, corrected, strict=True):
            row["corrected_p_value"] = corrected_p
            statistical_rows.append(row)
        pd.DataFrame(statistical_rows).to_csv(layout.statistics_dir / "statistical_tests.csv", index=False)
    else:
        pd.DataFrame().to_csv(layout.statistics_dir / "statistical_tests.csv", index=False)

    overall.loc[overall["method"].isin({"proposed_fts_anfis_bn", *ABLATION_METHODS})].to_csv(
        layout.metrics_dir / "ablation_results.csv",
        index=False,
    )
