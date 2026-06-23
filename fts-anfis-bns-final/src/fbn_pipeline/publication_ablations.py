from __future__ import annotations

import numpy as np

from .fold_manifest import FoldManifest
from .nested_evaluation import (
    LeakageError,
    NestedDataset,
    _compute_metrics_from_predictions,
    _select_threshold,
    _validate_outer_fold_subject_coverage,
    _validate_outer_prediction_coverage,
    assert_no_subject_overlap,
)
from .nested_results import CandidateResult, NestedEvaluationResult, OuterFoldResult
from .official_holdout import OfficialHoldoutManifest, OfficialHoldoutResult, OfficialHoldoutSplit
from .production_adapters import _fit_signal_models, _make_fts_encoder, _predict_signal_evidence
from .production_bn import fit_bn_from_cross_fitted_evidence, generate_cross_fitted_signal_evidence
from .structured_config import EvaluationConfig


FTS_SUMMARY_TRANSITION_TERMS = (
    ("medium_to_high", 5),
    ("high_to_high", 8),
)


def build_frozen_proposed_configuration_map(result: NestedEvaluationResult) -> dict[int, dict[str, object]]:
    frozen: dict[int, dict[str, object]] = {}
    for fold in result.outer_fold_results:
        hyper = dict(fold.selected_hyperparameters)
        frozen[int(fold.outer_fold_index)] = {
            "outer_fold_index": int(fold.outer_fold_index),
            "outer_test_subject": fold.outer_test_subject,
            "mf_type": str(hyper["mf_type"]),
            "n_rules": int(hyper["n_rules"]),
            "selected_threshold": float(fold.selected_threshold),
            "source_method": result.method_name,
            "configuration_hash": fold.configuration_hash,
        }
    return frozen


def build_frozen_official_configuration(result: OfficialHoldoutResult) -> dict[str, object]:
    hyper = dict(result.selected_hyperparameters)
    return {
        "mf_type": str(hyper["mf_type"]),
        "n_rules": int(hyper["n_rules"]),
        "selected_threshold": float(result.selected_threshold),
        "source_method": result.method_name,
    }


def compute_fts_summary_signal_evidence(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    if features.ndim != 2 or features.shape[1] != 36:
        raise ValueError("Expected FTS feature matrix with shape (N, 36)")
    signal_scores: list[np.ndarray] = []
    for signal_index in range(4):
        start = signal_index * 9
        block = features[:, start:start + 9]
        signal_scores.append(block[:, 5] + block[:, 8])
    return np.column_stack(signal_scores)


def run_frozen_ablation(
    *,
    dataset: NestedDataset,
    evaluation: EvaluationConfig,
    manifest: FoldManifest,
    proposed_result: NestedEvaluationResult,
    ablation_name: str,
    allow_partial_outer_folds: bool = False,
) -> NestedEvaluationResult:
    if ablation_name not in {"fts_anfis_mean", "fts_summary_bn"}:
        raise ValueError(f"Unsupported ablation_name: {ablation_name!r}")
    _validate_outer_fold_subject_coverage(
        dataset=dataset,
        manifest=manifest,
        allow_partial_outer_folds=allow_partial_outer_folds,
    )
    frozen = build_frozen_proposed_configuration_map(proposed_result)
    aggregated_probabilities = np.full(len(dataset.y), np.nan, dtype=float)
    aggregated_predictions = np.full(len(dataset.y), -1, dtype=int)
    prediction_counts = np.zeros(len(dataset.y), dtype=int)
    sample_index_to_dataset_local = {
        int(sample_index): local_index
        for local_index, sample_index in enumerate(dataset.sample_indices.tolist())
    }
    outer_fold_results: list[OuterFoldResult] = []

    for outer_fold in manifest.outer_folds:
        frozen_cfg = frozen[int(outer_fold.outer_fold_index)]
        mf_type = str(frozen_cfg["mf_type"])
        n_rules = int(frozen_cfg["n_rules"])
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

        oof_probabilities = np.full(len(outer_train_data.y), np.nan, dtype=float)
        inner_counts = np.zeros(len(outer_train_data.y), dtype=int)
        sample_index_to_local = {
            int(sample_index): local_index
            for local_index, sample_index in enumerate(outer_train_data.sample_indices.tolist())
        }
        for inner_fold in outer_fold.inner_folds:
            tr_idx = np.array([sample_index_to_local[int(i)] for i in inner_fold.train_indices], dtype=int)
            va_idx = np.array([sample_index_to_local[int(i)] for i in inner_fold.valid_indices], dtype=int)
            train_data = outer_train_data.subset(tr_idx)
            valid_data = outer_train_data.subset(va_idx)
            assert_no_subject_overlap(train_data.subject_ids, valid_data.subject_ids, "ablation inner split")
            probabilities = _predict_ablation_probabilities(
                train_data=train_data,
                eval_data=valid_data,
                mf_type=mf_type,
                n_rules=n_rules,
                seed=outer_fold.seed + inner_fold.inner_fold_index,
                ablation_name=ablation_name,
            )
            if len(probabilities) != len(va_idx):
                raise LeakageError("Ablation inner prediction length does not match validation samples")
            oof_probabilities[va_idx] = probabilities
            inner_counts[va_idx] += 1
        if np.any(np.isnan(oof_probabilities)):
            raise LeakageError("Ablation inner OOF predictions are incomplete")
        if np.any(inner_counts != 1):
            raise LeakageError("Ablation inner OOF prediction coverage must equal one per sample")

        threshold = _select_threshold(outer_train_data.y, oof_probabilities, evaluation.threshold_grid)
        ranking_metrics = _compute_metrics_from_predictions(
            outer_train_data.y,
            oof_probabilities,
            (oof_probabilities >= threshold).astype(int),
        )
        outer_probabilities = _predict_ablation_probabilities(
            train_data=outer_train_data,
            eval_data=outer_test_data,
            mf_type=mf_type,
            n_rules=n_rules,
            seed=outer_fold.seed,
            ablation_name=ablation_name,
        )
        outer_predictions = (outer_probabilities >= threshold).astype(int)
        aggregated_probabilities[outer_test_indices] = outer_probabilities
        aggregated_predictions[outer_test_indices] = outer_predictions
        prediction_counts[outer_test_indices] += 1
        candidate_result = CandidateResult(
            method_name=ablation_name,
            candidate_id=f"{ablation_name}_{mf_type}_{n_rules}",
            hyperparameters={"mf_type": mf_type, "n_rules": n_rules},
            inner_oof_probabilities=oof_probabilities,
            inner_oof_predictions=(oof_probabilities >= threshold).astype(int),
            inner_oof_metrics=ranking_metrics,
            selected_threshold=float(threshold),
            model_complexity=float(n_rules),
            status="success",
            failure_reason=None,
            outer_fold_index=int(outer_fold.outer_fold_index),
            failed_inner_fold_index=None,
        )
        outer_fold_results.append(
            OuterFoldResult(
                outer_fold_index=int(outer_fold.outer_fold_index),
                outer_test_subject=outer_fold.outer_test_subject,
                outer_training_subjects=list(outer_fold.outer_train_subjects),
                selected_candidate_id=candidate_result.candidate_id,
                selected_hyperparameters=dict(candidate_result.hyperparameters),
                selected_threshold=float(threshold),
                inner_selection_metrics=ranking_metrics,
                outer_test_sample_ids=list(outer_test_data.sample_ids),
                outer_test_true_labels=outer_test_data.y.copy(),
                outer_test_probabilities=outer_probabilities,
                outer_test_predictions=outer_predictions,
                seed=evaluation.seed,
                configuration_hash=str(frozen_cfg["configuration_hash"]),
                candidate_results=[candidate_result],
            )
        )

    covered_indices = _validate_outer_prediction_coverage(
        aggregated_probabilities=aggregated_probabilities,
        prediction_counts=prediction_counts,
        allow_partial_outer_folds=allow_partial_outer_folds,
    )
    aggregated_labels = dataset.y[covered_indices].copy()
    aggregated_probabilities = aggregated_probabilities[covered_indices]
    aggregated_predictions = aggregated_predictions[covered_indices]
    return NestedEvaluationResult(
        method_name=ablation_name,
        outer_fold_results=outer_fold_results,
        aggregated_sample_ids=[dataset.sample_ids[i] for i in covered_indices.tolist()],
        aggregated_subject_ids=dataset.subject_ids[covered_indices].astype(str).tolist(),
        aggregated_labels=aggregated_labels,
        aggregated_probabilities=aggregated_probabilities,
        aggregated_predictions=aggregated_predictions,
        overall_metrics=_compute_metrics_from_predictions(
            aggregated_labels,
            aggregated_probabilities,
            aggregated_predictions,
        ),
        per_subject_metrics=[],
        runtime_metadata={
            "manifest_hash": manifest.manifest_hash,
            "dataset_order_hash": manifest.dataset_order_hash,
            "frozen_from_method": proposed_result.method_name,
        },
    )


def run_frozen_official_ablation(
    *,
    split: OfficialHoldoutSplit,
    evaluation: EvaluationConfig,
    manifest: OfficialHoldoutManifest,
    proposed_result: OfficialHoldoutResult,
    ablation_name: str,
) -> OfficialHoldoutResult:
    if ablation_name not in {"fts_anfis_mean", "fts_summary_bn"}:
        raise ValueError(f"Unsupported ablation_name: {ablation_name!r}")
    frozen_cfg = build_frozen_official_configuration(proposed_result)
    train_dataset = split.train_dataset
    test_dataset = split.test_dataset
    sample_index_to_train_local = {
        int(sample_index): local_index
        for local_index, sample_index in enumerate(train_dataset.sample_indices.tolist())
    }
    oof_probabilities = np.full(len(train_dataset.y), np.nan, dtype=float)
    prediction_counts = np.zeros(len(train_dataset.y), dtype=int)
    for fold in manifest.folds:
        tr_idx = np.array([sample_index_to_train_local[int(i)] for i in fold.train_indices], dtype=int)
        va_idx = np.array([sample_index_to_train_local[int(i)] for i in fold.valid_indices], dtype=int)
        train_data = train_dataset.subset(tr_idx)
        valid_data = train_dataset.subset(va_idx)
        assert_no_subject_overlap(train_data.subject_ids, valid_data.subject_ids, "official ablation inner split")
        assert_no_subject_overlap(train_data.subject_ids, split.test_subject_ids, "official ablation train-vs-test")
        assert_no_subject_overlap(valid_data.subject_ids, split.test_subject_ids, "official ablation valid-vs-test")
        probabilities = _predict_ablation_probabilities(
            train_data=train_data,
            eval_data=valid_data,
            mf_type=str(frozen_cfg["mf_type"]),
            n_rules=int(frozen_cfg["n_rules"]),
            seed=fold.seed,
            ablation_name=ablation_name,
        )
        if len(probabilities) != len(va_idx):
            raise LeakageError("Official ablation validation prediction length does not match validation samples")
        oof_probabilities[va_idx] = probabilities
        prediction_counts[va_idx] += 1
    if np.any(np.isnan(oof_probabilities)):
        raise LeakageError("Official ablation training OOF predictions are incomplete")
    if np.any(prediction_counts != 1):
        raise LeakageError("Official ablation training OOF coverage must equal one per sample")

    threshold = _select_threshold(train_dataset.y, oof_probabilities, evaluation.threshold_grid)
    ranking_metrics = _compute_metrics_from_predictions(
        train_dataset.y,
        oof_probabilities,
        (oof_probabilities >= threshold).astype(int),
    )
    final_probabilities = _predict_ablation_probabilities(
        train_data=train_dataset,
        eval_data=test_dataset,
        mf_type=str(frozen_cfg["mf_type"]),
        n_rules=int(frozen_cfg["n_rules"]),
        seed=evaluation.seed,
        ablation_name=ablation_name,
    )
    if len(final_probabilities) != len(test_dataset.y):
        raise LeakageError("Official ablation final prediction length does not match test samples")
    final_predictions = (final_probabilities >= threshold).astype(int)
    candidate_result = CandidateResult(
        method_name=ablation_name,
        candidate_id=f"{ablation_name}_{frozen_cfg['mf_type']}_{frozen_cfg['n_rules']}",
        hyperparameters={
            "mf_type": str(frozen_cfg["mf_type"]),
            "n_rules": int(frozen_cfg["n_rules"]),
            "frozen_from_method": proposed_result.method_name,
        },
        inner_oof_probabilities=oof_probabilities,
        inner_oof_predictions=(oof_probabilities >= threshold).astype(int),
        inner_oof_metrics=ranking_metrics,
        selected_threshold=float(threshold),
        model_complexity=float(frozen_cfg["n_rules"]),
        status="success",
        failure_reason=None,
        outer_fold_index=None,
        failed_inner_fold_index=None,
    )
    return OfficialHoldoutResult(
        method_name=ablation_name,
        protocol_name=manifest.protocol_name,
        selected_candidate_id=candidate_result.candidate_id,
        selected_hyperparameters=dict(candidate_result.hyperparameters),
        selected_threshold=float(threshold),
        candidate_results=[candidate_result],
        train_oof_sample_ids=list(train_dataset.sample_ids),
        train_oof_subject_ids=train_dataset.subject_ids.astype(str).tolist(),
        train_oof_labels=train_dataset.y.copy(),
        train_oof_probabilities=oof_probabilities,
        train_oof_predictions=(oof_probabilities >= threshold).astype(int),
        final_test_sample_ids=list(test_dataset.sample_ids),
        final_test_subject_ids=test_dataset.subject_ids.astype(str).tolist(),
        final_test_labels=test_dataset.y.copy(),
        final_test_probabilities=final_probabilities,
        final_test_predictions=final_predictions,
        overall_test_metrics=_compute_metrics_from_predictions(
            test_dataset.y,
            final_probabilities,
            final_predictions,
        ),
        per_subject_test_metrics=[],
        runtime_metadata={
            "manifest_hash": manifest.manifest_hash,
            "dataset_order_hash": manifest.dataset_order_hash,
            "protocol_name": manifest.protocol_name,
            "frozen_from_method": proposed_result.method_name,
        },
    )


def _predict_ablation_probabilities(
    *,
    train_data: NestedDataset,
    eval_data: NestedDataset,
    mf_type: str,
    n_rules: int,
    seed: int,
    ablation_name: str,
) -> np.ndarray:
    if ablation_name == "fts_anfis_mean":
        encoder, signal_models = _fit_signal_models(
            X_train=train_data.X,
            y_train=train_data.y,
            mf_type=mf_type,
            n_rules=n_rules,
            seed=seed,
        )
        evidence = _predict_signal_evidence(
            encoder=encoder,
            signal_models=signal_models,
            X_data=eval_data.X,
        )
        return evidence.mean(axis=1)

    crossfit = generate_cross_fitted_signal_evidence(
        dataset=train_data,
        n_splits=min(5, len(np.unique(train_data.subject_ids))),
        seed=seed,
        predictor=lambda cf_train, cf_valid, cf_seed: _summary_with_train_encoder(cf_train, cf_valid, mf_type),
        metadata={"membership_type": mf_type, "rule_count": n_rules, "seed": seed},
    )
    bn = fit_bn_from_cross_fitted_evidence(
        evidence_result=crossfit,
        y_true=train_data.y,
        n_bins=5,
        alpha=0.5,
        soft_evidence=True,
    )
    encoder = _make_fts_encoder(mf_type)
    train_features = encoder.fit_transform(train_data.X)
    if train_features.shape[1] != 36:
        raise LeakageError("FTS summary ablation expects 36 FTS features")
    eval_features = encoder.transform(eval_data.X)
    return bn.predict_proba(compute_fts_summary_signal_evidence(eval_features))


def _summary_with_train_encoder(train_data: NestedDataset, valid_data: NestedDataset, mf_type: str) -> np.ndarray:
    encoder = _make_fts_encoder(mf_type)
    encoder.fit(train_data.X)
    valid_features = encoder.transform(valid_data.X)
    return compute_fts_summary_signal_evidence(valid_features)
