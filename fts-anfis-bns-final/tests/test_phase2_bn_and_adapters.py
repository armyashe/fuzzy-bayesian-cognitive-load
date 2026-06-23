from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbn_pipeline.fold_manifest import build_subject_loso_manifest
from fbn_pipeline.models.bn_fusion import BNFusion
from fbn_pipeline.nested_evaluation import NestedDataset, run_nested_subject_loso
from fbn_pipeline.production_adapters import (
    FTSLogisticRegressionAdapter,
    FTSRBFSVMAdapter,
)
from fbn_pipeline.production_bn import (
    fit_bn_from_cross_fitted_evidence,
    generate_cross_fitted_signal_evidence,
)
from fbn_pipeline.structured_config import EvaluationConfig


def make_eval_config() -> EvaluationConfig:
    return EvaluationConfig(
        outer_protocol="loso",
        inner_folds=2,
        selection_metric="balanced_accuracy",
        threshold_metric="balanced_accuracy",
        threshold_grid=(0.3, 0.4, 0.5, 0.6, 0.7),
        seed=7,
        deterministic_threads=1,
        output_directory="artifacts/phase2-tests",
    )


def make_grouped_dataset() -> NestedDataset:
    n_subjects = 4
    samples_per_subject = 3
    total = n_subjects * samples_per_subject
    subject_ids = np.repeat([f"s{i}" for i in range(n_subjects)], samples_per_subject)
    sample_ids = [f"sample_{i}" for i in range(total)]
    X = np.linspace(-1.0, 1.0, total * 12, dtype=float).reshape(total, 4, 3)
    y = np.array([0, 0, 1] * n_subjects, dtype=int)
    return NestedDataset(X=X, y=y, subject_ids=subject_ids, sample_ids=sample_ids)


def test_cross_fitted_evidence_covers_each_sample_once_and_bn_records_cross_fitted_source() -> None:
    dataset = make_grouped_dataset()

    def predictor(train_data: NestedDataset, valid_data: NestedDataset, seed: int) -> np.ndarray:
        assert not set(train_data.subject_ids.tolist()) & set(valid_data.subject_ids.tolist())
        base = valid_data.X.mean(axis=(1, 2))
        return np.column_stack(
            [
                0.50 + 0.10 * base,
                0.45 + 0.08 * base,
                0.55 + 0.06 * base,
                0.40 + 0.04 * base,
            ]
        )

    crossfit = generate_cross_fitted_signal_evidence(
        dataset=dataset,
        n_splits=2,
        seed=11,
        predictor=predictor,
        metadata={
            "outer_fold_index": 0,
            "inner_fold_index": 0,
            "membership_type": "triangular",
            "rule_count": 3,
        },
    )
    bn = fit_bn_from_cross_fitted_evidence(
        evidence_result=crossfit,
        y_true=dataset.y,
        n_bins=5,
        alpha=0.5,
        soft_evidence=True,
    )

    assert crossfit.evidence.shape == (len(dataset.y), 4)
    assert np.all(crossfit.coverage_counts == 1)
    assert bn.fit_source_ == "cross_fitted"
    assert bn.fit_sample_ids_ == dataset.sample_ids


def test_soft_evidence_explanation_reconstructs_predict_proba() -> None:
    probs = np.array(
        [
            [0.15, 0.30, 0.45, 0.60],
            [0.20, 0.35, 0.40, 0.75],
            [0.55, 0.50, 0.65, 0.80],
            [0.75, 0.70, 0.60, 0.55],
            [0.85, 0.65, 0.70, 0.90],
        ],
        dtype=float,
    )
    y = np.array([0, 0, 1, 1, 1], dtype=int)
    bn = BNFusion(n_bins=5, alpha=0.5, soft=True).fit(probs, y)

    sample = np.array([0.52, 0.48, 0.61, 0.77], dtype=float)
    posterior = float(bn.predict_proba(sample.reshape(1, -1))[0])
    trace = bn.explain_sample(sample, threshold=0.42)

    assert abs(trace["reconstructed_posterior"]["task"] - posterior) < 1e-8
    assert trace["selected_threshold"] == 0.42
    assert trace["predicted_class"] == int(posterior >= 0.42)
    assert len(trace["signal_contributions"]) == 4


def test_production_adapters_use_shared_manifest_and_select_memberships_independently(tmp_path: Path) -> None:
    dataset = make_grouped_dataset()
    manifest = build_subject_loso_manifest(
        dataset=dataset,
        evaluation=make_eval_config(),
        output_path=tmp_path / "fold_manifest.json",
    )

    lr_result = run_nested_subject_loso(
        dataset=dataset,
        adapter=FTSLogisticRegressionAdapter(
            membership_types=("gaussian", "trapezoid"),
            c_values=(0.1,),
        ),
        evaluation=make_eval_config(),
        manifest=manifest,
    )
    svm_result = run_nested_subject_loso(
        dataset=dataset,
        adapter=FTSRBFSVMAdapter(
            membership_types=("triangular", "gbell"),
            c_values=(0.1,),
            gamma_values=("scale",),
        ),
        evaluation=make_eval_config(),
        manifest=manifest,
    )

    assert lr_result.runtime_metadata["manifest_hash"] == manifest.manifest_hash
    assert svm_result.runtime_metadata["manifest_hash"] == manifest.manifest_hash
    assert lr_result.outer_fold_results[0].selected_hyperparameters["mf_type"] in {"gaussian", "trapezoid"}
    assert svm_result.outer_fold_results[0].selected_hyperparameters["mf_type"] in {"triangular", "gbell"}
