from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbn_pipeline.fold_manifest import build_subject_loso_manifest
from fbn_pipeline.nested_evaluation import CandidateSpec, LeakageError, NestedDataset, run_nested_subject_loso
from fbn_pipeline.production_adapters import (
    FTSXGBoostAdapter,
    HAS_TORCH,
    Raw1DCNNAdapter,
    _prepare_raw_cnn_arrays,
)
from fbn_pipeline.structured_config import EvaluationConfig


def make_eval_config() -> EvaluationConfig:
    return EvaluationConfig(
        outer_protocol="loso",
        inner_folds=2,
        selection_metric="balanced_accuracy",
        threshold_metric="balanced_accuracy",
        threshold_grid=(0.3, 0.5, 0.7),
        seed=13,
        deterministic_threads=1,
        output_directory="artifacts/phase2-tests",
    )


def make_grouped_dataset() -> NestedDataset:
    sample_ids = [f"s{i}" for i in range(9)]
    subject_ids = np.repeat(["u0", "u1", "u2"], 3)
    X = np.zeros((9, 4, 30), dtype=float)
    for i in range(9):
        X[i, :, 0] = i + 1
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=int)
    return NestedDataset(X=X, y=y, subject_ids=subject_ids, sample_ids=sample_ids)


def make_grouped_raw_dataset() -> NestedDataset:
    sample_ids = [f"r{i}" for i in range(15)]
    subject_ids = np.repeat(["u0", "u1", "u2", "u3", "u4"], 3)
    X = np.zeros((15, 4, 30), dtype=float)
    for i in range(15):
        X[i] = float(i + 1)
    y = np.array([0, 1, 0] * 5, dtype=int)
    return NestedDataset(X=X, y=y, subject_ids=subject_ids, sample_ids=sample_ids)


def test_xgboost_adapter_uses_manifest_and_training_only_fts(monkeypatch, tmp_path: Path) -> None:
    dataset = make_grouped_dataset()
    manifest = build_subject_loso_manifest(
        dataset=dataset,
        evaluation=make_eval_config(),
        output_path=tmp_path / "fold_manifest.json",
    )
    manifest.outer_folds = [manifest.outer_folds[0]]
    outer_test_markers = {
        int(dataset.X[i, 0, 0])
        for i, subject in enumerate(dataset.subject_ids.tolist())
        if subject == manifest.outer_folds[0].outer_test_subject
    }
    fit_markers: list[set[int]] = []

    class SpyEncoder:
        def fit_transform(self, X):
            markers = {int(v) for v in X[:, 0, 0].tolist()}
            fit_markers.append(markers)
            assert not markers & outer_test_markers
            return np.repeat(X[:, 0, :9], 4, axis=1)

        def transform(self, X):
            return np.repeat(X[:, 0, :9], 4, axis=1)

    class StubXGBClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X, y):
            self.bias = float(np.mean(y))
            return self

        def predict_proba(self, X):
            p = np.full(len(X), self.bias, dtype=float)
            return np.column_stack([1.0 - p, p])

    monkeypatch.setattr("fbn_pipeline.production_adapters._make_fts_encoder", lambda mf_type: SpyEncoder())
    monkeypatch.setattr("fbn_pipeline.production_adapters._get_xgb_classifier", lambda: StubXGBClassifier)

    result = run_nested_subject_loso(
        dataset=dataset,
        adapter=FTSXGBoostAdapter(
            membership_types=("triangular",),
            n_estimators_values=(100,),
            max_depth_values=(3,),
            learning_rate_values=(0.1,),
        ),
        evaluation=make_eval_config(),
        manifest=manifest,
        allow_partial_outer_folds=True,
    )

    assert result.runtime_metadata["manifest_hash"] == manifest.manifest_hash
    assert fit_markers


class PartialFailureAdapter:
    method_name = "partial_failure"

    def candidate_specs(self) -> list[CandidateSpec]:
        return [
            CandidateSpec(candidate_id="bad", hyperparameters={}, complexity=1.0),
            CandidateSpec(candidate_id="good", hyperparameters={}, complexity=2.0),
        ]

    def fit_predict_inner_fold(self, train_data, valid_data, candidate, *, seed):
        if candidate.candidate_id == "bad":
            raise ValueError("ponytail: synthetic candidate failure")
        return np.full(len(valid_data.y), 0.6, dtype=float)

    def fit_predict_outer_fold(self, train_data, test_data, candidate, *, seed):
        return np.full(len(test_data.y), 0.6, dtype=float)


class AllFailureAdapter:
    method_name = "all_failure"

    def candidate_specs(self) -> list[CandidateSpec]:
        return [CandidateSpec(candidate_id="bad", hyperparameters={}, complexity=1.0)]

    def fit_predict_inner_fold(self, train_data, valid_data, candidate, *, seed):
        raise RuntimeError("broken candidate")

    def fit_predict_outer_fold(self, train_data, test_data, candidate, *, seed):
        raise AssertionError("should not reach outer fit")


class ConstantAdapter:
    method_name = "constant"

    def candidate_specs(self) -> list[CandidateSpec]:
        return [CandidateSpec(candidate_id="ok", hyperparameters={}, complexity=1.0)]

    def fit_predict_inner_fold(self, train_data, valid_data, candidate, *, seed):
        return np.full(len(valid_data.y), 0.6, dtype=float)

    def fit_predict_outer_fold(self, train_data, test_data, candidate, *, seed):
        return np.full(len(test_data.y), 0.6, dtype=float)


def test_candidate_failures_are_recorded_and_all_failed_outer_fold_raises() -> None:
    dataset = make_grouped_dataset()
    partial = run_nested_subject_loso(
        dataset=dataset,
        adapter=PartialFailureAdapter(),
        evaluation=make_eval_config(),
    )

    failed = partial.outer_fold_results[0].candidate_results[0]
    assert failed.status == "failed"
    assert "ValueError" in str(failed.failure_reason)
    assert partial.outer_fold_results[0].selected_candidate_id == "good"

    with pytest.raises(RuntimeError, match="No valid candidates remain"):
        run_nested_subject_loso(
            dataset=dataset,
            adapter=AllFailureAdapter(),
            evaluation=make_eval_config(),
        )


def test_partial_outer_fold_mode_must_be_explicit(tmp_path: Path) -> None:
    dataset = make_grouped_dataset()
    manifest = build_subject_loso_manifest(
        dataset=dataset,
        evaluation=make_eval_config(),
        output_path=tmp_path / "fold_manifest.json",
    )
    manifest.outer_folds = manifest.outer_folds[:1]

    with pytest.raises(LeakageError, match="every subject exactly once|missing predictions"):
        run_nested_subject_loso(
            dataset=dataset,
            adapter=ConstantAdapter(),
            evaluation=make_eval_config(),
            manifest=manifest,
        )

    partial = run_nested_subject_loso(
        dataset=dataset,
        adapter=ConstantAdapter(),
        evaluation=make_eval_config(),
        manifest=manifest,
        allow_partial_outer_folds=True,
    )

    assert partial.outer_fold_results[0].outer_test_subject == manifest.outer_folds[0].outer_test_subject
    assert partial.aggregated_probabilities.shape == (3,)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_raw_cnn_preprocessing_uses_grouped_train_only_normalization() -> None:
    train_subject_ids = np.array(["a", "a", "b", "b", "c", "c"], dtype=str)
    train_X = np.zeros((6, 4, 30), dtype=float)
    for idx, value in enumerate((1.0, 2.0, 10.0, 11.0, 20.0, 21.0)):
        train_X[idx] = value
    train_data = NestedDataset(
        X=train_X,
        y=np.array([0, 1, 0, 1, 0, 1], dtype=int),
        subject_ids=train_subject_ids,
        sample_ids=[f"t{idx}" for idx in range(6)],
    )
    eval_data = NestedDataset(
        X=np.full((2, 4, 30), 100.0, dtype=float),
        y=np.array([0, 1], dtype=int),
        subject_ids=np.array(["z", "z"], dtype=str),
        sample_ids=["e0", "e1"],
    )

    prepared = _prepare_raw_cnn_arrays(train_data=train_data, eval_data=eval_data, seed=0)

    assert prepared.train_x.shape[1:] == (4, 30)
    assert prepared.valid_x.shape[1:] == (4, 30)
    assert prepared.eval_x.shape[1:] == (4, 30)
    assert set(prepared.train_subjects.tolist()).isdisjoint(prepared.valid_subjects.tolist())
    assert set(prepared.valid_subjects.tolist()) <= {"a", "b", "c"}
    assert set(prepared.train_subjects.tolist()) | set(prepared.valid_subjects.tolist()) == {"a", "b", "c"}

    train_subject_mask = np.isin(train_subject_ids, prepared.train_subjects)
    expected_mean = train_X[train_subject_mask].mean(axis=(0, 2))
    expected_std = train_X[train_subject_mask].std(axis=(0, 2))
    np.testing.assert_allclose(prepared.channel_mean, expected_mean)
    np.testing.assert_allclose(prepared.channel_std, np.where(expected_std == 0.0, 1.0, expected_std))


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_raw_cnn_adapter_runs_reduced_nested_smoke_with_manifest(tmp_path: Path) -> None:
    dataset = make_grouped_raw_dataset()
    manifest = build_subject_loso_manifest(
        dataset=dataset,
        evaluation=make_eval_config(),
        output_path=tmp_path / "fold_manifest.json",
    )
    manifest.outer_folds = manifest.outer_folds[:1]

    result = run_nested_subject_loso(
        dataset=dataset,
        adapter=Raw1DCNNAdapter(
            learning_rate_values=(1e-3,),
            dropout_values=(0.0,),
            epochs=3,
            patience=1,
        ),
        evaluation=make_eval_config(),
        manifest=manifest,
        allow_partial_outer_folds=True,
    )

    assert result.runtime_metadata["manifest_hash"] == manifest.manifest_hash
    assert result.outer_fold_results[0].candidate_results[0].status == "success"
    assert result.aggregated_probabilities.shape == (3,)
