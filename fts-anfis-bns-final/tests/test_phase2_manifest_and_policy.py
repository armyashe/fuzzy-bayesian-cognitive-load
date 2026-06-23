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

from fbn_pipeline.fold_manifest import build_subject_loso_manifest, load_fold_manifest
from fbn_pipeline.nested_evaluation import CandidateSpec, NestedDataset, run_nested_subject_loso
from fbn_pipeline.production_data import load_full_subject_dataset
from fbn_pipeline.structured_config import EvaluationConfig


def make_eval_config() -> EvaluationConfig:
    return EvaluationConfig(
        outer_protocol="loso",
        inner_folds=2,
        selection_metric="balanced_accuracy",
        threshold_metric="balanced_accuracy",
        threshold_grid=(0.3, 0.4, 0.5, 0.6, 0.7),
        seed=42,
        deterministic_threads=1,
        output_directory="artifacts/phase2-tests",
    )


def make_grouped_dataset() -> NestedDataset:
    sample_ids = [f"s{i}" for i in range(8)]
    subject_ids = np.array(["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"])
    X = np.arange(8, dtype=float).reshape(-1, 1)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    return NestedDataset(X=X, y=y, subject_ids=subject_ids, sample_ids=sample_ids)


class ThresholdPolicyAdapter:
    method_name = "threshold_policy"

    def candidate_specs(self) -> list[CandidateSpec]:
        return [
            CandidateSpec(candidate_id="candidate_a", hyperparameters={}, complexity=1.0),
            CandidateSpec(candidate_id="candidate_b", hyperparameters={}, complexity=2.0),
        ]

    def fit_predict_inner_fold(self, train_data, valid_data, candidate, *, seed):
        mapping = {
            "candidate_a": {
                "s0": 0.34, "s1": 0.46, "s2": 0.34, "s3": 0.46,
                "s4": 0.34, "s5": 0.46, "s6": 0.34, "s7": 0.46,
            },
            "candidate_b": {
                "s0": 0.60, "s1": 0.55, "s2": 0.30, "s3": 0.55,
                "s4": 0.60, "s5": 0.55, "s6": 0.30, "s7": 0.55,
            },
        }
        return np.array([mapping[candidate.candidate_id][sid] for sid in valid_data.sample_ids], dtype=float)

    def fit_predict_outer_fold(self, train_data, test_data, candidate, *, seed):
        return np.full(len(test_data.y), 0.5, dtype=float)


def test_full_subject_dataset_loader_returns_all_23_subjects() -> None:
    dataset = load_full_subject_dataset(ROOT / "last_30s_segments")
    assert len(np.unique(dataset.subject_ids)) == 23
    assert len(dataset.sample_ids) == len(dataset.y)


def test_shared_manifest_persists_dataset_hash_and_fold_assignments(tmp_path: Path) -> None:
    manifest = build_subject_loso_manifest(
        dataset=make_grouped_dataset(),
        evaluation=make_eval_config(),
        output_path=tmp_path / "fold_manifest.json",
    )
    loaded = load_fold_manifest(tmp_path / "fold_manifest.json")

    assert loaded.dataset_order_hash == manifest.dataset_order_hash
    assert loaded.outer_folds[0].outer_test_subject == manifest.outer_folds[0].outer_test_subject
    assert loaded.outer_folds[0].inner_folds[0].valid_sample_ids == manifest.outer_folds[0].inner_folds[0].valid_sample_ids


def test_candidate_ranking_uses_threshold_optimized_inner_oof_balanced_accuracy(tmp_path: Path) -> None:
    dataset = make_grouped_dataset()
    manifest = build_subject_loso_manifest(
        dataset=dataset,
        evaluation=make_eval_config(),
        output_path=tmp_path / "fold_manifest.json",
    )
    result = run_nested_subject_loso(
        dataset=dataset,
        adapter=ThresholdPolicyAdapter(),
        evaluation=make_eval_config(),
        manifest=manifest,
    )

    assert result.outer_fold_results[0].selected_candidate_id == "candidate_a"
    assert result.outer_fold_results[0].selected_threshold == pytest.approx(0.4, abs=1e-8)
    assert result.runtime_metadata["manifest_hash"] == manifest.manifest_hash


def test_manifest_outer_indices_use_stable_sample_indices_not_local_positions(tmp_path: Path) -> None:
    base = make_grouped_dataset()
    subset = NestedDataset(
        X=base.X[:6].copy(),
        y=base.y[:6].copy(),
        subject_ids=base.subject_ids[:6].copy(),
        sample_ids=list(base.sample_ids[:6]),
        sample_indices=np.array([10, 11, 12, 13, 14, 15], dtype=int),
    )
    manifest = build_subject_loso_manifest(
        dataset=subset,
        evaluation=make_eval_config(),
        output_path=tmp_path / "fold_manifest.json",
    )
    result = run_nested_subject_loso(
        dataset=subset,
        adapter=ThresholdPolicyAdapter(),
        evaluation=make_eval_config(),
        manifest=manifest,
    )

    assert len(result.outer_fold_results) == len(np.unique(subset.subject_ids))
