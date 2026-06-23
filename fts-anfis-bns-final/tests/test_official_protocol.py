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

from fbn_pipeline.nested_evaluation import CandidateSpec, NestedDataset
from fbn_pipeline.official_holdout import (
    OFFICIAL_TEST_SAMPLE_COUNT,
    OFFICIAL_TEST_SUBJECT_COUNT,
    OFFICIAL_TRAIN_SAMPLE_COUNT,
    OFFICIAL_TRAIN_SUBJECT_COUNT,
    OFFICIAL_PROTOCOL,
    OfficialHoldoutSplit,
    build_official_group_manifest,
    load_official_holdout_split,
    run_official_holdout_evaluation,
)
from fbn_pipeline.structured_config import EvaluationConfig


def make_eval_config() -> EvaluationConfig:
    return EvaluationConfig(
        outer_protocol="loso",
        inner_folds=2,
        selection_metric="balanced_accuracy",
        threshold_metric="balanced_accuracy",
        threshold_grid=(0.3, 0.5, 0.7),
        seed=19,
        deterministic_threads=1,
        output_directory="artifacts/phase3-tests",
    )


def make_synthetic_official_split() -> OfficialHoldoutSplit:
    train_subject_ids = np.repeat(["tr0", "tr1", "tr2", "tr3"], 3)
    test_subject_ids = np.repeat(["te0", "te1"], 3)
    train_X = np.zeros((12, 4, 30), dtype=float)
    test_X = np.zeros((6, 4, 30), dtype=float)
    for i in range(12):
        train_X[i] = float(i + 1)
    for i in range(6):
        test_X[i] = float(i + 101)
    train_dataset = NestedDataset(
        X=train_X,
        y=np.array([0, 1, 0] * 4, dtype=int),
        subject_ids=train_subject_ids,
        sample_ids=[f"train_{i}" for i in range(12)],
    )
    test_dataset = NestedDataset(
        X=test_X,
        y=np.array([0, 1, 0, 1, 0, 1], dtype=int),
        subject_ids=test_subject_ids,
        sample_ids=[f"test_{i}" for i in range(6)],
    )
    all_dataset = NestedDataset(
        X=np.concatenate([train_X, test_X], axis=0),
        y=np.concatenate([train_dataset.y, test_dataset.y], axis=0),
        subject_ids=np.concatenate([train_subject_ids, test_subject_ids], axis=0),
        sample_ids=train_dataset.sample_ids + test_dataset.sample_ids,
    )
    return OfficialHoldoutSplit(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        all_dataset=all_dataset,
        train_subject_ids=sorted(np.unique(train_subject_ids).tolist()),
        test_subject_ids=sorted(np.unique(test_subject_ids).tolist()),
        protocol_name=OFFICIAL_PROTOCOL,
    )


class ConstantOfficialAdapter:
    method_name = "constant_official"

    def candidate_specs(self) -> list[CandidateSpec]:
        return [
            CandidateSpec(candidate_id="weak", hyperparameters={"bias": 0.4}, complexity=1.0),
            CandidateSpec(candidate_id="strong", hyperparameters={"bias": 0.6}, complexity=2.0),
        ]

    def fit_predict_inner_fold(self, train_data, valid_data, candidate, *, seed):
        return np.full(len(valid_data.y), float(candidate.hyperparameters["bias"]), dtype=float)

    def fit_predict_outer_fold(self, train_data, test_data, candidate, *, seed):
        return np.full(len(test_data.y), float(candidate.hyperparameters["bias"]), dtype=float)


def test_official_train_and_test_subjects_have_zero_overlap_and_expected_counts() -> None:
    split = load_official_holdout_split(ROOT / "last_30s_segments")

    assert len(split.train_subject_ids) == OFFICIAL_TRAIN_SUBJECT_COUNT
    assert len(split.test_subject_ids) == OFFICIAL_TEST_SUBJECT_COUNT
    assert len(split.train_dataset.y) == OFFICIAL_TRAIN_SAMPLE_COUNT
    assert len(split.test_dataset.y) == OFFICIAL_TEST_SAMPLE_COUNT
    assert set(split.train_subject_ids).isdisjoint(split.test_subject_ids)


def test_groupkfold_manifest_uses_only_official_training_subjects() -> None:
    split = load_official_holdout_split(ROOT / "last_30s_segments")
    manifest = build_official_group_manifest(split=split, evaluation=make_eval_config(), n_splits=5)

    assert manifest.protocol_name == OFFICIAL_PROTOCOL
    for fold in manifest.folds:
        assert set(fold.train_subjects).isdisjoint(split.test_subject_ids)
        assert set(fold.valid_subjects).isdisjoint(split.test_subject_ids)
        assert set(fold.train_subjects).isdisjoint(fold.valid_subjects)


def test_changing_official_test_labels_does_not_change_selection_or_probabilities() -> None:
    split = make_synthetic_official_split()
    manifest = build_official_group_manifest(split=split, evaluation=make_eval_config(), n_splits=2)
    adapter = ConstantOfficialAdapter()

    baseline = run_official_holdout_evaluation(
        split=split,
        adapter=adapter,
        evaluation=make_eval_config(),
        manifest=manifest,
    )

    flipped = make_synthetic_official_split()
    flipped.test_dataset.y = 1 - flipped.test_dataset.y
    rerun = run_official_holdout_evaluation(
        split=flipped,
        adapter=ConstantOfficialAdapter(),
        evaluation=make_eval_config(),
        manifest=manifest,
    )

    assert baseline.selected_candidate_id == rerun.selected_candidate_id
    assert baseline.selected_hyperparameters == rerun.selected_hyperparameters
    assert baseline.selected_threshold == rerun.selected_threshold
    np.testing.assert_allclose(baseline.final_test_probabilities, rerun.final_test_probabilities)


def test_official_protocol_final_predictions_use_only_test_samples_and_share_ids() -> None:
    split = load_official_holdout_split(ROOT / "last_30s_segments")
    manifest = build_official_group_manifest(split=split, evaluation=make_eval_config(), n_splits=5)

    result_a = run_official_holdout_evaluation(
        split=split,
        adapter=ConstantOfficialAdapter(),
        evaluation=make_eval_config(),
        manifest=manifest,
    )
    result_b = run_official_holdout_evaluation(
        split=split,
        adapter=ConstantOfficialAdapter(),
        evaluation=make_eval_config(),
        manifest=manifest,
    )

    assert len(result_a.final_test_sample_ids) == OFFICIAL_TEST_SAMPLE_COUNT
    assert set(result_a.final_test_sample_ids) == set(split.test_dataset.sample_ids)
    assert set(result_a.final_test_sample_ids).isdisjoint(split.train_dataset.sample_ids)
    assert result_a.final_test_sample_ids == result_b.final_test_sample_ids
