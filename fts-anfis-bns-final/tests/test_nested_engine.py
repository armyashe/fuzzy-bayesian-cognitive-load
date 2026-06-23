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

from fbn_pipeline.nested_adapters import LogisticRegressionAdapter
from fbn_pipeline.nested_evaluation import (
    LeakageError,
    NestedDataset,
    assert_no_subject_overlap,
    run_nested_subject_loso,
)
from fbn_pipeline.nested_results import load_nested_evaluation_result, save_nested_evaluation_result
from fbn_pipeline.structured_config import EvaluationConfig


def make_grouped_dataset(n_subjects: int = 6, samples_per_subject: int = 4) -> NestedDataset:
    rng = np.random.default_rng(42)
    total = n_subjects * samples_per_subject
    subject_ids = np.repeat([f"s{i}" for i in range(n_subjects)], samples_per_subject)
    sample_ids = [f"sample_{i}" for i in range(total)]
    X = rng.normal(size=(total, 5))
    subject_signal = np.repeat(np.linspace(-1.0, 1.0, n_subjects), samples_per_subject)
    within_subject_pattern = np.tile(np.array([-0.8, 0.8, -0.6, 0.6]), n_subjects)
    X[:, 0] += 0.4 * subject_signal + within_subject_pattern
    y = (X[:, 0] + 0.15 * X[:, 1] > 0.0).astype(int)
    return NestedDataset(X=X, y=y, subject_ids=subject_ids, sample_ids=sample_ids)


def make_eval_config() -> EvaluationConfig:
    return EvaluationConfig(
        outer_protocol="loso",
        inner_folds=5,
        selection_metric="balanced_accuracy",
        threshold_metric="balanced_accuracy",
        threshold_grid=(0.3, 0.4, 0.5, 0.6, 0.7),
        seed=42,
        deterministic_threads=1,
        output_directory="artifacts/nested-tests",
    )


def test_each_subject_is_outer_test_once_and_each_sample_gets_one_prediction() -> None:
    result = run_nested_subject_loso(
        dataset=make_grouped_dataset(),
        adapter=LogisticRegressionAdapter(candidate_cs=(0.1, 1.0)),
        evaluation=make_eval_config(),
    )

    outer_subjects = [fold.outer_test_subject for fold in result.outer_fold_results]
    assert sorted(outer_subjects) == sorted(set(outer_subjects))
    assert len(result.aggregated_sample_ids) == len(set(result.aggregated_sample_ids))
    assert len(result.aggregated_sample_ids) == len(result.aggregated_predictions)


def test_outer_test_subjects_never_enter_inner_training_subjects() -> None:
    result = run_nested_subject_loso(
        dataset=make_grouped_dataset(),
        adapter=LogisticRegressionAdapter(candidate_cs=(0.1, 1.0)),
        evaluation=make_eval_config(),
    )

    for fold in result.outer_fold_results:
        assert fold.outer_test_subject not in fold.outer_training_subjects


def test_changing_outer_test_labels_does_not_change_selected_candidate() -> None:
    dataset = make_grouped_dataset()
    adapter = LogisticRegressionAdapter(candidate_cs=(0.1, 1.0, 10.0))
    baseline = run_nested_subject_loso(dataset=dataset, adapter=adapter, evaluation=make_eval_config())

    for fold in baseline.outer_fold_results:
        flipped_y = dataset.y.copy()
        mask = dataset.subject_ids == fold.outer_test_subject
        flipped_y[mask] = 1 - flipped_y[mask]
        flipped = NestedDataset(
            X=dataset.X.copy(),
            y=flipped_y,
            subject_ids=dataset.subject_ids.copy(),
            sample_ids=list(dataset.sample_ids),
        )
        rerun = run_nested_subject_loso(dataset=flipped, adapter=adapter, evaluation=make_eval_config())
        rerun_fold = rerun.outer_fold_results[fold.outer_fold_index]
        assert rerun_fold.selected_candidate_id == fold.selected_candidate_id
        assert rerun_fold.selected_threshold == fold.selected_threshold


def test_deliberate_leakage_raises() -> None:
    with pytest.raises(LeakageError):
        assert_no_subject_overlap(["s1", "s2"], ["s2"], "inner")


def test_repeated_runs_with_same_seed_are_deterministic_and_roundtrip(tmp_path: Path) -> None:
    dataset = make_grouped_dataset()
    adapter = LogisticRegressionAdapter(candidate_cs=(0.1, 1.0))
    result_a = run_nested_subject_loso(dataset=dataset, adapter=adapter, evaluation=make_eval_config())
    result_b = run_nested_subject_loso(dataset=dataset, adapter=adapter, evaluation=make_eval_config())

    assert [fold.selected_candidate_id for fold in result_a.outer_fold_results] == [
        fold.selected_candidate_id for fold in result_b.outer_fold_results
    ]
    np.testing.assert_allclose(result_a.aggregated_probabilities, result_b.aggregated_probabilities)

    save_nested_evaluation_result(result_a, tmp_path)
    loaded = load_nested_evaluation_result(tmp_path)
    assert loaded.aggregated_sample_ids == result_a.aggregated_sample_ids
    np.testing.assert_array_equal(loaded.aggregated_predictions, result_a.aggregated_predictions)
