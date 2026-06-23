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

from fbn_pipeline.nested_results import (
    CandidateResult,
    NestedEvaluationResult,
    OuterFoldResult,
    load_nested_evaluation_result,
    save_nested_evaluation_result,
)
from fbn_pipeline.structured_config import EvaluationConfig
from fbn_pipeline.structured_config import validate_candidate_ids


def test_evaluation_config_rejects_inner_folds_above_available_groups() -> None:
    cfg = EvaluationConfig(
        outer_protocol="loso",
        inner_folds=5,
        selection_metric="balanced_accuracy",
        threshold_metric="balanced_accuracy",
        threshold_grid=(0.3, 0.5, 0.7),
        seed=42,
        deterministic_threads=1,
        output_directory="artifacts/nested",
    )
    with pytest.raises(ValueError):
        cfg.validate_available_groups(["s1", "s2", "s3", "s4"])


def test_nested_result_roundtrip_preserves_sample_order(tmp_path: Path) -> None:
    candidate = CandidateResult(
        method_name="dummy",
        candidate_id="c1",
        hyperparameters={"strength": 1.0},
        inner_oof_probabilities=np.array([0.1, 0.9]),
        inner_oof_predictions=np.array([0, 1]),
        inner_oof_metrics={"bal_acc": 1.0},
        selected_threshold=0.5,
        model_complexity=1.0,
        status="ok",
        failure_reason=None,
    )
    outer = OuterFoldResult(
        outer_fold_index=0,
        outer_test_subject="s3",
        outer_training_subjects=["s1", "s2"],
        selected_candidate_id="c1",
        selected_hyperparameters={"strength": 1.0},
        selected_threshold=0.5,
        inner_selection_metrics={"bal_acc": 1.0},
        outer_test_sample_ids=["a", "b"],
        outer_test_true_labels=np.array([0, 1]),
        outer_test_probabilities=np.array([0.2, 0.8]),
        outer_test_predictions=np.array([0, 1]),
        seed=42,
        configuration_hash="abc123",
        candidate_results=[candidate],
    )
    result = NestedEvaluationResult(
        method_name="dummy",
        outer_fold_results=[outer],
        aggregated_sample_ids=["a", "b"],
        aggregated_subject_ids=["s3", "s3"],
        aggregated_labels=np.array([0, 1]),
        aggregated_probabilities=np.array([0.2, 0.8]),
        aggregated_predictions=np.array([0, 1]),
        overall_metrics={"bal_acc": 1.0},
        per_subject_metrics=[{"subject": "s3", "bal_acc": 1.0}],
    )

    save_nested_evaluation_result(result, tmp_path)
    loaded = load_nested_evaluation_result(tmp_path)

    assert loaded.aggregated_sample_ids == ["a", "b"]
    assert loaded.aggregated_subject_ids == ["s3", "s3"]
    np.testing.assert_array_equal(loaded.aggregated_labels, np.array([0, 1]))
    np.testing.assert_array_equal(loaded.aggregated_probabilities, np.array([0.2, 0.8]))
    np.testing.assert_array_equal(loaded.aggregated_predictions, np.array([0, 1]))


def test_duplicate_candidate_ids_are_rejected() -> None:
    with pytest.raises(ValueError):
        validate_candidate_ids(("c1", "c1"))
