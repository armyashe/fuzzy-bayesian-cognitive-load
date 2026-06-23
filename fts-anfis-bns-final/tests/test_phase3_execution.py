from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbn_pipeline.nested_evaluation import CandidateSpec, CandidateResult, NestedEvaluationResult, OuterFoldResult
from fbn_pipeline.publication_ablations import (
    build_frozen_proposed_configuration_map,
    compute_fts_summary_signal_evidence,
)


def make_proposed_result() -> NestedEvaluationResult:
    candidate = CandidateResult(
        method_name="proposed_fts_anfis_bn",
        candidate_id="proposed_gaussian_3",
        hyperparameters={"mf_type": "gaussian", "n_rules": 3},
        inner_oof_probabilities=np.array([0.4, 0.6], dtype=float),
        inner_oof_predictions=np.array([0, 1], dtype=int),
        inner_oof_metrics={"bal_acc": 0.75, "auc": 0.80},
        selected_threshold=0.4,
        model_complexity=3.0,
        status="success",
        failure_reason=None,
        outer_fold_index=0,
        failed_inner_fold_index=None,
    )
    outer = OuterFoldResult(
        outer_fold_index=0,
        outer_test_subject="s0",
        outer_training_subjects=["s1", "s2"],
        selected_candidate_id="proposed_gaussian_3",
        selected_hyperparameters={"mf_type": "gaussian", "n_rules": 3},
        selected_threshold=0.4,
        inner_selection_metrics={"bal_acc": 0.75, "auc": 0.80},
        outer_test_sample_ids=["a", "b"],
        outer_test_true_labels=np.array([0, 1], dtype=int),
        outer_test_probabilities=np.array([0.3, 0.7], dtype=float),
        outer_test_predictions=np.array([0, 1], dtype=int),
        seed=11,
        configuration_hash="cfg0",
        candidate_results=[candidate],
    )
    return NestedEvaluationResult(
        method_name="proposed_fts_anfis_bn",
        outer_fold_results=[outer],
        aggregated_sample_ids=["a", "b"],
        aggregated_subject_ids=["s0", "s0"],
        aggregated_labels=np.array([0, 1], dtype=int),
        aggregated_probabilities=np.array([0.3, 0.7], dtype=float),
        aggregated_predictions=np.array([0, 1], dtype=int),
        overall_metrics={"bal_acc": 1.0},
        per_subject_metrics=[],
        runtime_metadata={"manifest_hash": "m0", "dataset_order_hash": "d0"},
    )


def test_ablation_configuration_map_reuses_selected_outer_fold_configuration() -> None:
    proposed = make_proposed_result()
    frozen = build_frozen_proposed_configuration_map(proposed)

    assert frozen[0]["mf_type"] == "gaussian"
    assert frozen[0]["n_rules"] == 3
    assert frozen[0]["selected_threshold"] == 0.4
    assert frozen[0]["source_method"] == "proposed_fts_anfis_bn"


def test_fts_summary_bn_uses_fixed_transition_terms() -> None:
    features = np.array(
        [
            [0.1, 0.0, 0.2, 0.0, 0.1, 0.3, 0.0, 0.1, 0.4] * 4,
            [0.2, 0.0, 0.1, 0.0, 0.1, 0.2, 0.0, 0.1, 0.5] * 4,
        ],
        dtype=float,
    )
    evidence = compute_fts_summary_signal_evidence(features)

    assert evidence.shape == (2, 4)
    assert np.allclose(evidence[0], np.full(4, 0.3 + 0.4))
    assert np.allclose(evidence[1], np.full(4, 0.2 + 0.5))
