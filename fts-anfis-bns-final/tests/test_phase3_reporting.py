from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbn_pipeline.publication_artifacts import (
    create_run_layout,
    validate_official_holdout_predictions,
    validate_publication_predictions,
)
from fbn_pipeline.publication_metrics import (
    compute_pooled_metrics_from_predictions,
    compute_subject_metric_summary,
    compute_per_subject_metrics_from_predictions,
)
from fbn_pipeline.publication_statistics import (
    compute_bootstrap_confidence_intervals,
    holm_correction,
    run_paired_wilcoxon_by_subject,
)


def make_prediction_frame() -> pd.DataFrame:
    rows = [
        {
            "method": "proposed",
            "outer_fold_index": 0,
            "subject_id": "s0",
            "sample_id": "s0_a",
            "true_label": 0,
            "score_or_probability": 0.10,
            "prediction": 0,
            "threshold": 0.50,
            "seed": 7,
            "configuration_hash": "cfg0",
            "dataset_order_hash": "data0",
        },
        {
            "method": "proposed",
            "outer_fold_index": 0,
            "subject_id": "s0",
            "sample_id": "s0_b",
            "true_label": 1,
            "score_or_probability": 0.80,
            "prediction": 1,
            "threshold": 0.50,
            "seed": 7,
            "configuration_hash": "cfg0",
            "dataset_order_hash": "data0",
        },
        {
            "method": "proposed",
            "outer_fold_index": 1,
            "subject_id": "s1",
            "sample_id": "s1_a",
            "true_label": 1,
            "score_or_probability": 0.60,
            "prediction": 1,
            "threshold": 0.50,
            "seed": 7,
            "configuration_hash": "cfg1",
            "dataset_order_hash": "data0",
        },
        {
            "method": "proposed",
            "outer_fold_index": 1,
            "subject_id": "s1",
            "sample_id": "s1_b",
            "true_label": 1,
            "score_or_probability": 0.70,
            "prediction": 1,
            "threshold": 0.50,
            "seed": 7,
            "configuration_hash": "cfg1",
            "dataset_order_hash": "data0",
        },
    ]
    return pd.DataFrame(rows)


def test_create_run_layout_requires_explicit_overwrite(tmp_path: Path) -> None:
    layout = create_run_layout(
        output_root=tmp_path / "artifacts" / "phase3_publication",
        run_name="nested_full_v1",
        mode="publication",
        overwrite=False,
    )
    (layout.metadata_dir / "runtime_metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite"):
        create_run_layout(
            output_root=tmp_path / "artifacts" / "phase3_publication",
            run_name="nested_full_v1",
            mode="publication",
            overwrite=False,
        )


def test_pooled_and_per_subject_metrics_keep_subject_alignment_and_track_undefined_auc() -> None:
    frame = make_prediction_frame()

    pooled = compute_pooled_metrics_from_predictions(frame)
    per_subject = compute_per_subject_metrics_from_predictions(frame)
    summary = compute_subject_metric_summary(per_subject)

    assert pooled["balanced_accuracy"] == pytest.approx(1.0, abs=1e-8)
    assert pooled["confusion_matrix"] == [[1, 0], [0, 3]]
    assert per_subject.loc[per_subject["subject_id"] == "s1", "roc_auc"].isna().item()
    assert per_subject.loc[per_subject["subject_id"] == "s1", "roc_auc_undefined_reason"].item() == "single_class_true_labels"
    assert set(summary["statistic"]) >= {"mean", "std", "median", "iqr", "min", "max"}


def test_subject_cluster_bootstrap_resamples_subjects_not_rows() -> None:
    frame = make_prediction_frame()
    intervals = compute_bootstrap_confidence_intervals(
        frame,
        metric_names=("balanced_accuracy", "accuracy"),
        n_bootstrap=25,
        seed=11,
        method="proposed",
    )

    assert set(intervals["metric"]) == {"balanced_accuracy", "accuracy"}
    assert intervals["bootstrap_repetitions"].nunique() == 1
    assert intervals["resample_unit"].nunique() == 1
    assert intervals["resample_unit"].iloc[0] == "subject_cluster"


def test_paired_wilcoxon_aligns_by_subject_and_reports_effect_size() -> None:
    subject_metrics = pd.DataFrame(
        [
            {"method": "proposed", "subject_id": "s0", "balanced_accuracy": 1.0},
            {"method": "proposed", "subject_id": "s1", "balanced_accuracy": 0.8},
            {"method": "baseline", "subject_id": "s0", "balanced_accuracy": 0.5},
            {"method": "baseline", "subject_id": "s1", "balanced_accuracy": 0.8},
        ]
    )

    result = run_paired_wilcoxon_by_subject(
        subject_metrics,
        method_a="proposed",
        method_b="baseline",
        metric="balanced_accuracy",
    )

    assert result["paired_subject_count"] == 2
    assert result["zero_difference_count"] == 1
    assert "effect_size_rank_biserial" in result
    corrected = holm_correction([0.01, 0.03, 0.20])
    assert corrected == sorted(corrected)


def test_validate_publication_predictions_rejects_incomplete_coverage_by_default() -> None:
    frame = make_prediction_frame().iloc[:2].copy()

    with pytest.raises(ValueError, match="complete coverage"):
        validate_publication_predictions(
            frame,
            expected_subject_ids=["s0", "s1"],
            expected_sample_ids=["s0_a", "s0_b", "s1_a", "s1_b"],
            mode="publication",
            allow_partial_outer_folds=False,
        )

    validate_publication_predictions(
        frame,
        expected_subject_ids=["s0", "s1"],
        expected_sample_ids=["s0_a", "s0_b", "s1_a", "s1_b"],
        mode="smoke",
        allow_partial_outer_folds=True,
    )


def test_validate_official_holdout_predictions_requires_shared_complete_test_rows() -> None:
    frame = pd.DataFrame(
        [
            {
                "method": "proposed",
                "subject_id": "te0",
                "sample_id": "test_0",
                "prediction_role": "official_test",
            },
            {
                "method": "baseline",
                "subject_id": "te0",
                "sample_id": "test_0",
                "prediction_role": "official_test",
            },
        ]
    )

    validate_official_holdout_predictions(
        frame,
        expected_train_sample_ids=["train_0"],
        expected_test_sample_ids=["test_0"],
        expected_test_subject_ids=["te0"],
        mode="publication",
        protocol="official_holdout_groupkfold",
        complete_coverage=True,
    )

    with pytest.raises(ValueError, match="exactly 2 final predictions per method|complete coverage"):
        validate_official_holdout_predictions(
            frame.iloc[:1].copy(),
            expected_train_sample_ids=["train_0"],
            expected_test_sample_ids=["test_0", "test_1"],
            expected_test_subject_ids=["te0", "te1"],
            mode="publication",
            protocol="official_holdout_groupkfold",
            complete_coverage=True,
        )
