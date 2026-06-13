from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from .config import ExperimentConfig


@dataclass(slots=True)
class TrainedModels:
    """Fitted model objects and the per-signal evidence used at test time."""

    fts: Any | None
    bn: Any | None
    anfis_list: list[Any] | None
    test_evidence: np.ndarray
    train_evidence: np.ndarray | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the legacy model dict shape used by dashboard/reporting code."""
        return {
            "fts": self.fts,
            "bn": self.bn,
            "anfis_list": self.anfis_list,
            "te_probs": self.test_evidence,
            "tr_probs": self.train_evidence,
        }


@dataclass(slots=True)
class CvResult:
    oof_post: np.ndarray
    oof_y: np.ndarray
    fold_indices: list[tuple[np.ndarray, np.ndarray]]

    def __iter__(self) -> Iterator[Any]:
        yield self.oof_post
        yield self.oof_y
        yield self.fold_indices


@dataclass(slots=True)
class TestResult:
    y_pred: np.ndarray
    posterior: np.ndarray
    metrics: dict[str, float]
    models: TrainedModels

    def __iter__(self) -> Iterator[Any]:
        yield self.y_pred
        yield self.posterior
        yield self.metrics
        yield self.models.as_dict()


@dataclass(slots=True)
class RunResult:
    config: ExperimentConfig
    dataset: Any
    fold_indices: list[tuple[np.ndarray, np.ndarray]]
    best_threshold: float
    threshold_score: float
    oof_post: np.ndarray
    oof_pred: np.ndarray
    oof_metrics: dict[str, float]
    test_post: np.ndarray
    test_pred: np.ndarray
    test_metrics: dict[str, float]
    models: TrainedModels
    computed_test_metrics: dict[str, float] | None = None
    reported_test_metrics: dict[str, float] | None = None
    report_source: str = "computed"
    reference_path: str = ""
    subject_rows: list[dict] | None = None
    artifact_dir: str | None = None
    run_name: str = ""
    artifact_results: dict[str, Any] | None = None
    pipeline_desc: str = ""
    preset_name: str = ""
    variant: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return the legacy dict shape used by notebooks and Streamlit."""
        return {
            "config": self.config,
            "dataset": self.dataset,
            "fold_indices": self.fold_indices,
            "best_threshold": self.best_threshold,
            "threshold_score": self.threshold_score,
            "oof_post": self.oof_post,
            "oof_pred": self.oof_pred,
            "oof_metrics": self.oof_metrics,
            "test_post": self.test_post,
            "test_pred": self.test_pred,
            "test_metrics": self.test_metrics,
            "computed_test_metrics": self.computed_test_metrics or self.test_metrics,
            "reported_test_metrics": self.reported_test_metrics or self.test_metrics,
            "report_source": self.report_source,
            "reference_path": self.reference_path,
            "subject_rows": self.subject_rows,
            "models": self.models.as_dict(),
            "artifact_dir": self.artifact_dir,
            "run_name": self.run_name,
            "artifact_results": self.artifact_results,
            "pipeline_desc": self.pipeline_desc,
            "preset_name": self.preset_name,
            "variant": self.variant,
        }
