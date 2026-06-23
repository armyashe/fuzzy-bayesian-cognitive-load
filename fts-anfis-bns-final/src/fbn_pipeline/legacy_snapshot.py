from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .ablation import Ablation
from .config import ExperimentConfig, apply_preset, assert_thesis_config, setup_seed
from .data import load_dataset
from .metrics import compute_oof_metrics, find_best_threshold
from .pipeline import retrain_and_test_result, run_cv_result
from .stages import extract_features_for_ablation


@dataclass(slots=True)
class LegacySnapshot:
    dataset_stats: dict[str, object]
    split_stats: dict[str, object]
    shapes: dict[str, list[int]]
    selected_threshold: float
    oof_metrics: dict[str, float]
    test_metrics: dict[str, float]
    rounded_posterior_hash: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _build_legacy_config() -> ExperimentConfig:
    cfg = ExperimentConfig(data_path="last_30s_segments")
    cfg = apply_preset(cfg, "THESIS_FINAL")
    cfg = cfg.merged(ablation=Ablation.FULL.value)
    assert_thesis_config(cfg)
    return cfg


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _rounded_hash(values: np.ndarray, decimals: int = 4) -> str:
    rounded = np.round(np.asarray(values, dtype=np.float64), decimals=decimals)
    return hashlib.sha256(np.ascontiguousarray(rounded).tobytes()).hexdigest()


def run_legacy_snapshot() -> dict[str, object]:
    project_root = _project_root()
    cfg = _build_legacy_config()
    dataset = load_dataset(project_root / cfg.data_path)

    setup_seed(cfg.seed)
    cv = run_cv_result(dataset.X_train, dataset.y_train, dataset.user_train, cfg)
    best_thr, _ = find_best_threshold(
        cv.oof_y,
        cv.oof_post,
        metric=cfg.threshold_metric,
        n_grid=cfg.threshold_grid,
    )
    oof_metrics = compute_oof_metrics(cv.oof_y, cv.oof_post, best_thr)

    setup_seed(cfg.seed)
    test = retrain_and_test_result(
        dataset.X_train,
        dataset.y_train,
        dataset.X_test,
        dataset.y_test,
        best_thr,
        cfg,
    )

    fts_train, _, _ = extract_features_for_ablation(dataset.X_train, dataset.X_test, cfg)

    full_subject_ids = np.concatenate([dataset.user_train, dataset.user_test])
    full_labels = np.concatenate([dataset.y_train, dataset.y_test])
    snapshot = LegacySnapshot(
        dataset_stats={
            "n_samples": int(len(full_labels)),
            "n_subjects": int(len(np.unique(full_subject_ids))),
            "class_counts": {
                "rest": int(np.sum(full_labels == 0)),
                "task": int(np.sum(full_labels == 1)),
            },
        },
        split_stats={
            "train_subjects": sorted(np.unique(dataset.user_train).tolist()),
            "test_subjects": sorted(np.unique(dataset.user_test).tolist()),
            "train_class_counts": {
                "rest": int(np.sum(dataset.y_train == 0)),
                "task": int(np.sum(dataset.y_train == 1)),
            },
            "test_class_counts": {
                "rest": int(np.sum(dataset.y_test == 0)),
                "task": int(np.sum(dataset.y_test == 1)),
            },
        },
        shapes={
            "full_dataset": [int(len(full_labels)), int(dataset.X_train.shape[1]), int(dataset.X_train.shape[2])],
            "train_X": list(dataset.X_train.shape),
            "test_X": list(dataset.X_test.shape),
            "train_y": list(dataset.y_train.shape),
            "test_y": list(dataset.y_test.shape),
            "fts_train": list(fts_train.shape),
            "oof_post": list(cv.oof_post.shape),
            "final_posterior": list(test.posterior.shape),
        },
        selected_threshold=float(best_thr),
        oof_metrics=dict(oof_metrics),
        test_metrics=dict(test.metrics),
        rounded_posterior_hash=_rounded_hash(test.posterior),
    )
    return snapshot.as_dict()


def run_legacy_snapshot_twice() -> dict[str, object]:
    first = run_legacy_snapshot()
    second = run_legacy_snapshot()
    if json.dumps(first, sort_keys=True) != json.dumps(second, sort_keys=True):
        raise RuntimeError("Legacy snapshot is not deterministic across identical runs.")
    return first
