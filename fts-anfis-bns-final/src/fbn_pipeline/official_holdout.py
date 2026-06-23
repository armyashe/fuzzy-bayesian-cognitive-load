from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.model_selection import GroupKFold

from .data import TEST_IDS, TRAIN_IDS
from .metrics import per_subject_metrics
from .nested_evaluation import (
    EXPECTED_CANDIDATE_ERRORS,
    CandidateSpec,
    LeakageError,
    NestedDataset,
    _compute_metrics_from_predictions,
    _select_candidate,
    _select_threshold,
    assert_no_subject_overlap,
)
from .nested_results import CandidateResult
from .production_data import load_full_subject_dataset
from .structured_config import EvaluationConfig, validate_candidate_ids


OFFICIAL_PROTOCOL = "official_holdout_groupkfold"
OFFICIAL_SMOKE_PROTOCOL = "official_holdout_smoke"
OFFICIAL_TRAIN_SUBJECT_COUNT = 18
OFFICIAL_TEST_SUBJECT_COUNT = 5
OFFICIAL_TRAIN_SAMPLE_COUNT = 632
OFFICIAL_TEST_SAMPLE_COUNT = 193
OFFICIAL_TOTAL_SAMPLE_COUNT = 825


@dataclass(slots=True)
class OfficialHoldoutSplit:
    train_dataset: NestedDataset
    test_dataset: NestedDataset
    all_dataset: NestedDataset
    train_subject_ids: list[str]
    test_subject_ids: list[str]
    protocol_name: str


@dataclass(slots=True)
class GroupFoldManifest:
    fold_index: int
    train_indices: list[int]
    valid_indices: list[int]
    train_subjects: list[str]
    valid_subjects: list[str]
    train_sample_ids: list[str]
    valid_sample_ids: list[str]
    seed: int


@dataclass(slots=True)
class OfficialHoldoutManifest:
    protocol_name: str
    dataset_order_hash: str
    train_subject_ids: list[str]
    test_subject_ids: list[str]
    train_sample_ids: list[str]
    test_sample_ids: list[str]
    train_sample_count: int
    test_sample_count: int
    inner_folds: int
    seed: int
    folds: list[GroupFoldManifest]
    manifest_hash: str = ""

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        if not data["manifest_hash"]:
            data["manifest_hash"] = compute_manifest_hash(data)
        return data


@dataclass(slots=True)
class OfficialHoldoutResult:
    method_name: str
    protocol_name: str
    selected_candidate_id: str
    selected_hyperparameters: dict[str, object]
    selected_threshold: float
    candidate_results: list[CandidateResult]
    train_oof_sample_ids: list[str]
    train_oof_subject_ids: list[str]
    train_oof_labels: np.ndarray
    train_oof_probabilities: np.ndarray
    train_oof_predictions: np.ndarray
    final_test_sample_ids: list[str]
    final_test_subject_ids: list[str]
    final_test_labels: np.ndarray
    final_test_probabilities: np.ndarray
    final_test_predictions: np.ndarray
    overall_test_metrics: dict[str, float]
    per_subject_test_metrics: list[dict[str, object]]
    runtime_metadata: dict[str, object]


def load_official_holdout_split(
    data_dir: Path | str,
    *,
    protocol_name: str = OFFICIAL_PROTOCOL,
    train_subject_limit: int | None = None,
    test_subject_limit: int | None = None,
) -> OfficialHoldoutSplit:
    all_dataset = load_full_subject_dataset(data_dir)
    train_subject_ids = list(TRAIN_IDS)
    test_subject_ids = list(TEST_IDS)
    if train_subject_limit is not None:
        train_subject_ids = train_subject_ids[: int(train_subject_limit)]
    if test_subject_limit is not None:
        test_subject_ids = test_subject_ids[: int(test_subject_limit)]

    train_indices = np.array(
        [idx for idx, subject_id in enumerate(all_dataset.subject_ids.astype(str).tolist()) if subject_id in set(train_subject_ids)],
        dtype=int,
    )
    test_indices = np.array(
        [idx for idx, subject_id in enumerate(all_dataset.subject_ids.astype(str).tolist()) if subject_id in set(test_subject_ids)],
        dtype=int,
    )
    train_dataset = all_dataset.subset(train_indices)
    test_dataset = all_dataset.subset(test_indices)

    split = OfficialHoldoutSplit(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        all_dataset=all_dataset,
        train_subject_ids=sorted(np.unique(train_dataset.subject_ids.astype(str)).tolist()),
        test_subject_ids=sorted(np.unique(test_dataset.subject_ids.astype(str)).tolist()),
        protocol_name=protocol_name,
    )
    validate_official_holdout_split(
        split,
        strict_counts=protocol_name == OFFICIAL_PROTOCOL,
    )
    return split


def validate_official_holdout_split(split: OfficialHoldoutSplit, *, strict_counts: bool) -> None:
    overlap = sorted(set(split.train_subject_ids) & set(split.test_subject_ids))
    if overlap:
        raise ValueError(f"Official train/test subjects overlap: {overlap}")
    if strict_counts:
        if len(split.train_subject_ids) != OFFICIAL_TRAIN_SUBJECT_COUNT:
            raise ValueError(f"Expected {OFFICIAL_TRAIN_SUBJECT_COUNT} training subjects, got {len(split.train_subject_ids)}")
        if len(split.test_subject_ids) != OFFICIAL_TEST_SUBJECT_COUNT:
            raise ValueError(f"Expected {OFFICIAL_TEST_SUBJECT_COUNT} test subjects, got {len(split.test_subject_ids)}")
        if len(split.train_dataset.y) != OFFICIAL_TRAIN_SAMPLE_COUNT:
            raise ValueError(f"Expected {OFFICIAL_TRAIN_SAMPLE_COUNT} training samples, got {len(split.train_dataset.y)}")
        if len(split.test_dataset.y) != OFFICIAL_TEST_SAMPLE_COUNT:
            raise ValueError(f"Expected {OFFICIAL_TEST_SAMPLE_COUNT} test samples, got {len(split.test_dataset.y)}")
        if len(split.all_dataset.y) != OFFICIAL_TOTAL_SAMPLE_COUNT:
            raise ValueError(f"Expected {OFFICIAL_TOTAL_SAMPLE_COUNT} total samples, got {len(split.all_dataset.y)}")


def build_official_group_manifest(
    *,
    split: OfficialHoldoutSplit,
    evaluation: EvaluationConfig,
    n_splits: int = 5,
    protocol_name: str | None = None,
    output_path: Path | str | None = None,
) -> OfficialHoldoutManifest:
    protocol = protocol_name or split.protocol_name
    train_dataset = split.train_dataset
    dataset_hash = compute_dataset_order_hash(train_dataset)
    splitter = GroupKFold(n_splits=n_splits)
    folds: list[GroupFoldManifest] = []
    for fold_index, (tr_idx, va_idx) in enumerate(
        splitter.split(train_dataset.X, train_dataset.y, train_dataset.subject_ids.astype(str))
    ):
        train_data = train_dataset.subset(np.asarray(tr_idx, dtype=int))
        valid_data = train_dataset.subset(np.asarray(va_idx, dtype=int))
        assert_no_subject_overlap(train_data.subject_ids, valid_data.subject_ids, "official holdout groupkfold")
        assert_no_subject_overlap(train_data.subject_ids, split.test_subject_ids, "official holdout train-vs-test")
        assert_no_subject_overlap(valid_data.subject_ids, split.test_subject_ids, "official holdout valid-vs-test")
        folds.append(
            GroupFoldManifest(
                fold_index=fold_index,
                train_indices=train_data.sample_indices.astype(int).tolist(),
                valid_indices=valid_data.sample_indices.astype(int).tolist(),
                train_subjects=sorted(np.unique(train_data.subject_ids.astype(str)).tolist()),
                valid_subjects=sorted(np.unique(valid_data.subject_ids.astype(str)).tolist()),
                train_sample_ids=list(train_data.sample_ids),
                valid_sample_ids=list(valid_data.sample_ids),
                seed=evaluation.seed + fold_index,
            )
        )

    manifest = OfficialHoldoutManifest(
        protocol_name=protocol,
        dataset_order_hash=dataset_hash,
        train_subject_ids=list(split.train_subject_ids),
        test_subject_ids=list(split.test_subject_ids),
        train_sample_ids=list(train_dataset.sample_ids),
        test_sample_ids=list(split.test_dataset.sample_ids),
        train_sample_count=len(train_dataset.y),
        test_sample_count=len(split.test_dataset.y),
        inner_folds=n_splits,
        seed=evaluation.seed,
        folds=folds,
    )
    manifest.manifest_hash = compute_manifest_hash(manifest.to_dict())
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def run_official_holdout_evaluation(
    *,
    split: OfficialHoldoutSplit,
    adapter,
    evaluation: EvaluationConfig,
    manifest: OfficialHoldoutManifest,
) -> OfficialHoldoutResult:
    train_dataset = split.train_dataset
    test_dataset = split.test_dataset
    candidate_specs = adapter.candidate_specs()
    validate_candidate_ids(tuple(spec.candidate_id for spec in candidate_specs))
    if not candidate_specs:
        raise ValueError("At least one candidate is required")

    sample_index_to_train_local = {
        int(sample_index): local_index
        for local_index, sample_index in enumerate(train_dataset.sample_indices.tolist())
    }
    candidate_results: list[dict[str, object]] = []

    for candidate_rank, spec in enumerate(candidate_specs):
        oof_probabilities = np.full(len(train_dataset.y), np.nan, dtype=float)
        prediction_counts = np.zeros(len(train_dataset.y), dtype=int)
        failure_reason: str | None = None
        failed_fold_index: int | None = None
        try:
            for fold in manifest.folds:
                if hasattr(adapter, "set_inner_fold_context"):
                    adapter.set_inner_fold_context(fold.fold_index, None)
                tr_idx = np.array([sample_index_to_train_local[int(i)] for i in fold.train_indices], dtype=int)
                va_idx = np.array([sample_index_to_train_local[int(i)] for i in fold.valid_indices], dtype=int)
                train_data = train_dataset.subset(tr_idx)
                valid_data = train_dataset.subset(va_idx)
                assert_no_subject_overlap(train_data.subject_ids, valid_data.subject_ids, "official holdout inner split")
                assert_no_subject_overlap(train_data.subject_ids, split.test_subject_ids, "official holdout train-vs-test")
                assert_no_subject_overlap(valid_data.subject_ids, split.test_subject_ids, "official holdout valid-vs-test")
                valid_probabilities = np.asarray(
                    adapter.fit_predict_inner_fold(
                        train_data,
                        valid_data,
                        spec,
                        seed=fold.seed + candidate_rank,
                    ),
                    dtype=float,
                )
                if len(valid_probabilities) != len(va_idx):
                    raise LeakageError("Official holdout validation prediction length does not match validation samples")
                oof_probabilities[va_idx] = valid_probabilities
                prediction_counts[va_idx] += 1
            if np.any(np.isnan(oof_probabilities)):
                raise LeakageError("A training OOF sample received zero predictions")
            if np.any(prediction_counts != 1):
                raise LeakageError("Each training OOF sample must receive exactly one prediction")
        except EXPECTED_CANDIDATE_ERRORS as exc:
            failure_reason = f"{type(exc).__name__}: {exc}"
            failed_fold_index = fold.fold_index if "fold" in locals() else None

        if failure_reason is not None:
            result = CandidateResult(
                method_name=adapter.method_name,
                candidate_id=spec.candidate_id,
                hyperparameters=dict(spec.hyperparameters),
                inner_oof_probabilities=np.array([], dtype=float),
                inner_oof_predictions=np.array([], dtype=int),
                inner_oof_metrics={"bal_acc": float("nan"), "auc": float("nan")},
                selected_threshold=float("nan"),
                model_complexity=float(spec.complexity),
                status="failed",
                failure_reason=failure_reason,
                outer_fold_index=None,
                failed_inner_fold_index=failed_fold_index,
            )
            candidate_results.append({"spec": spec, "threshold": float("nan"), "metrics": result.inner_oof_metrics, "result": result})
            continue

        threshold = _select_threshold(train_dataset.y, oof_probabilities, evaluation.threshold_grid)
        ranking_metrics = _compute_metrics_from_predictions(
            train_dataset.y,
            oof_probabilities,
            (oof_probabilities >= threshold).astype(int),
        )
        result = CandidateResult(
            method_name=adapter.method_name,
            candidate_id=spec.candidate_id,
            hyperparameters=dict(spec.hyperparameters),
            inner_oof_probabilities=oof_probabilities,
            inner_oof_predictions=(oof_probabilities >= threshold).astype(int),
            inner_oof_metrics=ranking_metrics,
            selected_threshold=float(threshold),
            model_complexity=float(spec.complexity),
            status="success",
            failure_reason=None,
            outer_fold_index=None,
            failed_inner_fold_index=None,
        )
        candidate_results.append({"spec": spec, "threshold": threshold, "metrics": ranking_metrics, "result": result})

    selected = _select_candidate(candidate_results)
    if hasattr(adapter, "set_outer_fold_context"):
        adapter.set_outer_fold_context(SimpleNamespace(outer_fold_index=0))
    final_probabilities = np.asarray(
        adapter.fit_predict_outer_fold(
            train_dataset,
            test_dataset,
            selected["spec"],
            seed=evaluation.seed,
        ),
        dtype=float,
    )
    if len(final_probabilities) != len(test_dataset.y):
        raise LeakageError("Official holdout final prediction length does not match test samples")
    final_predictions = (final_probabilities >= selected["threshold"]).astype(int)
    overall_metrics = _compute_metrics_from_predictions(test_dataset.y, final_probabilities, final_predictions)
    return OfficialHoldoutResult(
        method_name=adapter.method_name,
        protocol_name=manifest.protocol_name,
        selected_candidate_id=selected["spec"].candidate_id,
        selected_hyperparameters=dict(selected["spec"].hyperparameters),
        selected_threshold=float(selected["threshold"]),
        candidate_results=[item["result"] for item in candidate_results],
        train_oof_sample_ids=list(train_dataset.sample_ids),
        train_oof_subject_ids=train_dataset.subject_ids.astype(str).tolist(),
        train_oof_labels=train_dataset.y.copy(),
        train_oof_probabilities=np.asarray(selected["result"].inner_oof_probabilities),
        train_oof_predictions=np.asarray(selected["result"].inner_oof_predictions),
        final_test_sample_ids=list(test_dataset.sample_ids),
        final_test_subject_ids=test_dataset.subject_ids.astype(str).tolist(),
        final_test_labels=test_dataset.y.copy(),
        final_test_probabilities=final_probabilities,
        final_test_predictions=final_predictions,
        overall_test_metrics=overall_metrics,
        per_subject_test_metrics=per_subject_metrics(
            test_dataset.y,
            final_predictions,
            final_probabilities,
            test_dataset.subject_ids,
        ),
        runtime_metadata={
            "manifest_hash": manifest.manifest_hash,
            "dataset_order_hash": manifest.dataset_order_hash,
            "protocol_name": manifest.protocol_name,
        },
    )


def compute_dataset_order_hash(dataset: NestedDataset) -> str:
    payload = [
        {
            "sample_index": int(dataset.sample_indices[i]),
            "sample_id": dataset.sample_ids[i],
            "subject_id": str(dataset.subject_ids[i]),
            "label": int(dataset.y[i]),
        }
        for i in range(len(dataset.y))
    ]
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_manifest_hash(data: dict[str, object]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
