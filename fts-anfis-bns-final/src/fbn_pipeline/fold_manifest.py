from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import GroupKFold

from .structured_config import EvaluationConfig

if TYPE_CHECKING:
    from .nested_evaluation import NestedDataset


@dataclass(slots=True)
class InnerFoldManifest:
    inner_fold_index: int
    train_indices: list[int]
    valid_indices: list[int]
    train_subjects: list[str]
    valid_subjects: list[str]
    train_sample_ids: list[str]
    valid_sample_ids: list[str]


@dataclass(slots=True)
class OuterFoldManifest:
    outer_fold_index: int
    outer_test_subject: str
    outer_train_subjects: list[str]
    outer_test_indices: list[int]
    outer_train_indices: list[int]
    inner_folds: list[InnerFoldManifest]
    seed: int


@dataclass(slots=True)
class FoldManifest:
    dataset_order_hash: str
    sample_indices: list[int]
    sample_ids: list[str]
    subject_ids: list[str]
    labels: list[int]
    seed: int
    outer_protocol: str
    inner_folds: int
    outer_folds: list[OuterFoldManifest]
    manifest_hash: str = ""

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        if not data["manifest_hash"]:
            data["manifest_hash"] = compute_manifest_hash(data)
        return data


def build_subject_loso_manifest(
    *,
    dataset: NestedDataset,
    evaluation: EvaluationConfig,
    output_path: Path | str | None = None,
) -> FoldManifest:
    dataset_hash = compute_dataset_order_hash(dataset)
    unique_subjects = sorted(np.unique(dataset.subject_ids).tolist())
    outer_folds: list[OuterFoldManifest] = []

    for outer_fold_index, outer_subject in enumerate(unique_subjects):
        outer_test_indices = np.flatnonzero(dataset.subject_ids == outer_subject)
        outer_train_indices = np.flatnonzero(dataset.subject_ids != outer_subject)
        outer_test_sample_indices = dataset.sample_indices[outer_test_indices]
        outer_train_sample_indices = dataset.sample_indices[outer_train_indices]
        outer_train_subjects = sorted(np.unique(dataset.subject_ids[outer_train_indices]).tolist())
        _assert_no_subject_overlap(dataset.subject_ids[outer_train_indices], dataset.subject_ids[outer_test_indices], "manifest outer split")
        evaluation.validate_available_groups(outer_train_subjects)

        inner_fold_entries: list[InnerFoldManifest] = []
        outer_train_groups = dataset.subject_ids[outer_train_indices]
        gkf = GroupKFold(n_splits=evaluation.inner_folds)
        for inner_fold_index, (rel_train_idx, rel_valid_idx) in enumerate(
            gkf.split(dataset.X[outer_train_indices], dataset.y[outer_train_indices], outer_train_groups)
        ):
            abs_train_idx = outer_train_indices[np.asarray(rel_train_idx)]
            abs_valid_idx = outer_train_indices[np.asarray(rel_valid_idx)]
            train_sample_indices = dataset.sample_indices[abs_train_idx]
            valid_sample_indices = dataset.sample_indices[abs_valid_idx]
            train_subjects = sorted(np.unique(dataset.subject_ids[abs_train_idx]).tolist())
            valid_subjects = sorted(np.unique(dataset.subject_ids[abs_valid_idx]).tolist())
            _assert_no_subject_overlap(train_subjects, valid_subjects, "manifest inner split")
            inner_fold_entries.append(
                InnerFoldManifest(
                    inner_fold_index=inner_fold_index,
                    train_indices=train_sample_indices.astype(int).tolist(),
                    valid_indices=valid_sample_indices.astype(int).tolist(),
                    train_subjects=train_subjects,
                    valid_subjects=valid_subjects,
                    train_sample_ids=[dataset.sample_ids[i] for i in abs_train_idx.tolist()],
                    valid_sample_ids=[dataset.sample_ids[i] for i in abs_valid_idx.tolist()],
                )
            )

        outer_folds.append(
            OuterFoldManifest(
                outer_fold_index=outer_fold_index,
                outer_test_subject=outer_subject,
                outer_train_subjects=outer_train_subjects,
                outer_test_indices=outer_test_sample_indices.astype(int).tolist(),
                outer_train_indices=outer_train_sample_indices.astype(int).tolist(),
                inner_folds=inner_fold_entries,
                seed=evaluation.seed + outer_fold_index,
            )
        )

    manifest = FoldManifest(
        dataset_order_hash=dataset_hash,
        sample_indices=dataset.sample_indices.astype(int).tolist(),
        sample_ids=list(dataset.sample_ids),
        subject_ids=dataset.subject_ids.astype(str).tolist(),
        labels=dataset.y.astype(int).tolist(),
        seed=evaluation.seed,
        outer_protocol=evaluation.outer_protocol,
        inner_folds=evaluation.inner_folds,
        outer_folds=outer_folds,
    )
    manifest.manifest_hash = compute_manifest_hash(manifest.to_dict())
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_fold_manifest(path: Path | str) -> FoldManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    outer_folds = [
        OuterFoldManifest(
            outer_fold_index=item["outer_fold_index"],
            outer_test_subject=item["outer_test_subject"],
            outer_train_subjects=list(item["outer_train_subjects"]),
            outer_test_indices=list(item["outer_test_indices"]),
            outer_train_indices=list(item["outer_train_indices"]),
            inner_folds=[
                InnerFoldManifest(
                    inner_fold_index=inner["inner_fold_index"],
                    train_indices=list(inner["train_indices"]),
                    valid_indices=list(inner["valid_indices"]),
                    train_subjects=list(inner["train_subjects"]),
                    valid_subjects=list(inner["valid_subjects"]),
                    train_sample_ids=list(inner["train_sample_ids"]),
                    valid_sample_ids=list(inner["valid_sample_ids"]),
                )
                for inner in item["inner_folds"]
            ],
            seed=item["seed"],
        )
        for item in data["outer_folds"]
    ]
    return FoldManifest(
        dataset_order_hash=data["dataset_order_hash"],
        sample_indices=list(data["sample_indices"]),
        sample_ids=list(data["sample_ids"]),
        subject_ids=list(data["subject_ids"]),
        labels=list(data["labels"]),
        seed=data["seed"],
        outer_protocol=data["outer_protocol"],
        inner_folds=data["inner_folds"],
        outer_folds=outer_folds,
        manifest_hash=data["manifest_hash"],
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


def validate_manifest_against_dataset(manifest: FoldManifest, dataset: NestedDataset) -> None:
    if manifest.dataset_order_hash != compute_dataset_order_hash(dataset):
        raise ValueError("Manifest dataset_order_hash does not match the current dataset order")
    if manifest.sample_indices != dataset.sample_indices.astype(int).tolist():
        raise ValueError("Manifest sample indices do not match the current dataset")
    if manifest.sample_ids != list(dataset.sample_ids):
        raise ValueError("Manifest sample IDs do not match the current dataset")
    if manifest.subject_ids != dataset.subject_ids.astype(str).tolist():
        raise ValueError("Manifest subject IDs do not match the current dataset")
    if manifest.labels != dataset.y.astype(int).tolist():
        raise ValueError("Manifest labels do not match the current dataset")


def _assert_no_subject_overlap(
    train_subjects: list[str] | np.ndarray,
    other_subjects: list[str] | np.ndarray,
    context: str,
) -> None:
    overlap = set(np.asarray(train_subjects).tolist()) & set(np.asarray(other_subjects).tolist())
    if overlap:
        raise ValueError(f"Subject overlap detected in {context}: {sorted(overlap)}")
