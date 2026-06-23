from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Iterable

import pandas as pd


RUN_SUBDIRECTORIES = (
    "metadata",
    "predictions",
    "metrics",
    "statistics",
    "interpretability",
    "tables",
    "figures",
)


@dataclass(frozen=True, slots=True)
class PublicationRunLayout:
    run_root: Path
    metadata_dir: Path
    predictions_dir: Path
    metrics_dir: Path
    statistics_dir: Path
    interpretability_dir: Path
    tables_dir: Path
    figures_dir: Path
    mode: str
    run_name: str


def load_run_layout(run_root: Path | str, *, mode: str | None = None, run_name: str | None = None) -> PublicationRunLayout:
    root = Path(run_root)
    resolved_mode = mode or _load_mode_from_metadata(root)
    resolved_name = run_name or root.name
    return PublicationRunLayout(
        run_root=root,
        metadata_dir=root / "metadata",
        predictions_dir=root / "predictions",
        metrics_dir=root / "metrics",
        statistics_dir=root / "statistics",
        interpretability_dir=root / "interpretability",
        tables_dir=root / "tables",
        figures_dir=root / "figures",
        mode=resolved_mode,
        run_name=resolved_name,
    )


def create_run_layout(
    *,
    output_root: Path | str,
    run_name: str,
    mode: str,
    overwrite: bool = False,
) -> PublicationRunLayout:
    if not run_name or not str(run_name).strip():
        raise ValueError("run_name is required")
    if mode not in {"publication", "smoke"}:
        raise ValueError(f"Unsupported mode: {mode!r}")

    root = Path(output_root) / str(run_name)
    if root.exists() and _contains_results(root) and not overwrite:
        raise FileExistsError(
            f"Run directory {root} already contains results; pass --overwrite to replace it."
        )
    root.mkdir(parents=True, exist_ok=True)
    subdirs = {name: root / name for name in RUN_SUBDIRECTORIES}
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return PublicationRunLayout(
        run_root=root,
        metadata_dir=subdirs["metadata"],
        predictions_dir=subdirs["predictions"],
        metrics_dir=subdirs["metrics"],
        statistics_dir=subdirs["statistics"],
        interpretability_dir=subdirs["interpretability"],
        tables_dir=subdirs["tables"],
        figures_dir=subdirs["figures"],
        mode=mode,
        run_name=str(run_name),
    )


def collect_runtime_metadata(
    *,
    run_name: str,
    mode: str,
    allow_partial_outer_folds: bool,
    seed: int,
    configuration_hash: str,
    dataset_order_hash: str,
    command: str | None = None,
    protocol: str | None = None,
    complete_coverage: bool | None = None,
    train_subject_ids: list[str] | None = None,
    test_subject_ids: list[str] | None = None,
    train_sample_count: int | None = None,
    test_sample_count: int | None = None,
    groupkfold_split_count: int | None = None,
    model_selection_metric: str | None = None,
    threshold_selection_metric: str | None = None,
    package_names: Iterable[str] = ("numpy", "pandas", "scikit-learn", "scipy", "openpyxl", "xgboost", "torch", "pytest"),
) -> dict[str, object]:
    return {
        "run_name": run_name,
        "mode": mode,
        "allow_partial_outer_folds": bool(allow_partial_outer_folds),
        "complete_coverage": complete_coverage,
        "protocol": protocol,
        "seed": int(seed),
        "configuration_hash": configuration_hash,
        "dataset_order_hash": dataset_order_hash,
        "train_subject_count": None if train_subject_ids is None else len(train_subject_ids),
        "test_subject_count": None if test_subject_ids is None else len(test_subject_ids),
        "train_sample_count": train_sample_count,
        "test_sample_count": test_sample_count,
        "groupkfold_split_count": groupkfold_split_count,
        "model_selection_metric": model_selection_metric,
        "threshold_selection_metric": threshold_selection_metric,
        "training_subject_ids": train_subject_ids,
        "test_subject_ids": test_subject_ids,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "command": command or "",
        "package_versions": {
            name: _safe_version(name) for name in package_names
        },
        **_safe_git_provenance(),
    }


def write_runtime_metadata(layout: PublicationRunLayout, metadata: dict[str, object]) -> Path:
    output_path = layout.metadata_dir / "runtime_metadata.json"
    output_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return output_path


def validate_publication_predictions(
    frame: pd.DataFrame,
    *,
    expected_subject_ids: list[str],
    expected_sample_ids: list[str],
    mode: str,
    allow_partial_outer_folds: bool,
) -> None:
    required_columns = {"subject_id", "sample_id", "outer_fold_index"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing required columns: {sorted(missing)}")

    if frame["sample_id"].duplicated().any():
        duplicates = sorted(frame.loc[frame["sample_id"].duplicated(), "sample_id"].astype(str).tolist())
        raise ValueError(f"Duplicate outer prediction rows found for sample IDs: {duplicates}")

    outer_subject_counts = frame.groupby("subject_id")["outer_fold_index"].nunique().to_dict()
    if any(count > 1 for count in outer_subject_counts.values()):
        raise ValueError("Each subject must appear as outer test exactly once")

    expected_subject_set = {str(subject) for subject in expected_subject_ids}
    expected_sample_set = {str(sample_id) for sample_id in expected_sample_ids}
    actual_subject_set = set(frame["subject_id"].astype(str))
    actual_sample_set = set(frame["sample_id"].astype(str))
    missing_subjects = sorted(expected_subject_set - actual_subject_set)
    missing_samples = sorted(expected_sample_set - actual_sample_set)

    if (missing_subjects or missing_samples) and not allow_partial_outer_folds:
        raise ValueError(
            "Publication mode requires complete coverage; "
            f"missing subjects={missing_subjects}, missing samples={len(missing_samples)}"
        )
    if mode == "publication" and allow_partial_outer_folds:
        raise ValueError("Publication mode cannot allow partial outer folds")


def validate_official_holdout_predictions(
    frame: pd.DataFrame,
    *,
    expected_train_sample_ids: list[str],
    expected_test_sample_ids: list[str],
    expected_test_subject_ids: list[str],
    mode: str,
    protocol: str,
    complete_coverage: bool,
) -> None:
    required_columns = {"method", "subject_id", "sample_id", "prediction_role"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame is missing required columns: {sorted(missing)}")

    test_rows = frame.loc[frame["prediction_role"] == "official_test"].copy()
    if test_rows.empty:
        raise ValueError("Official publication outputs require official_test prediction rows")
    if test_rows[["method", "sample_id"]].duplicated().any():
        raise ValueError("Duplicate method/sample rows found in official_test predictions")
    if set(test_rows["sample_id"].astype(str)) & set(map(str, expected_train_sample_ids)):
        raise ValueError("Official final predictions contain training sample IDs")

    grouped = test_rows.groupby("method", sort=True)
    expected_test_sample_set = set(map(str, expected_test_sample_ids))
    expected_test_subject_set = set(map(str, expected_test_subject_ids))
    reference_sample_ids: set[str] | None = None
    for method, method_rows in grouped:
        sample_ids = set(method_rows["sample_id"].astype(str))
        subject_ids = set(method_rows["subject_id"].astype(str))
        if mode == "publication":
            if len(method_rows) != len(expected_test_sample_ids):
                raise ValueError(
                    "Official publication mode requires exactly "
                    f"{len(expected_test_sample_ids)} final predictions per method"
                )
            if sample_ids != expected_test_sample_set:
                raise ValueError("Official publication mode requires complete coverage of official test sample IDs")
            if subject_ids != expected_test_subject_set:
                raise ValueError("Official publication mode requires complete coverage of official test subject IDs")
        if reference_sample_ids is None:
            reference_sample_ids = sample_ids
        elif sample_ids != reference_sample_ids:
            raise ValueError("All methods must predict the same official test sample IDs")

    if mode == "publication" and protocol != "official_holdout_groupkfold":
        raise ValueError("Publication mode must use protocol official_holdout_groupkfold for primary results")
    if mode == "publication" and not complete_coverage:
        raise ValueError("Publication artifacts require complete coverage")


def _contains_results(root: Path) -> bool:
    if not root.exists():
        return False
    return any(root.iterdir())


def _safe_version(package_name: str) -> str | None:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _load_mode_from_metadata(root: Path) -> str:
    metadata_path = root / "metadata" / "runtime_metadata.json"
    if metadata_path.exists():
        try:
            return str(json.loads(metadata_path.read_text(encoding="utf-8")).get("mode", "publication"))
        except json.JSONDecodeError:
            return "publication"
    return "publication"


def _safe_git_provenance() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return {"git_commit_hash": None, "git_worktree_dirty": None}
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(status)
    except Exception:
        dirty = None
    return {"git_commit_hash": commit, "git_worktree_dirty": dirty}
