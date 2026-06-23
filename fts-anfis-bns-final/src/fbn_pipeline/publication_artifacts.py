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
    package_names: Iterable[str] = ("numpy", "pandas", "scikit-learn", "scipy", "openpyxl", "xgboost", "torch", "pytest"),
) -> dict[str, object]:
    return {
        "run_name": run_name,
        "mode": mode,
        "allow_partial_outer_folds": bool(allow_partial_outer_folds),
        "seed": int(seed),
        "configuration_hash": configuration_hash,
        "dataset_order_hash": dataset_order_hash,
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
