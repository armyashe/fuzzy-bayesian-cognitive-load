from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .config import ExperimentConfig
from .data import TrainingSplit, load_training_split
from .results import DevelopmentResult, DevelopmentSelection
from .runner import (
    _deterministic_thread_context,
    _json_hash,
    lock_development_config,
    run_development,
    select_best_development_config,
)

JOINT_MEMBERSHIP_FUNCTIONS = ["triangular", "gaussian", "trapezoid", "gbell"]
JOINT_RULE_SIZES = [3, 6, 8, 12]
JOINT_TIE_BREAK_RULES = [
    "oof_macro_f1",
    "oof_auc",
    "lower_model_complexity",
    "configuration_id_lexicographic",
]


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(text, encoding=encoding)
    tmp_path.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(_json_ready(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def compute_groupkfold_indices(
    X_train: np.ndarray,
    y_train: np.ndarray,
    user_train: np.ndarray,
    *,
    k_folds: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    gkf = GroupKFold(n_splits=k_folds)
    return [
        (np.asarray(tr_idx, dtype=int), np.asarray(va_idx, dtype=int))
        for tr_idx, va_idx in gkf.split(X_train, y_train, user_train)
    ]


def fold_assignment_hash(
    fold_indices: list[tuple[np.ndarray, np.ndarray]],
    user_train: np.ndarray,
) -> str:
    payload = []
    for fold_i, (tr_idx, va_idx) in enumerate(fold_indices):
        payload.append(
            {
                "fold": fold_i,
                "train_idx": np.asarray(tr_idx, dtype=int).tolist(),
                "val_idx": np.asarray(va_idx, dtype=int).tolist(),
                "train_users": sorted(np.unique(user_train[tr_idx]).tolist()),
                "val_users": sorted(np.unique(user_train[va_idx]).tolist()),
            }
        )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _pretty_mf_name(mf_type: str) -> str:
    mapping = {
        "triangular": "Triangular",
        "gaussian": "Gaussian",
        "trapezoid": "Trapezoidal",
        "gbell": "Generalized bell",
    }
    return mapping[mf_type]


def joint_configuration_id(mf_type: str, n_rules: int) -> str:
    return f"PUB_JOINT__MF_{mf_type.upper()}__R{n_rules}"


def build_joint_configs(
    *,
    base_cfg: ExperimentConfig,
    membership_functions: list[str],
    rule_sizes: list[int],
) -> list[ExperimentConfig]:
    configs = []
    for mf_type, n_rules in itertools.product(membership_functions, rule_sizes):
        configs.append(
            base_cfg.merged(
                mf_type=mf_type,
                n_rules=int(n_rules),
                bn_structure="naive",
                bn_soft=True,
                n_bins=5,
                alpha=0.5,
                rank_metric="balanced_accuracy",
                threshold_metric="balanced_accuracy",
                variant_name=joint_configuration_id(mf_type, int(n_rules)),
                experiment_group="PUB_JOINT",
            )
        )
    return configs


def validate_joint_development_results(
    *,
    results: list[DevelopmentResult],
    membership_functions: list[str],
    rule_sizes: list[int],
    user_train: np.ndarray,
    expected_samples: int,
) -> DevelopmentSelection:
    expected_pairs = {
        (mf_type, int(n_rules))
        for mf_type, n_rules in itertools.product(membership_functions, rule_sizes)
    }
    observed_pairs = {
        (str(result.config.mf_type), int(result.config.n_rules))
        for result in results
    }
    if len(results) != len(expected_pairs) or observed_pairs != expected_pairs:
        raise ValueError(
            f"Joint selection must complete all 16 configurations; "
            f"expected {len(expected_pairs)} got {len(results)}"
        )

    expected_index_set = set(range(expected_samples))
    for result in results:
        if len(result.oof_post) != expected_samples:
            raise ValueError(
                f"{result.configuration_id} must produce exactly {expected_samples} "
                f"OOF predictions; got {len(result.oof_post)}"
            )
        if np.isnan(result.oof_post).any():
            raise ValueError(f"{result.configuration_id} contains NaN OOF predictions")
        if len(result.fold_indices) == 0:
            raise ValueError(f"{result.configuration_id} is missing fold assignments")

        val_seen: list[int] = []
        for fold_i, (tr_idx, va_idx) in enumerate(result.fold_indices):
            train_users = set(np.asarray(user_train)[tr_idx].tolist())
            val_users = set(np.asarray(user_train)[va_idx].tolist())
            if train_users.intersection(val_users):
                raise ValueError(
                    f"{result.configuration_id} has participant leakage in fold {fold_i}"
                )
            val_seen.extend(np.asarray(va_idx, dtype=int).tolist())
        if set(val_seen) != expected_index_set or len(val_seen) != expected_samples:
            raise ValueError(
                f"{result.configuration_id} must cover each development sample exactly "
                f"once in validation and yield {expected_samples} OOF predictions"
            )

    return select_best_development_config(results)


def _joint_long_form_df(
    results: list[DevelopmentResult],
    selection: DevelopmentSelection,
) -> pd.DataFrame:
    rows = []
    selected_id = selection.selected_result.configuration_id
    for result in selection.ranked_results:
        rows.append(
            {
                "configuration_id": result.configuration_id,
                "mf_type": result.config.mf_type,
                "n_rules": int(result.config.n_rules),
                "oof_balanced_accuracy": float(result.oof_metrics["bal_acc"]),
                "oof_macro_f1": float(result.oof_metrics["macro_f1"]),
                "oof_roc_auc": float(result.oof_metrics["auc"]),
                "oof_accuracy": float(result.oof_metrics["accuracy"]),
                "selected_threshold": float(result.threshold),
                "selected": result.configuration_id == selected_id,
                "selection_rank": selection.ranked_results.index(result) + 1,
                "seed": int(result.config.seed),
                "k_folds": int(result.config.k_folds),
                "threshold_metric": result.threshold_metric,
                "prediction_hash": _json_hash(np.asarray(result.oof_post, dtype=float).tolist()),
            }
        )
    return pd.DataFrame(rows)


def _joint_matrix_df(long_form_df: pd.DataFrame) -> pd.DataFrame:
    pivot = long_form_df.pivot(
        index="mf_type",
        columns="n_rules",
        values="oof_balanced_accuracy",
    ).loc[JOINT_MEMBERSHIP_FUNCTIONS, JOINT_RULE_SIZES]
    pivot.index = [_pretty_mf_name(mf_type) for mf_type in pivot.index]
    pivot.columns = [f"{int(col)} rules" for col in pivot.columns]
    pivot.insert(0, "Membership function", pivot.index.tolist())
    return pivot.reset_index(drop=True)


def _markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, (float, np.floating)):
                values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_joint_selection_artifacts(
    *,
    results: list[DevelopmentResult],
    selection: DevelopmentSelection,
    artifact_dir: Path,
    fold_assignment_hash: str,
    legacy_note: str,
) -> dict[str, Path]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    long_form_df = _joint_long_form_df(results, selection)
    matrix_df = _joint_matrix_df(long_form_df)
    selected = selection.selected_result

    long_form_path = artifact_dir / "joint_fts_anfis_selection_oof.csv"
    matrix_path = artifact_dir / "joint_fts_anfis_balanced_accuracy_matrix.csv"
    report_path = artifact_dir / "joint_fts_anfis_selection_report.md"
    summary_path = artifact_dir / "joint_fts_anfis_selection_summary.json"

    long_form_df.to_csv(long_form_path, index=False)
    matrix_df.to_csv(matrix_path, index=False)

    report_lines = [
        "# Joint FTS-ANFIS Selection Report",
        "",
        "## OOF Balanced Accuracy Matrix",
        "",
        _markdown_table(matrix_df),
        "",
        "## Selected Configuration",
        "",
        f"- Configuration ID: `{selected.configuration_id}`",
        f"- Membership function: `{selected.config.mf_type}`",
        f"- Number of rules: `{int(selected.config.n_rules)}`",
        f"- OOF balanced accuracy: `{selected.oof_metrics['bal_acc']:.4f}`",
        f"- OOF Macro-F1: `{selected.oof_metrics['macro_f1']:.4f}`",
        f"- OOF ROC-AUC: `{selected.oof_metrics['auc']:.4f}`",
        f"- Selected threshold: `{selected.threshold:.4f}`",
        f"- Fold assignment hash: `{fold_assignment_hash}`",
        "",
        "The hold-out set was not used for membership-function selection, "
        "rule-base selection, threshold selection, configuration ranking, or "
        "any publication-time hyperparameter tuning.",
        "",
        "## Legacy Sequential Artifacts",
        "",
        legacy_note,
    ]
    atomic_write_text(report_path, "\n".join(report_lines))

    summary_payload = {
        "protocol_version": "publication_protocol_v2",
        "selection_mode": "joint_fts_anfis_selection",
        "selection_metric": selection.ranking_metric,
        "tie_break_rules": list(selection.tie_break_rules),
        "selected_configuration_id": selected.configuration_id,
        "selected_mf_type": selected.config.mf_type,
        "selected_n_rules": int(selected.config.n_rules),
        "selected_threshold": float(selected.threshold),
        "selected_oof_balanced_accuracy": float(selected.oof_metrics["bal_acc"]),
        "selected_oof_macro_f1": float(selected.oof_metrics["macro_f1"]),
        "selected_oof_roc_auc": float(selected.oof_metrics["auc"]),
        "seed": int(selected.config.seed),
        "k_folds": int(selected.config.k_folds),
        "fold_assignment_hash": fold_assignment_hash,
        "long_form_csv": str(long_form_path),
        "balanced_accuracy_matrix_csv": str(matrix_path),
        "report_md": str(report_path),
        "created_at_utc": _utc_timestamp(),
    }
    atomic_write_json(summary_path, summary_payload)
    return {
        "long_form_path": long_form_path,
        "matrix_path": matrix_path,
        "report_path": report_path,
        "summary_path": summary_path,
    }


def preserve_legacy_sequential_artifacts(development_dir: Path) -> dict[str, Path]:
    legacy_dir = development_dir / "legacy_sequential_selection"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_files = [
        "fts_selection_oof.csv",
        "anfis_rule_selection_oof.csv",
        "bn_selection_oof.csv",
        "publication_development_summary.json",
        "publication_development_summary.md",
    ]
    moved: dict[str, Path] = {}
    for filename in legacy_files:
        src = development_dir / filename
        dst = legacy_dir / filename
        if src.exists() and not dst.exists():
            src.replace(dst)
        if dst.exists():
            moved[filename] = dst
    readme_path = legacy_dir / "README.md"
    note = (
        "# Legacy Sequential Selection Artifacts\n\n"
        "These files were produced by the older sequential publication workflow "
        "(FTS membership selection -> ANFIS rule selection -> BN selection). "
        "They are retained only for provenance and exploratory comparison. "
        "They must not be used as the source of truth for the current "
        "publication lock or hold-out evaluation.\n"
    )
    atomic_write_text(readme_path, note)
    moved["README.md"] = readme_path
    return moved


def publication_protocol_paths(project_root: Path | str) -> dict[str, Path]:
    root = Path(project_root).resolve()
    artifact_root = root / "artifacts" / "publication_protocol_v2"
    development_dir = artifact_root / "development"
    return {
        "project_root": root,
        "artifact_root": artifact_root,
        "development_dir": development_dir,
        "lock_path": artifact_root / "locked_config.json",
        "candidate_manifest_path": development_dir / "publication_candidate_manifest.json",
        "joint_summary_path": development_dir / "joint_fts_anfis_selection_summary.json",
    }


def run_joint_fts_anfis_selection(
    *,
    training_split: TrainingSplit,
    artifact_root: Path,
    membership_functions: list[str] | None = None,
    rule_sizes: list[int] | None = None,
    seed: int = 42,
    print_progress: bool = True,
) -> dict[str, Any]:
    membership_functions = membership_functions or list(JOINT_MEMBERSHIP_FUNCTIONS)
    rule_sizes = rule_sizes or list(JOINT_RULE_SIZES)

    base_cfg = ExperimentConfig(
        seed=seed,
        k_folds=5,
        n_sets=3,
        mf_type="gaussian",
        mf_sigma_scale=0.4,
        fts_norm="row",
        grid_margin_scale=0.05,
        anfis_mf_type="gaussian",
        n_rules=6,
        sigma_scale=1.3,
        ridge_lambda=0.01,
        n_epochs=50,
        lr=0.003,
        n_bins=5,
        alpha=0.5,
        bn_structure="naive",
        bn_soft=True,
        bn_reliability=False,
        calibrate="none",
        threshold_metric="balanced_accuracy",
        threshold_grid=181,
        rank_metric="balanced_accuracy",
        ablation="exp1",
        experiment_group="PUB_JOINT",
    )
    configs = build_joint_configs(
        base_cfg=base_cfg,
        membership_functions=membership_functions,
        rule_sizes=rule_sizes,
    )

    fold_indices = compute_groupkfold_indices(
        training_split.X_train,
        training_split.y_train,
        training_split.user_train,
        k_folds=base_cfg.k_folds,
    )
    fold_hash = fold_assignment_hash(fold_indices, training_split.user_train)

    with _deterministic_thread_context():
        results = [
            run_development(
                training_split.X_train,
                training_split.y_train,
                training_split.user_train,
                cfg,
                configuration_id=cfg.variant_name,
                print_progress=print_progress,
                fold_indices=fold_indices,
            )
            for cfg in configs
        ]

    selection = validate_joint_development_results(
        results=results,
        membership_functions=membership_functions,
        rule_sizes=rule_sizes,
        user_train=training_split.user_train,
        expected_samples=int(training_split.X_train.shape[0]),
    )

    development_dir = artifact_root / "development"
    legacy_paths = preserve_legacy_sequential_artifacts(development_dir)
    artifacts = build_joint_selection_artifacts(
        results=results,
        selection=selection,
        artifact_dir=development_dir,
        fold_assignment_hash=fold_hash,
        legacy_note=(
            "Legacy sequential artifacts were preserved under "
            "`development/legacy_sequential_selection/` and are reference-only."
        ),
    )

    return {
        "results": results,
        "selection": selection,
        "selected_result": selection.selected_result,
        "selected_threshold": float(selection.selected_result.threshold),
        "long_form_df": _joint_long_form_df(results, selection),
        "matrix_df": _joint_matrix_df(_joint_long_form_df(results, selection)),
        "fold_indices": fold_indices,
        "fold_assignment_hash": fold_hash,
        "artifacts": artifacts,
        "legacy_paths": legacy_paths,
    }


def update_publication_joint_lock(
    *,
    lock_path: Path,
    selection: DevelopmentSelection,
    selected_result: DevelopmentResult,
    fold_assignment_hash: str,
    source_joint_selection_artifact: Path,
    candidate_manifest_hash: str | None = None,
) -> dict[str, Any]:
    prediction_hash = _json_hash(np.asarray(selected_result.oof_post, dtype=float).tolist())
    extra_metadata = {
        "selection_mode": "joint_fts_anfis_selection",
        "selected_mf_type": selected_result.config.mf_type,
        "selected_n_rules": int(selected_result.config.n_rules),
        "selected_oof_threshold": float(selected_result.threshold),
        "selected_oof_balanced_accuracy": float(selected_result.oof_metrics["bal_acc"]),
        "fold_assignment_hash": fold_assignment_hash,
        "source_joint_selection_artifact": str(source_joint_selection_artifact),
        "prediction_hash": prediction_hash,
        "rank_metric": "balanced_accuracy",
    }
    if candidate_manifest_hash is not None:
        extra_metadata["candidate_manifest_hash"] = candidate_manifest_hash
    return lock_development_config(
        selected_result,
        selection,
        output_path=lock_path,
        extra_metadata=extra_metadata,
    )


def run_publication_joint_selection(
    project_root: Path | str,
    *,
    print_progress: bool = True,
) -> dict[str, Any]:
    paths = publication_protocol_paths(project_root)
    training_split = load_training_split(paths["project_root"] / "last_30s_segments")
    result = run_joint_fts_anfis_selection(
        training_split=training_split,
        artifact_root=paths["artifact_root"],
        print_progress=print_progress,
    )
    candidate_manifest_hash = None
    if paths["candidate_manifest_path"].exists():
        manifest_payload = json.loads(paths["candidate_manifest_path"].read_text(encoding="utf-8"))
        candidate_manifest_hash = manifest_payload.get("manifest_hash")
    lock_payload = update_publication_joint_lock(
        lock_path=paths["lock_path"],
        selection=result["selection"],
        selected_result=result["selected_result"],
        fold_assignment_hash=result["fold_assignment_hash"],
        source_joint_selection_artifact=result["artifacts"]["summary_path"],
        candidate_manifest_hash=candidate_manifest_hash,
    )
    result["lock_payload"] = lock_payload
    result["paths"] = paths
    return result
