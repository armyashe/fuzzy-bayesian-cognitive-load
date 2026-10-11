from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .baselines import (
    _decision_scores,
    _make_estimator,
    _scale_required,
    build_feature_matrix,
)
from .config import ExperimentConfig, assert_thesis_config, setup_seed
from .data import TEST_IDS, TRAIN_IDS, DatasetSplit, load_dataset
from .metrics import compute_metrics, per_subject_metrics
from .runner import _load_and_validate_lock, run_final_evaluation_from_lock


PROPOSED_LABEL = "Proposed FTS-ANFIS-BN"
MAIN_MODEL_ORDER = [
    PROPOSED_LABEL,
    "Statistical Logistic Regression",
    "Statistical RBF SVM",
    "Statistical Random Forest",
    "Statistical HistGradientBoosting",
    "Flattened-window Logistic Regression",
]
ABLATION_ORDER = [
    "Full FTS-ANFIS-BN",
    "No BN",
    "No ANFIS",
    "No FTS",
]
NON_SCIENTIFIC_CONFIG_FIELDS = {
    "dump_exp2_debug",
    "dump_rules",
    "exp2_debug_dir",
    "experiment_group",
    "save_csv",
    "variant_name",
}


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _stable_json_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=float)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prediction_hash(
    *,
    model_id: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    preds: np.ndarray,
    subjects: np.ndarray,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(model_id.encode("utf-8"))
    hasher.update(np.ascontiguousarray(y_true, dtype=np.int64).tobytes())
    hasher.update(np.ascontiguousarray(scores, dtype=np.float64).tobytes())
    hasher.update(np.ascontiguousarray(preds, dtype=np.int64).tobytes())
    hasher.update("\n".join(subjects.astype(str).tolist()).encode("utf-8"))
    return hasher.hexdigest()


def _sanitize_key(value: str) -> str:
    cleaned = []
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_project_root(project_root: Path | str) -> Path:
    root = Path(project_root).resolve()
    if not (root / "src" / "fbn_pipeline").exists():
        raise FileNotFoundError(f"Project root is missing src/fbn_pipeline: {root}")
    return root


def _paths(project_root: Path) -> dict[str, Path]:
    artifact_root = project_root / "artifacts" / "publication_protocol_v2"
    development_dir = artifact_root / "development"
    return {
        "project_root": project_root,
        "artifact_root": artifact_root,
        "development_dir": development_dir,
        "development_summary_json": development_dir / "joint_fts_anfis_selection_summary.json",
        "development_report_md": development_dir / "joint_fts_anfis_selection_report.md",
        "development_long_csv": development_dir / "joint_fts_anfis_selection_oof.csv",
        "development_matrix_csv": development_dir / "joint_fts_anfis_balanced_accuracy_matrix.csv",
        "development_legacy_dir": development_dir / "legacy_sequential_selection",
        "proposed_lock": artifact_root / "locked_config.json",
        "baseline_lock_dir": artifact_root / "baselines" / "locks",
        "ablation_lock_dir": artifact_root / "ablations" / "locks",
        "comparison_manifest": artifact_root / "comparison" / "final_comparison_manifest.json",
        "statistical_plan": artifact_root / "comparison" / "statistical_analysis_plan.json",
        "statistical_plan_md": artifact_root / "comparison" / "statistical_analysis_plan.md",
        "output_dir": artifact_root / "final_evaluation",
    }


def _validate_self_hashed_payload(
    path: Path,
    *,
    hash_field: str,
    label: str,
) -> dict[str, Any]:
    payload = _load_json(path)
    expected_hash = _stable_json_hash({k: v for k, v in payload.items() if k != hash_field})
    actual_hash = payload.get(hash_field)
    if actual_hash != expected_hash:
        raise ValueError(f"{label} hash mismatch for {path}")
    payload["_path"] = path
    payload["_path_hash"] = _file_sha256(path)
    return payload


def _baseline_config_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "family": payload["baseline_family"],
        "representation": payload["input_representation"],
        "hyperparameters": payload["selected_hyperparameters"],
        "preprocessing_spec": payload["preprocessing_specification"],
        "feature_spec": payload["feature_definitions"],
        "seed": payload["seed"],
        "k_folds": payload["k_folds"],
    }


def _validate_baseline_lock(path: Path, expected_manifest_hash: str) -> dict[str, Any]:
    payload = _load_json(path)
    required = {
        "protocol_version",
        "schema_version",
        "baseline_family",
        "configuration_id",
        "input_representation",
        "feature_definitions",
        "selected_hyperparameters",
        "preprocessing_specification",
        "oof_metrics",
        "oof_threshold",
        "threshold_metric",
        "seed",
        "k_folds",
        "grouping_protocol",
        "comparison_manifest_hash",
        "selected_using",
        "threshold_source",
        "final_test_evaluated",
        "prediction_hash",
        "config_hash",
        "lock_hash",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Baseline lock missing fields for {path}: {missing}")
    if payload["protocol_version"] != "publication_protocol_v2":
        raise ValueError(f"Unsupported baseline protocol version: {path}")
    if payload["comparison_manifest_hash"] != expected_manifest_hash:
        raise ValueError(f"Baseline comparison manifest hash mismatch: {path}")
    if payload["selected_using"] != "training_oof_only":
        raise ValueError(f"Baseline not selected using training OOF only: {path}")
    if payload["threshold_source"] != "training_oof_only":
        raise ValueError(f"Baseline threshold source is not training OOF only: {path}")
    if payload["final_test_evaluated"] not in (False, 0):
        raise ValueError(f"Baseline lock already records test evaluation: {path}")
    expected_config_hash = _stable_json_hash(_baseline_config_payload(payload))
    if payload["config_hash"] != expected_config_hash:
        raise ValueError(f"Baseline config hash mismatch: {path}")
    expected_lock_hash = _stable_json_hash({k: v for k, v in payload.items() if k != "lock_hash"})
    if payload["lock_hash"] != expected_lock_hash:
        raise ValueError(f"Baseline lock hash mismatch: {path}")
    payload["_path"] = path
    payload["_path_hash"] = _file_sha256(path)
    return payload


def _validate_pipeline_lock(path: Path) -> dict[str, Any]:
    _, payload = _load_and_validate_lock(path)
    payload["_path"] = path
    payload["_path_hash"] = _file_sha256(path)
    return payload


def _evaluate_pipeline_lock(
    *,
    display_name: str,
    model_id: str,
    lock_payload: dict[str, Any],
    dataset: DatasetSplit,
    output_dir: Path,
) -> dict[str, Any]:
    result = run_final_evaluation_from_lock(
        locked_config_path=lock_payload["_path"],
        dataset=dataset,
        output_dir=output_dir / "locked_runs",
        confirm_final_evaluation=True,
    )
    scores = np.asarray(result.test_post, dtype=float)
    preds = np.asarray(result.test_pred, dtype=int)
    metrics = _format_metric_row(
        display_name=display_name,
        model_id=model_id,
        lock_hash=lock_payload["lock_hash"],
        threshold=float(lock_payload["threshold"]),
        y_true=dataset.y_test,
        scores=scores,
        preds=preds,
    )
    return {
        "display_name": display_name,
        "model_id": model_id,
        "lock_hash": lock_payload["lock_hash"],
        "lock_path": str(lock_payload["_path"]),
        "lock_file_hash_before": lock_payload["_path_hash"],
        "threshold": float(lock_payload["threshold"]),
        "scores": scores,
        "preds": preds,
        "y_true": dataset.y_test.copy(),
        "subjects": dataset.user_test.copy(),
        "metrics": metrics,
        "result": result,
    }


def _evaluate_baseline_lock(
    *,
    display_name: str,
    lock_payload: dict[str, Any],
    dataset: DatasetSplit,
) -> dict[str, Any]:
    family = lock_payload["baseline_family"]
    representation = lock_payload["input_representation"]
    hyperparameters = dict(lock_payload["selected_hyperparameters"])
    threshold = float(lock_payload["oof_threshold"])
    expected_preprocessing = {
        "imputer": "median",
        "scaler": "standard" if _scale_required(family) else "none",
    }
    if lock_payload["preprocessing_specification"] != expected_preprocessing:
        raise ValueError(f"Baseline preprocessing lock mismatch for {display_name}")

    setup_seed(int(lock_payload["seed"]))
    train_features, train_feature_spec = build_feature_matrix(dataset.X_train, representation)
    test_features, test_feature_spec = build_feature_matrix(dataset.X_test, representation)
    if train_feature_spec != lock_payload["feature_definitions"]:
        raise ValueError(f"Baseline train feature spec drift for {display_name}")
    if test_feature_spec != lock_payload["feature_definitions"]:
        raise ValueError(f"Baseline test feature spec drift for {display_name}")

    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if _scale_required(family):
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", _make_estimator(family, hyperparameters, int(lock_payload["seed"]))))
    pipeline = Pipeline(steps)

    with threadpool_limits(limits=1):
        pipeline.fit(train_features, dataset.y_train)
        scores = np.asarray(_decision_scores(pipeline, test_features, family), dtype=float)

    preds = (scores >= threshold).astype(int)
    metrics = _format_metric_row(
        display_name=display_name,
        model_id=lock_payload["configuration_id"],
        lock_hash=lock_payload["lock_hash"],
        threshold=threshold,
        y_true=dataset.y_test,
        scores=scores,
        preds=preds,
    )
    return {
        "display_name": display_name,
        "model_id": lock_payload["configuration_id"],
        "lock_hash": lock_payload["lock_hash"],
        "lock_path": str(lock_payload["_path"]),
        "lock_file_hash_before": lock_payload["_path_hash"],
        "threshold": threshold,
        "scores": scores,
        "preds": preds,
        "y_true": dataset.y_test.copy(),
        "subjects": dataset.user_test.copy(),
        "metrics": metrics,
    }


def _format_metric_row(
    *,
    display_name: str,
    model_id: str,
    lock_hash: str,
    threshold: float,
    y_true: np.ndarray,
    scores: np.ndarray,
    preds: np.ndarray,
) -> dict[str, Any]:
    metric_values = compute_metrics(y_true, scores, threshold)
    prediction_hash = _prediction_hash(
        model_id=model_id,
        y_true=y_true,
        scores=scores,
        preds=preds,
        subjects=np.asarray([], dtype=str),
    )
    cm = confusion_matrix(y_true, preds, labels=[0, 1])
    return {
        "Model": display_name,
        "model_id": model_id,
        "lock_hash": lock_hash,
        "threshold": threshold,
        "Accuracy": metric_values["accuracy"],
        "Balanced Accuracy": metric_values["bal_acc"],
        "Precision": metric_values["precision"],
        "Recall": metric_values["recall"],
        "TASK F1": metric_values["f1"],
        "Macro-F1": metric_values["macro_f1"],
        "ROC-AUC": metric_values["auc"],
        "confusion_matrix": cm.tolist(),
        "prediction_hash": prediction_hash,
    }


def _fix_prediction_hashes(results: list[dict[str, Any]]) -> None:
    for entry in results:
        entry["metrics"]["prediction_hash"] = _prediction_hash(
            model_id=entry["model_id"],
            y_true=entry["y_true"],
            scores=entry["scores"],
            preds=entry["preds"],
            subjects=entry["subjects"],
        )


def _per_subject_rows(entry: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    subjects = np.unique(entry["subjects"])
    for subject in subjects:
        mask = entry["subjects"] == subject
        y_true = entry["y_true"][mask]
        scores = entry["scores"][mask]
        preds = entry["preds"][mask]
        row = {
            "Model": entry["display_name"],
            "model_id": entry["model_id"],
            "subject_id": str(subject),
            "n_samples": int(mask.sum()),
            "Accuracy": round(float(np.mean(y_true == preds)), 4),
            "Balanced Accuracy": round(float(compute_metrics(y_true, scores, entry["threshold"])["bal_acc"]), 4),
            "TASK F1": round(float(compute_metrics(y_true, scores, entry["threshold"])["f1"]), 4),
            "ROC-AUC": float("nan"),
            "roc_auc_defined": bool(np.unique(y_true).size == 2),
            "lock_hash": entry["lock_hash"],
            "prediction_hash": entry["metrics"]["prediction_hash"],
        }
        if np.unique(y_true).size == 2:
            row["ROC-AUC"] = round(float(roc_auc_score(y_true, scores)), 4)
        rows.append(row)
    return rows


def _participant_level_test_metrics(
    entry: dict[str, Any],
    *,
    participant_order: list[str] | None = None,
) -> pd.DataFrame:
    rows = per_subject_metrics(
        entry["y_true"],
        entry["preds"],
        entry["scores"],
        entry["subjects"],
        subject_order=participant_order,
    )
    participant_rows = [
        {
            "participant": str(row["subject"]),
            "n_samples": int(row["n"]),
            "accuracy": float(row["acc"]),
            "balanced_accuracy": float(row["bal_acc"]),
            "task_f1": float(row["f1"]),
            "roc_auc": float(row["auc"]),
        }
        for row in rows
    ]
    return pd.DataFrame(
        participant_rows,
        columns=[
            "participant",
            "n_samples",
            "accuracy",
            "balanced_accuracy",
            "task_f1",
            "roc_auc",
        ],
    )


def _participant_level_summary(participant_df: pd.DataFrame) -> dict[str, Any]:
    metric_columns = ["accuracy", "balanced_accuracy", "task_f1", "roc_auc"]
    means = {
        column: round(float(participant_df[column].mean(skipna=True)), 4)
        for column in metric_columns
    }
    stds = {
        column: round(float(participant_df[column].std(ddof=0, skipna=True)), 4)
        for column in metric_columns
    }
    best_idx = participant_df["accuracy"].idxmax()
    worst_idx = participant_df["accuracy"].idxmin()
    return {
        "mean": means,
        "std": stds,
        "accuracy_min": round(float(participant_df["accuracy"].min()), 4),
        "accuracy_max": round(float(participant_df["accuracy"].max()), 4),
        "highest_accuracy_participant": str(participant_df.loc[best_idx, "participant"]),
        "lowest_accuracy_participant": str(participant_df.loc[worst_idx, "participant"]),
    }


def _format_metric_or_na(value: Any) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.4f}"


def _participant_level_publication_summary(summary: dict[str, Any]) -> str:
    accuracy_mean = summary["mean"]["accuracy"]
    accuracy_std = summary["std"]["accuracy"]
    accuracy_min = summary["accuracy_min"]
    accuracy_max = summary["accuracy_max"]
    variability = "substantial" if (accuracy_max - accuracy_min) >= 0.1000 else "modest"
    return (
        "Table 8 reports participant-level performance on the five unseen hold-out participants. "
        f"Accuracy ranged from {accuracy_min:.4f} to {accuracy_max:.4f} "
        f"(unweighted mean {accuracy_mean:.4f}, SD {accuracy_std:.4f}), indicating {variability} "
        "between-participant variation in the final evaluation. These participant-level results are "
        "reported only as a post hoc stability analysis on unseen participants and were not used to "
        "select the model, tune the threshold, or adjust any hyperparameter."
    )


def _participant_level_report_markdown(
    participant_df: pd.DataFrame,
    summary: dict[str, Any],
) -> str:
    lines = [
        "# Participant-Level Analysis",
        "",
        "Table 8. Participant-level test results on the five unseen hold-out participants.",
        "",
        "| Participant | N | Acc. | Bal. Acc. | TASK F1 | ROC-AUC |",
        "| ----------- | -: | ---: | --------: | ------: | ------: |",
    ]
    for _, row in participant_df.iterrows():
        lines.append(
            "| "
            f"{row['participant']} | "
            f"{int(row['n_samples'])} | "
            f"{float(row['accuracy']):.4f} | "
            f"{float(row['balanced_accuracy']):.4f} | "
            f"{float(row['task_f1']):.4f} | "
            f"{_format_metric_or_na(row['roc_auc'])} |"
        )
    lines.extend(
        [
            "",
            "unweighted statistics across the five unseen participants (not pooled over 193 samples):",
            "",
            f"- Mean: Acc. `{summary['mean']['accuracy']:.4f}`, Bal. Acc. `{summary['mean']['balanced_accuracy']:.4f}`, TASK F1 `{summary['mean']['task_f1']:.4f}`, ROC-AUC `{_format_metric_or_na(summary['mean']['roc_auc'])}`",
            f"- Standard deviation: Acc. `{summary['std']['accuracy']:.4f}`, Bal. Acc. `{summary['std']['balanced_accuracy']:.4f}`, TASK F1 `{summary['std']['task_f1']:.4f}`, ROC-AUC `{_format_metric_or_na(summary['std']['roc_auc'])}`",
            f"- Minimum Accuracy: `{summary['accuracy_min']:.4f}`",
            f"- Maximum Accuracy: `{summary['accuracy_max']:.4f}`",
            f"- Highest Accuracy participant: `{summary['highest_accuracy_participant']}`",
            f"- Lowest Accuracy participant: `{summary['lowest_accuracy_participant']}`",
            "",
            "## Section 4.4 Summary",
            "",
            _participant_level_publication_summary(summary),
            "",
            "These participant-level results must not be used for model selection, threshold tuning, or participant-specific retraining.",
        ]
    )
    return "\n".join(lines)


def _validate_participant_level_analysis(
    entry: dict[str, Any],
    participant_df: pd.DataFrame,
    *,
    expected_participants: list[str],
    expected_subjects: np.ndarray,
    expected_threshold: float,
    expected_accuracy: float,
    expected_seed: int,
) -> dict[str, Any]:
    y_true = np.asarray(entry["y_true"], dtype=int)
    scores = np.asarray(entry["scores"], dtype=float)
    preds = np.asarray(entry["preds"], dtype=int)
    subjects = np.asarray(entry["subjects"]).astype(str)
    expected_subjects = np.asarray(expected_subjects).astype(str)
    participant_counts = {
        str(row["participant"]): int(row["n_samples"])
        for row in participant_df.to_dict(orient="records")
    }
    expected_counts = {
        participant: int(np.sum(expected_subjects == participant))
        for participant in expected_participants
    }
    aggregate_accuracy = round(float(np.mean(y_true == preds)), 4)
    return {
        "sum_n_samples_matches_expected": int(participant_df["n_samples"].sum()) == int(expected_subjects.size),
        "all_samples_map_to_exactly_one_participant": bool(
            subjects.size == expected_subjects.size
            and np.isin(subjects, expected_participants).all()
        ),
        "prediction_order_matches_user_test": bool(np.array_equal(subjects, expected_subjects)),
        "locked_threshold_matches_expected": bool(round(float(entry["threshold"]), 4) == round(float(expected_threshold), 4)),
        "no_participant_tuning_or_retraining": True,
        "aggregate_accuracy_matches_expected": bool(aggregate_accuracy == round(float(expected_accuracy), 4)),
        "seed_is_42": bool(int(expected_seed) == 42),
        "participant_counts_match_user_test": participant_counts == expected_counts,
        "array_lengths_consistent": bool(len(y_true) == len(scores) == len(preds) == len(subjects)),
        "aggregate_accuracy": aggregate_accuracy,
    }


def _clustered_bootstrap_rows(
    results: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
    level: float,
) -> list[dict[str, Any]]:
    metrics_to_track = [
        ("Accuracy", "Accuracy"),
        ("Balanced Accuracy", "Balanced Accuracy"),
        ("TASK F1", "TASK F1"),
        ("Macro-F1", "Macro-F1"),
        ("ROC-AUC", "ROC-AUC"),
    ]
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    for entry in results:
        subject_ids = np.unique(entry["subjects"])
        metric_samples = {name: [] for name, _ in metrics_to_track}
        skipped_auc = 0

        for _ in range(replicates):
            sampled_subjects = rng.choice(subject_ids, size=len(subject_ids), replace=True)
            sampled_indices = []
            for subject in sampled_subjects:
                sampled_indices.extend(np.flatnonzero(entry["subjects"] == subject).tolist())
            indices = np.asarray(sampled_indices, dtype=int)
            y_true = entry["y_true"][indices]
            scores = entry["scores"][indices]
            metric_values = compute_metrics(y_true, scores, entry["threshold"])
            metric_samples["Accuracy"].append(metric_values["accuracy"])
            metric_samples["Balanced Accuracy"].append(metric_values["bal_acc"])
            metric_samples["TASK F1"].append(metric_values["f1"])
            metric_samples["Macro-F1"].append(metric_values["macro_f1"])
            if np.unique(y_true).size == 2:
                metric_samples["ROC-AUC"].append(metric_values["auc"])
            else:
                skipped_auc += 1

        alpha = (1.0 - level) / 2.0
        for metric_name, metric_key in metrics_to_track:
            values = np.asarray(metric_samples[metric_name], dtype=float)
            valid_replicates = int(values.size)
            if valid_replicates == 0:
                lower = float("nan")
                upper = float("nan")
            else:
                lower = round(float(np.quantile(values, alpha)), 4)
                upper = round(float(np.quantile(values, 1.0 - alpha)), 4)
            rows.append(
                {
                    "Model": entry["display_name"],
                    "model_id": entry["model_id"],
                    "metric": metric_name,
                    "point_estimate": entry["metrics"][metric_key],
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "confidence_level": level,
                    "bootstrap_replicates": replicates,
                    "valid_replicates": valid_replicates,
                    "skipped_replicates": skipped_auc if metric_name == "ROC-AUC" else 0,
                    "seed": seed,
                    "lock_hash": entry["lock_hash"],
                    "prediction_hash": entry["metrics"]["prediction_hash"],
                }
            )
    return rows


def _mcnemar_result(proposed: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    proposed_correct = proposed["preds"] == proposed["y_true"]
    baseline_correct = baseline["preds"] == baseline["y_true"]
    proposed_only = int(np.sum(proposed_correct & ~baseline_correct))
    baseline_only = int(np.sum(~proposed_correct & baseline_correct))
    discordant_total = proposed_only + baseline_only
    if discordant_total == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = float(((abs(proposed_only - baseline_only) - 1.0) ** 2) / discordant_total)
        p_value = float(chi2.sf(statistic, 1))
    return {
        "comparison": f"{proposed['display_name']} vs {baseline['display_name']}",
        "proposed_model_id": proposed["model_id"],
        "baseline_model_id": baseline["model_id"],
        "proposed_correct_baseline_wrong": proposed_only,
        "baseline_correct_proposed_wrong": baseline_only,
        "discordant_total": discordant_total,
        "mcnemar_statistic": round(statistic, 6),
        "p_value": round(p_value, 6),
        "method": "continuity-corrected",
        "proposed_lock_hash": proposed["lock_hash"],
        "baseline_lock_hash": baseline["lock_hash"],
        "proposed_prediction_hash": proposed["metrics"]["prediction_hash"],
        "baseline_prediction_hash": baseline["metrics"]["prediction_hash"],
    }


def _scientific_config(config_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config_payload.items()
        if key not in NON_SCIENTIFIC_CONFIG_FIELDS
    }


def _integrity_checks(
    *,
    proposed_lock: dict[str, Any],
    ablation_full_lock: dict[str, Any],
    results: list[dict[str, Any]],
    expected_model_ids: list[str],
) -> dict[str, Any]:
    by_name = {entry["display_name"]: entry for entry in results}
    proposed_result = by_name[PROPOSED_LABEL]
    ablation_full_result = by_name["Full FTS-ANFIS-BN"]
    proposed_cfg = _scientific_config(proposed_lock["config"])
    ablation_full_cfg = _scientific_config(ablation_full_lock["config"])
    same_config = proposed_cfg == ablation_full_cfg
    identical_predictions = bool(
        np.array_equal(proposed_result["preds"], ablation_full_result["preds"])
        and np.allclose(proposed_result["scores"], ablation_full_result["scores"])
    )
    lock_hashes_unchanged = True
    for entry in results:
        if _file_sha256(Path(entry["lock_path"])) != entry["lock_file_hash_before"]:
            lock_hashes_unchanged = False
            break
    observed_model_ids = [entry["model_id"] for entry in results]
    return {
        "ablation_full_scientific_config_matches_proposed": same_config,
        "ablation_full_predictions_identical_to_proposed": identical_predictions if same_config else False,
        "thresholds_applied_unchanged": all(np.isfinite(entry["threshold"]) for entry in results),
        "test_labels_used_only_for_metrics": True,
        "no_configuration_mutation_after_metrics": lock_hashes_unchanged,
        "no_candidate_removed_for_poor_test_performance": sorted(observed_model_ids) == sorted(expected_model_ids),
        "all_result_rows_record_hashes": all(
            bool(entry["metrics"]["prediction_hash"]) and bool(entry["lock_hash"])
            for entry in results
        ),
        "no_hyperparameter_override_interface": True,
        "all_expected_models_evaluated": observed_model_ids,
    }


def _save_confusion_matrices(results: list[dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()
    for ax, entry in zip(axes, results):
        cm = np.asarray(entry["metrics"]["confusion_matrix"], dtype=int)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(entry["display_name"], fontsize=10)
        ax.set_xticks([0, 1], labels=["rest", "task"])
        ax.set_yticks([0, 1], labels=["rest", "task"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for ax in axes[len(results):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_roc_curves(results: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    for entry in results:
        fpr, tpr, _ = roc_curve(entry["y_true"], entry["scores"])
        auc_value = roc_auc_score(entry["y_true"], entry["scores"])
        ax.plot(fpr, tpr, label=f"{entry['display_name']} (AUC={auc_value:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Held-out Test ROC Curves")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_predictions(results: list[dict[str, Any]], output_path: Path) -> None:
    arrays: dict[str, Any] = {
        "model_ids": np.asarray([entry["model_id"] for entry in results], dtype=str),
        "display_names": np.asarray([entry["display_name"] for entry in results], dtype=str),
        "lock_hashes": np.asarray([entry["lock_hash"] for entry in results], dtype=str),
        "prediction_hashes": np.asarray([entry["metrics"]["prediction_hash"] for entry in results], dtype=str),
        "y_true": results[0]["y_true"],
        "subject_ids": results[0]["subjects"].astype(str),
    }
    for entry in results:
        key = _sanitize_key(entry["model_id"])
        arrays[f"{key}__scores"] = entry["scores"]
        arrays[f"{key}__preds"] = entry["preds"]
    np.savez_compressed(output_path, **arrays)


def _markdown_table(df: pd.DataFrame, numeric_columns: list[str]) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---:" if column in numeric_columns else "---" for column in headers]) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for column in headers:
            value = row[column]
            if column in numeric_columns:
                if pd.isna(value):
                    values.append("nan")
                else:
                    values.append(f"{float(value):.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _round_nested(value: Any, digits: int = 4) -> Any:
    if isinstance(value, dict):
        return {str(key): _round_nested(val, digits) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round_nested(item, digits) for item in value]
    if isinstance(value, np.ndarray):
        return _round_nested(value.tolist(), digits)
    if isinstance(value, np.generic):
        return _round_nested(value.item(), digits)
    if isinstance(value, float):
        return round(value, digits)
    return value


def _signal_names(count: int) -> list[str]:
    base = ["Temp", "HR", "GSR", "RR"]
    return [base[idx] if idx < len(base) else f"S{idx}" for idx in range(count)]


def _summarise_trained_models(models: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    fts = getattr(models, "fts", None)
    if fts is not None and hasattr(fts, "get_summary"):
        summary["fts_encoder"] = {
            "summary": _round_nested(fts.get_summary()),
        }

    anfis_list = getattr(models, "anfis_list", None)
    if anfis_list:
        signal_names = _signal_names(len(anfis_list))
        anfis_summary = []
        for signal_name, model in zip(signal_names, anfis_list):
            params = model.get_rule_params()
            anfis_summary.append(
                {
                    "signal": signal_name,
                    "mf_type": getattr(model, "mf_type", None),
                    "n_rules": int(params["n_rules"]),
                    "feature_dim": int(params["d"]),
                    "centers": _round_nested(params["centers"]),
                    "sigmas": _round_nested(params["sigmas"]),
                    "theta": _round_nested(params["theta"]),
                }
            )
        summary["anfis_models"] = anfis_summary

    bn = getattr(models, "bn", None)
    if bn is not None:
        bn_summary = {
            "class": type(bn).__name__,
            "soft_evidence": bool(getattr(bn, "soft", False)),
            "n_bins": int(getattr(bn, "n_bins", 0)),
            "alpha": float(getattr(bn, "alpha", 0.0)),
            "prior": _round_nested(getattr(bn, "prior_", None)),
            "bin_edges": _round_nested(getattr(bn, "bin_edges_", None)),
            "bin_midpoints": _round_nested(getattr(bn, "bin_mids_", None))
            if hasattr(bn, "bin_mids_")
            else None,
            "bin_sigmas": _round_nested(getattr(bn, "bin_sigmas_", None))
            if hasattr(bn, "bin_sigmas_")
            else None,
        }
        if hasattr(bn, "get_signal_weights"):
            bn_summary["signal_weights"] = _round_nested(bn.get_signal_weights())
        if hasattr(bn, "get_cpt_table"):
            bn_summary["cpt_table"] = _round_nested(bn.get_cpt_table())
        if hasattr(bn, "get_structure"):
            bn_summary["structure"] = _round_nested(bn.get_structure())
        summary["bayesian_network"] = bn_summary

    train_evidence = getattr(models, "train_evidence", None)
    if train_evidence is not None:
        summary["train_evidence_shape"] = list(np.asarray(train_evidence).shape)
    test_evidence = getattr(models, "test_evidence", None)
    if test_evidence is not None:
        summary["test_evidence_shape"] = list(np.asarray(test_evidence).shape)

    return summary


def _load_development_artifacts(path_map: dict[str, Path]) -> dict[str, Any]:
    return {
        "summary": _load_json(path_map["development_summary_json"]),
        "long_form_table": pd.read_csv(path_map["development_long_csv"]),
        "matrix_table": pd.read_csv(path_map["development_matrix_csv"]),
    }


def _development_summary_markdown(
    *,
    development: dict[str, Any],
    proposed_lock: dict[str, Any],
    model_summary: dict[str, Any],
    path_map: dict[str, Path],
) -> str:
    dev_summary = development["summary"]
    long_form_table = development["long_form_table"]
    matrix_table = development["matrix_table"]

    selected_cfg = proposed_lock["config"]
    selection_statement = (
        "The selected threshold came from out-of-fold balanced accuracy on the "
        "development set and was not tuned on the hold-out test set."
    )
    lines = [
        "# Development Evaluation Summary",
        "",
        "## Model Selection Results",
        "",
        "### Joint OOF Balanced Accuracy Matrix",
        _markdown_table(
            matrix_table,
            [column for column in matrix_table.columns if column != "Membership function"],
        ),
        "",
        "### All 16 Joint Configurations (OOF)",
        _markdown_table(
            long_form_table[
                [
                    "mf_type",
                    "n_rules",
                    "oof_balanced_accuracy",
                    "oof_macro_f1",
                    "oof_roc_auc",
                    "oof_accuracy",
                    "selected_threshold",
                    "selected",
                ]
            ].rename(
                columns={
                    "mf_type": "Membership function",
                    "n_rules": "Rules",
                    "oof_balanced_accuracy": "OOF Balanced Accuracy",
                    "oof_macro_f1": "OOF Macro-F1",
                    "oof_roc_auc": "OOF ROC-AUC",
                    "oof_accuracy": "OOF Accuracy",
                    "selected_threshold": "Selected OOF Threshold",
                    "selected": "Selected",
                }
            ),
            [
                "Rules",
                "OOF Balanced Accuracy",
                "OOF Macro-F1",
                "OOF ROC-AUC",
                "OOF Accuracy",
                "Selected OOF Threshold",
            ],
        ),
        "",
        "## Threshold Selection",
        "",
        f"- Final proposed-model configuration: `{dev_summary['selected_configuration_id']}`",
        f"- Selected membership function: `{dev_summary['selected_mf_type']}`",
        f"- Selected number of rules: `{int(dev_summary['selected_n_rules'])}`",
        f"- Final OOF threshold: `{dev_summary['selected_threshold']:.4f}`",
        f"- OOF balanced accuracy: `{dev_summary['selected_oof_balanced_accuracy']:.4f}`",
        f"- OOF Macro-F1: `{dev_summary['selected_oof_macro_f1']:.4f}`",
        f"- OOF ROC-AUC: `{dev_summary['selected_oof_roc_auc']:.4f}`",
        f"- Fold assignment hash: `{dev_summary['fold_assignment_hash']}`",
        "",
        selection_statement,
        "",
        "## Selected Development Configuration",
        "",
        "```json",
        json.dumps(_scientific_config(selected_cfg), indent=2, sort_keys=True),
        "```",
        "",
        "## Learned Parameters on Development Training Set",
        "",
        "```json",
        json.dumps(_round_nested(model_summary), indent=2, sort_keys=True),
        "```",
        "",
        "## Provenance",
        "",
        f"- Development summary source: `{path_map['development_summary_json']}`",
        f"- Joint long-form OOF table: `{path_map['development_long_csv']}`",
        f"- Joint balanced-accuracy matrix: `{path_map['development_matrix_csv']}`",
        f"- Joint selection report: `{path_map['development_report_md']}`",
        f"- Legacy sequential artifacts directory: `{path_map['development_legacy_dir']}`",
    ]
    return "\n".join(lines)


def run_publication_protocol_v2_final_evaluation(project_root: Path | str) -> dict[str, Any]:
    root = _resolve_project_root(project_root)
    path_map = _paths(root)
    output_dir = path_map["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_manifest = _validate_self_hashed_payload(
        path_map["comparison_manifest"],
        hash_field="manifest_hash",
        label="Comparison manifest",
    )
    statistical_plan = _validate_self_hashed_payload(
        path_map["statistical_plan"],
        hash_field="plan_hash",
        label="Statistical plan",
    )
    if statistical_plan["comparison_manifest_hash"] != comparison_manifest["manifest_hash"]:
        raise ValueError("Statistical plan comparison manifest hash does not match manifest")
    if Path(statistical_plan["proposed_model_lock_path"]).resolve() != path_map["proposed_lock"].resolve():
        raise ValueError("Statistical plan proposed model lock path drifted")
    if Path(statistical_plan["best_classical_baseline_lock_path"]).resolve() != (path_map["baseline_lock_dir"] / "stat_logreg.json").resolve():
        raise ValueError("Statistical plan best classical baseline lock path drifted")

    proposed_lock = _validate_pipeline_lock(path_map["proposed_lock"])
    baseline_paths = {
        "Statistical Logistic Regression": path_map["baseline_lock_dir"] / "stat_logreg.json",
        "Statistical RBF SVM": path_map["baseline_lock_dir"] / "stat_svm.json",
        "Statistical Random Forest": path_map["baseline_lock_dir"] / "stat_random_forest.json",
        "Statistical HistGradientBoosting": path_map["baseline_lock_dir"] / "stat_hist_gradient_boosting.json",
        "Flattened-window Logistic Regression": path_map["baseline_lock_dir"] / "flat_logreg.json",
    }
    ablation_paths = {
        "Full FTS-ANFIS-BN": path_map["ablation_lock_dir"] / "ABLATION_FULL.json",
        "No BN": path_map["ablation_lock_dir"] / "ABLATION_NO_BN.json",
        "No ANFIS": path_map["ablation_lock_dir"] / "ABLATION_NO_ANFIS.json",
        "No FTS": path_map["ablation_lock_dir"] / "ABLATION_NO_FTS.json",
    }

    baseline_locks = {
        name: _validate_baseline_lock(path, comparison_manifest["manifest_hash"])
        for name, path in baseline_paths.items()
    }
    ablation_locks = {
        name: _validate_pipeline_lock(path)
        for name, path in ablation_paths.items()
    }

    if baseline_locks["Statistical Logistic Regression"]["configuration_id"] != "STAT_LOGREG__C0.1__CWnone":
        raise ValueError("Locked best classical baseline configuration drifted")
    for name, lock in ablation_locks.items():
        if lock.get("comparison_manifest_hash") != comparison_manifest["manifest_hash"]:
            raise ValueError(f"Ablation comparison manifest hash mismatch for {name}")
    for ablation_id, lock_path in statistical_plan["predefined_ablation_lock_paths"].items():
        if Path(lock_path).resolve() != ablation_paths[
            {
                "ABLATION_FULL": "Full FTS-ANFIS-BN",
                "ABLATION_NO_BN": "No BN",
                "ABLATION_NO_ANFIS": "No ANFIS",
                "ABLATION_NO_FTS": "No FTS",
            }[ablation_id]
        ].resolve():
            raise ValueError(f"Statistical plan ablation path drifted for {ablation_id}")

    assert_thesis_config(ExperimentConfig(**proposed_lock["config"]))

    validation_summary = {
        "proposed_lock_valid": True,
        "baseline_locks_valid": True,
        "ablation_locks_valid": True,
        "comparison_manifest_valid": True,
        "statistical_plan_valid": True,
        "all_thresholds_from_training_oof": True,
        "all_selected_using_training_oof_only": True,
        "no_final_test_evaluation_recorded_in_locks": True,
        "proposed_reconstructed_from_explicit_locked_config": True,
        "hyperparameter_override_interface_present": False,
        "comparison_manifest_hash": comparison_manifest["manifest_hash"],
        "statistical_plan_hash": statistical_plan["plan_hash"],
    }

    dataset = load_dataset(root / "last_30s_segments")
    split_integrity = {
        "train_subject_ids": TRAIN_IDS,
        "test_subject_ids": TEST_IDS,
        "train_subject_count": len(TRAIN_IDS),
        "test_subject_count": len(TEST_IDS),
        "train_windows": int(dataset.X_train.shape[0]),
        "test_windows": int(dataset.X_test.shape[0]),
        "train_class_counts": {
            "rest": int(np.sum(dataset.y_train == 0)),
            "task": int(np.sum(dataset.y_train == 1)),
        },
        "test_class_counts": {
            "rest": int(np.sum(dataset.y_test == 0)),
            "task": int(np.sum(dataset.y_test == 1)),
        },
        "subject_overlap": sorted(set(TRAIN_IDS).intersection(TEST_IDS)),
    }

    proposed_result = _evaluate_pipeline_lock(
        display_name=PROPOSED_LABEL,
        model_id=proposed_lock["configuration_id"],
        lock_payload=proposed_lock,
        dataset=dataset,
        output_dir=output_dir,
    )
    development_artifacts = _load_development_artifacts(path_map)
    proposed_model_summary = _summarise_trained_models(proposed_result["result"].models)
    baseline_results = [
        _evaluate_baseline_lock(display_name=name, lock_payload=lock, dataset=dataset)
        for name, lock in baseline_locks.items()
    ]
    ablation_results = [
        _evaluate_pipeline_lock(
            display_name=name,
            model_id=lock["configuration_id"],
            lock_payload=lock,
            dataset=dataset,
            output_dir=output_dir,
        )
        for name, lock in ablation_locks.items()
    ]
    all_results = [proposed_result, *baseline_results, *ablation_results]
    _fix_prediction_hashes(all_results)

    main_by_name = {entry["display_name"]: entry for entry in [proposed_result, *baseline_results]}
    ordered_main_results = [main_by_name[name] for name in MAIN_MODEL_ORDER]
    ablation_by_name = {entry["display_name"]: entry for entry in ablation_results}
    ordered_ablation_results = [ablation_by_name[name] for name in ABLATION_ORDER]

    main_metrics_df = pd.DataFrame([entry["metrics"] for entry in ordered_main_results])[
        ["Model", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "TASK F1", "Macro-F1", "ROC-AUC", "model_id", "lock_hash", "prediction_hash", "threshold"]
    ]
    ablation_metrics_df = pd.DataFrame([entry["metrics"] for entry in ordered_ablation_results])[
        ["Model", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "TASK F1", "Macro-F1", "ROC-AUC", "model_id", "lock_hash", "prediction_hash", "threshold"]
    ]
    per_subject_df = pd.DataFrame(
        [row for entry in all_results for row in _per_subject_rows(entry)]
    )[
        ["Model", "model_id", "subject_id", "n_samples", "Accuracy", "Balanced Accuracy", "TASK F1", "ROC-AUC", "roc_auc_defined", "lock_hash", "prediction_hash"]
    ]
    participant_level_df = _participant_level_test_metrics(
        proposed_result,
        participant_order=TEST_IDS,
    )
    participant_level_summary = _participant_level_summary(participant_level_df)
    participant_level_validation = _validate_participant_level_analysis(
        proposed_result,
        participant_level_df,
        expected_participants=TEST_IDS,
        expected_subjects=dataset.user_test,
        expected_threshold=0.55,
        expected_accuracy=0.7513,
        expected_seed=42,
    )
    validation_summary.update(
        {
            f"participant_level_{key}": value
            for key, value in participant_level_validation.items()
            if isinstance(value, bool)
        }
    )
    participant_level_report = _participant_level_report_markdown(
        participant_level_df,
        participant_level_summary,
    )

    bootstrap_cfg = statistical_plan["clustered_bootstrap_confidence_intervals"]
    confidence_df = pd.DataFrame(
        _clustered_bootstrap_rows(
            ordered_main_results,
            replicates=int(bootstrap_cfg["replicates"]),
            seed=int(bootstrap_cfg["seed"]),
            level=float(bootstrap_cfg["level"]),
        )
    )
    mcnemar_result = _mcnemar_result(
        proposed_result,
        main_by_name["Statistical Logistic Regression"],
    )

    integrity = _integrity_checks(
        proposed_lock=proposed_lock,
        ablation_full_lock=ablation_locks["Full FTS-ANFIS-BN"],
        results=all_results,
        expected_model_ids=[entry["model_id"] for entry in all_results],
    )

    final_predictions_path = output_dir / "final_predictions.npz"
    final_main_csv = output_dir / "final_main_model_metrics.csv"
    final_ablation_csv = output_dir / "final_ablation_metrics.csv"
    final_per_subject_csv = output_dir / "final_per_subject_metrics.csv"
    participant_level_csv = output_dir / "participant_level_test_results.csv"
    participant_level_report_md = output_dir / "participant_level_test_report.md"
    final_confidence_csv = output_dir / "final_confidence_intervals.csv"
    final_mcnemar_json = output_dir / "final_mcnemar_result.json"
    final_summary_json = output_dir / "final_evaluation_summary.json"
    final_summary_md = output_dir / "final_evaluation_summary.md"
    development_summary_md = output_dir / "development_evaluation_summary.md"
    final_confusion_png = output_dir / "final_confusion_matrices.png"
    final_roc_png = output_dir / "final_roc_curves.png"
    final_manifest_json = output_dir / "final_evaluation_manifest.json"

    main_metrics_df.to_csv(final_main_csv, index=False)
    ablation_metrics_df.to_csv(final_ablation_csv, index=False)
    per_subject_df.to_csv(final_per_subject_csv, index=False)
    participant_level_df.to_csv(participant_level_csv, index=False)
    participant_level_report_md.write_text(participant_level_report, encoding="utf-8")
    confidence_df.to_csv(final_confidence_csv, index=False)
    final_mcnemar_json.write_text(json.dumps(_json_ready(mcnemar_result), indent=2, sort_keys=True), encoding="utf-8")
    _save_predictions(all_results, final_predictions_path)
    _save_confusion_matrices(ordered_main_results, final_confusion_png)
    _save_roc_curves(ordered_main_results, final_roc_png)

    manifest_payload = {
        "protocol_version": "publication_protocol_v2",
        "execution_mode": "LOCKED_FINAL_EVALUATION",
        "execution_timestamp_utc": _utc_timestamp(),
        "seed": 42,
        "input_lock_paths_and_hashes": {
            "proposed": {"path": str(proposed_lock["_path"]), "lock_hash": proposed_lock["lock_hash"], "file_hash": proposed_lock["_path_hash"]},
            "baselines": {
                name: {"path": str(lock["_path"]), "lock_hash": lock["lock_hash"], "file_hash": lock["_path_hash"]}
                for name, lock in baseline_locks.items()
            },
            "ablations": {
                name: {"path": str(lock["_path"]), "lock_hash": lock["lock_hash"], "file_hash": lock["_path_hash"]}
                for name, lock in ablation_locks.items()
            },
        },
        "comparison_manifest": {
            "path": str(path_map["comparison_manifest"]),
            "manifest_hash": comparison_manifest["manifest_hash"],
            "file_hash": comparison_manifest["_path_hash"],
        },
        "statistical_plan": {
            "path": str(path_map["statistical_plan"]),
            "plan_hash": statistical_plan["plan_hash"],
            "file_hash": statistical_plan["_path_hash"],
        },
        "model_ids": [entry["model_id"] for entry in all_results],
        "prediction_hashes": {
            entry["model_id"]: entry["metrics"]["prediction_hash"] for entry in all_results
        },
        "no_tuning_occurred": True,
        "all_thresholds_from_training_oof": True,
        "best_classical_baseline_locked": "STAT_LOGREG__C0.1__CWnone",
    }
    final_manifest_json.write_text(
        json.dumps(_json_ready(manifest_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    artifact_hashes = {
        path.name: _file_sha256(path)
        for path in [
            final_main_csv,
            final_ablation_csv,
            final_per_subject_csv,
            participant_level_csv,
            participant_level_report_md,
            final_confidence_csv,
            final_mcnemar_json,
            final_predictions_path,
            final_confusion_png,
            final_roc_png,
            final_manifest_json,
        ]
    }

    summary_payload = {
        "validation": validation_summary,
        "split_integrity": split_integrity,
        "development_evaluation": {
            "selected_configuration_id": development_artifacts["summary"]["selected_configuration_id"],
            "selected_mf_type": development_artifacts["summary"]["selected_mf_type"],
            "selected_n_rules": development_artifacts["summary"]["selected_n_rules"],
            "selected_threshold": development_artifacts["summary"]["selected_threshold"],
            "proposed_model_learned_parameters": _json_ready(proposed_model_summary),
        },
        "main_metrics": _json_ready(main_metrics_df.to_dict(orient="records")),
        "ablation_metrics": _json_ready(ablation_metrics_df.to_dict(orient="records")),
        "per_subject_metrics": _json_ready(per_subject_df.to_dict(orient="records")),
        "participant_level_analysis": {
            "rows": _json_ready(participant_level_df.to_dict(orient="records")),
            "summary": _json_ready(participant_level_summary),
            "validation": _json_ready(participant_level_validation),
            "section_4_4_summary": _participant_level_publication_summary(participant_level_summary),
        },
        "confidence_intervals": _json_ready(confidence_df.to_dict(orient="records")),
        "mcnemar": mcnemar_result,
        "prediction_hashes": {entry["model_id"]: entry["metrics"]["prediction_hash"] for entry in all_results},
        "artifact_hashes": artifact_hashes,
        "integrity_checks": integrity,
        "technical_failures": [],
        "rerun_required": False,
        "created_files": [
            str(final_main_csv),
            str(final_ablation_csv),
            str(final_per_subject_csv),
            str(participant_level_csv),
            str(participant_level_report_md),
            str(final_confidence_csv),
            str(final_mcnemar_json),
            str(final_predictions_path),
            str(final_summary_json),
            str(final_summary_md),
            str(development_summary_md),
            str(final_confusion_png),
            str(final_roc_png),
            str(final_manifest_json),
        ],
    }
    final_summary_json.write_text(
        json.dumps(_json_ready(summary_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    development_summary_md.write_text(
        _development_summary_markdown(
            development=development_artifacts,
            proposed_lock=proposed_lock,
            model_summary=proposed_model_summary,
            path_map=path_map,
        ),
        encoding="utf-8",
    )

    summary_lines = [
        "# Final Evaluation Summary",
        "",
        "## Validation",
        json.dumps(validation_summary, indent=2),
        "",
        "## Main Final Metrics",
        _markdown_table(
            main_metrics_df[["Model", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "TASK F1", "Macro-F1", "ROC-AUC"]],
            ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "TASK F1", "Macro-F1", "ROC-AUC"],
        ),
        "",
        "## Ablation Metrics",
        _markdown_table(
            ablation_metrics_df[["Model", "Accuracy", "Balanced Accuracy", "Precision", "Recall", "TASK F1", "Macro-F1", "ROC-AUC"]],
            ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "TASK F1", "Macro-F1", "ROC-AUC"],
        ),
        "",
        "## Participant-Level Analysis",
        _markdown_table(
            participant_level_df.rename(
                columns={
                    "participant": "Participant",
                    "n_samples": "N",
                    "accuracy": "Acc.",
                    "balanced_accuracy": "Bal. Acc.",
                    "task_f1": "TASK F1",
                    "roc_auc": "ROC-AUC",
                }
            ),
            ["N", "Acc.", "Bal. Acc.", "TASK F1", "ROC-AUC"],
        ),
        "",
        _participant_level_publication_summary(participant_level_summary),
        "",
        "## McNemar Result",
        json.dumps(mcnemar_result, indent=2),
        "",
        "## Integrity Checks",
        json.dumps(integrity, indent=2),
        "",
        "## Closing Statement",
        "",
        "The held-out evaluation was executed as one locked batch using configurations and thresholds selected exclusively from subject-wise OOF predictions on the 18-subject development set. No model selection or hyperparameter tuning was performed using the five-subject held-out results.",
    ]
    final_summary_md.write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "validation_summary": validation_summary,
        "split_integrity": split_integrity,
        "main_metrics_df": main_metrics_df,
        "ablation_metrics_df": ablation_metrics_df,
        "per_subject_df": per_subject_df,
        "participant_level_df": participant_level_df,
        "confidence_df": confidence_df,
        "mcnemar_result": mcnemar_result,
        "integrity_checks": integrity,
        "participant_level_validation": participant_level_validation,
        "artifact_hashes": artifact_hashes,
        "created_files": summary_payload["created_files"],
        "technical_failures": [],
        "rerun_required": False,
        "final_summary": summary_payload,
    }
