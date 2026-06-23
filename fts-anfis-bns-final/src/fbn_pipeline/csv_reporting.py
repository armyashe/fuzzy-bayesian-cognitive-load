from __future__ import annotations

import csv
import os
from typing import Any

from .results import DevelopmentResult, FinalEvaluationResult

CSV_COLUMNS = [
    # --- Experiment identification ---
    "experiment_group", "variant_name", "preset_name",
    "exp",               # legacy alias for variant_name / run label
    "eval_protocol",
    "ablation", "seed", "k_folds",
    # --- FTS / Stage 1 ---
    "n_sets", "mf_type", "fts_mf_type", "mf_sigma_scale", "gbell_b", "fts_norm",
    "trap_plateau_scale", "trap_width_scale", "grid_margin_scale",
    # --- ANFIS / Stage 2 ---
    "anfis_mf_type", "anfis_gbell_b",
    "n_rules",
    "sigma_scale",       # required name
    "anfis_sigma_scale", # legacy alias
    "ridge_lambda", "n_epochs", "lr",
    # --- BN / Stage 3 ---
    "n_bins", "alpha", "bn_structure", "bn_soft", "bn_reliability", "calibrate",
    # --- Extra feature-engineering flags ---
    "scale_fts", "robust_fts",
    # --- Ablation metadata ---
    "fusion_mode_when_no_bn", "no_stage1_feature_mode", "fts_summary_mode",
    "final_seed_offset",
    # --- Threshold selection ---
    "threshold_metric",  # required name
    "thr_metric",        # legacy alias
    "best_threshold",    # required name
    "best_thr",          # legacy alias
    "rank_metric",
    # --- OOF metrics (required) ---
    "oof_accuracy", "oof_precision", "oof_recall", "oof_f1", "oof_auc",
    # --- OOF metrics (legacy) ---
    "oof_acc", "oof_bal_acc", "oof_macro_f1",
    # --- Held-out test metrics (required) ---
    "test_accuracy", "test_precision", "test_recall", "test_f1", "test_auc",
    # --- Held-out test metrics (legacy) ---
    "test_acc", "test_bal_acc", "test_macro_f1",
]

DEVELOPMENT_CSV_COLUMNS = [
    "configuration_id",
    "experiment_group",
    "variant_name",
    "preset_name",
    "selection_rank",
    "selected",
    "selection_metric",
    "selection_metric_value",
    "tie_break_rules",
    "selected_threshold",
    "threshold_metric",
    "threshold_score",
    "seed",
    "k_folds",
    "n_sets",
    "mf_type",
    "anfis_mf_type",
    "n_rules",
    "n_bins",
    "alpha",
    "bn_structure",
    "bn_soft",
    "bn_reliability",
    "calibrate",
    "ablation",
    "oof_accuracy",
    "oof_bal_acc",
    "oof_macro_f1",
    "oof_f1",
    "oof_auc",
]

FINAL_TEST_CSV_COLUMNS = [
    "configuration_id",
    "locked_config_path",
    "threshold",
    "test_accuracy",
    "test_bal_acc",
    "test_macro_f1",
    "test_f1",
    "test_auc",
]


def log_to_csv(filepath: str, row: dict[str, Any]) -> None:
    _log_to_csv(filepath, row, CSV_COLUMNS)


def _log_to_csv(filepath: str, row: dict[str, Any], columns: list[str]) -> None:
    parent = os.path.dirname(filepath)
    if parent:
        os.makedirs(parent, exist_ok=True)
    file_exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def log_development_row(filepath: str, row: dict[str, Any]) -> None:
    _log_to_csv(filepath, row, DEVELOPMENT_CSV_COLUMNS)


def log_final_test_row(filepath: str, row: dict[str, Any]) -> None:
    _log_to_csv(filepath, row, FINAL_TEST_CSV_COLUMNS)


def make_result_row(
    cfg: dict[str, Any],
    exp_name: str,
    best_thr: float,
    *,
    oof: dict[str, Any] | None = None,
    test: dict[str, Any] | None = None,
    eval_protocol: str = "",
    rank_metric: str = "",
    thr_metric: str = "",
    preset_name: str = "",
) -> dict[str, Any]:
    oof = oof or {}
    test = test or {}
    variant = cfg.get("variant_name", "") or exp_name
    grp = cfg.get("experiment_group", "")
    return {
        # identification
        "experiment_group": grp,
        "variant_name": variant,
        "preset_name": preset_name or cfg.get("variant_name", "") or exp_name,
        "exp": exp_name,
        "eval_protocol": eval_protocol,
        "ablation": cfg.get("ablation", "exp1"),
        "seed": cfg.get("seed", ""),
        "k_folds": cfg.get("k_folds", ""),
        # FTS
        "n_sets": cfg["n_sets"],
        "mf_type": cfg["mf_type"],
        "fts_mf_type": cfg["mf_type"],
        "mf_sigma_scale": cfg["mf_sigma_scale"],
        "gbell_b": cfg["gbell_b"],
        "fts_norm": cfg["fts_norm"],
        "trap_plateau_scale": cfg.get("trap_plateau_scale", 0.15),
        "trap_width_scale": cfg.get("trap_width_scale", 1.0),
        "grid_margin_scale": cfg.get("grid_margin_scale", 0.05),
        # ANFIS
        "anfis_mf_type": cfg.get("anfis_mf_type", "gaussian"),
        "anfis_gbell_b": cfg.get("anfis_gbell_b", 2.0),
        "n_rules": cfg["n_rules"],
        "sigma_scale": cfg["sigma_scale"],
        "anfis_sigma_scale": cfg["sigma_scale"],  # legacy alias
        "ridge_lambda": cfg["ridge_lambda"],
        "n_epochs": cfg["n_epochs"],
        "lr": cfg["lr"],
        # BN
        "n_bins": cfg["n_bins"],
        "alpha": cfg["alpha"],
        "bn_structure": cfg.get("bn_structure", "naive"),
        "bn_soft": cfg.get("bn_soft", False),
        "bn_reliability": cfg.get("bn_reliability", False),
        "calibrate": cfg.get("calibrate", "none"),
        # feature-engineering flags
        "scale_fts": cfg.get("scale_fts", False),
        "robust_fts": cfg.get("robust_fts", False),
        # ablation metadata
        "fusion_mode_when_no_bn": cfg.get("fusion_mode_when_no_bn", "mean"),
        "no_stage1_feature_mode": cfg.get("no_stage1_feature_mode", "stats"),
        "fts_summary_mode": cfg.get("fts_summary_mode", "diag_high"),
        "final_seed_offset": cfg.get("final_seed_offset", 0),
        # threshold
        "threshold_metric": thr_metric,
        "thr_metric": thr_metric,
        "best_threshold": best_thr,
        "best_thr": best_thr,
        "rank_metric": rank_metric,
        # OOF required
        "oof_accuracy": oof.get("accuracy", oof.get("acc", "")),
        "oof_precision": oof.get("precision", ""),
        "oof_recall": oof.get("recall", ""),
        "oof_f1": oof.get("f1", ""),
        "oof_auc": oof.get("auc", ""),
        # OOF legacy
        "oof_acc": oof.get("acc", oof.get("accuracy", "")),
        "oof_bal_acc": oof.get("bal_acc", ""),
        "oof_macro_f1": oof.get("macro_f1", ""),
        # test required
        "test_accuracy": test.get("accuracy", test.get("acc", "")),
        "test_precision": test.get("precision", ""),
        "test_recall": test.get("recall", ""),
        "test_f1": test.get("f1", ""),
        "test_auc": test.get("auc", ""),
        # test legacy
        "test_acc": test.get("acc", test.get("accuracy", "")),
        "test_bal_acc": test.get("bal_acc", ""),
        "test_macro_f1": test.get("macro_f1", ""),
    }


def make_development_result_row(
    result: DevelopmentResult,
    *,
    selection_rank: int | None = None,
    selected: bool = False,
    selection_metric: str = "oof_balanced_accuracy",
    tie_break_rules: list[str] | None = None,
) -> dict[str, Any]:
    cfg = result.config.to_dict()
    oof = result.oof_metrics
    return {
        "configuration_id": result.configuration_id,
        "experiment_group": cfg.get("experiment_group", ""),
        "variant_name": cfg.get("variant_name", "") or result.configuration_id,
        "preset_name": cfg.get("variant_name", "") or result.configuration_id,
        "selection_rank": selection_rank if selection_rank is not None else "",
        "selected": bool(selected),
        "selection_metric": selection_metric,
        "selection_metric_value": result.selection_metric_value,
        "tie_break_rules": " > ".join(tie_break_rules or []),
        "selected_threshold": result.threshold,
        "threshold_metric": result.threshold_metric,
        "threshold_score": result.threshold_score,
        "seed": cfg.get("seed", ""),
        "k_folds": cfg.get("k_folds", ""),
        "n_sets": cfg.get("n_sets", ""),
        "mf_type": cfg.get("mf_type", ""),
        "anfis_mf_type": cfg.get("anfis_mf_type", ""),
        "n_rules": cfg.get("n_rules", ""),
        "n_bins": cfg.get("n_bins", ""),
        "alpha": cfg.get("alpha", ""),
        "bn_structure": cfg.get("bn_structure", ""),
        "bn_soft": cfg.get("bn_soft", ""),
        "bn_reliability": cfg.get("bn_reliability", ""),
        "calibrate": cfg.get("calibrate", ""),
        "ablation": cfg.get("ablation", ""),
        "oof_accuracy": oof.get("accuracy", ""),
        "oof_bal_acc": oof.get("bal_acc", ""),
        "oof_macro_f1": oof.get("macro_f1", ""),
        "oof_f1": oof.get("f1", ""),
        "oof_auc": oof.get("auc", ""),
    }


def make_final_test_result_row(
    configuration_id: str,
    result: FinalEvaluationResult,
) -> dict[str, Any]:
    metrics = result.test_metrics
    return {
        "configuration_id": configuration_id,
        "locked_config_path": result.locked_config_path,
        "threshold": result.threshold,
        "test_accuracy": metrics.get("accuracy", ""),
        "test_bal_acc": metrics.get("bal_acc", ""),
        "test_macro_f1": metrics.get("macro_f1", ""),
        "test_f1": metrics.get("f1", ""),
        "test_auc": metrics.get("auc", ""),
    }

