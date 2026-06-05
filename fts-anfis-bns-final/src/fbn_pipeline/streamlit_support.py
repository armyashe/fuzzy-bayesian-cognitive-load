from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import EXPERIMENT_PRESETS, ExperimentConfig, apply_preset, setup_seed
from .data import load_dataset
from .metrics import find_best_threshold, per_subject_metrics
from .pipeline import compute_oof_metrics, retrain_and_test, run_cv
from .bn_dashboard import export_bn_dashboard_artifacts


@dataclass(frozen=True, slots=True)
class ExperimentOption:
    key: str
    label: str
    help_text: str = ""


@dataclass(frozen=True, slots=True)
class ExperimentSelection:
    group_key: str
    group_label: str
    variant_key: str
    variant_label: str
    preset_name: str
    base_preset_name: str
    run_name: str
    summary: str
    supports_stage_demo: bool
    n_rules_override: int | None = None


GROUP_OPTIONS: tuple[ExperimentOption, ...] = (
    ExperimentOption(
        key="exp1",
        label="Exp 1 — FTS membership function comparison",
        help_text="Compare fuzzy membership functions and fuzzy partitions in the FTS encoder.",
    ),
    ExperimentOption(
        key="exp2",
        label="Exp 2 — ANFIS rule-number experiment",
        help_text="Keep the full pipeline and override the ANFIS rule count.",
    ),
    ExperimentOption(
        key="exp3",
        label="Exp 3 — Stage contribution / ablation experiment",
        help_text="Compare the contribution of major pipeline stages using the ablation logic already defined in source.",
    ),
)


GROUP1_VARIANTS: tuple[ExperimentOption, ...] = (
    ExperimentOption("A1", "A1 — Triangular MF"),
    ExperimentOption("A2", "A2 — Gaussian MF"),
    ExperimentOption("A3", "A3 — Trapezoid MF"),
    ExperimentOption("A4", "A4 — Generalized bell MF"),
    ExperimentOption("A5", "A5 — Gaussian MF with 4 fuzzy sets"),
)


GROUP3_VARIANTS: tuple[ExperimentOption, ...] = (
    ExperimentOption("EXP3_FULL", "Full pipeline — FTS → ANFIS → BN"),
    ExperimentOption("EXP3_NO_BN", "Without BN — FTS → ANFIS only"),
    ExperimentOption("EXP3_NO_ANFIS", "Without ANFIS — FTS summary → BN"),
)


RULE_COUNT_OPTIONS: tuple[int, ...] = (2, 4, 6, 8, 10, 12)


def get_group_options() -> tuple[ExperimentOption, ...]:
    return GROUP_OPTIONS


def get_variant_options(group_key: str) -> tuple[ExperimentOption, ...]:
    if group_key == "exp1":
        return GROUP1_VARIANTS
    if group_key == "exp3":
        return GROUP3_VARIANTS
    return tuple()


def get_group_label(group_key: str) -> str:
    for option in GROUP_OPTIONS:
        if option.key == group_key:
            return option.label
    raise KeyError(group_key)


def get_group_help(group_key: str) -> str:
    for option in GROUP_OPTIONS:
        if option.key == group_key:
            return option.help_text
    return ""


def get_group1_default_variant() -> str:
    return "A2"


def get_group2_default_base_preset() -> str:
    return "A2_soft" if "A2_soft" in EXPERIMENT_PRESETS else "A2"


def get_group3_default_variant() -> str:
    return "EXP3_FULL"


def group2_variant_label(base_preset_name: str) -> str:
    cfg = apply_preset(ExperimentConfig(), base_preset_name)
    return (
        f"{base_preset_name} — {cfg.mf_type.capitalize()} FTS + ANFIS + "
        f"{'soft BN' if cfg.bn_soft else 'hard BN'}"
    )


def resolve_selection(
    *,
    group_key: str,
    variant_key: str | None,
    base_preset_name: str,
    n_rules_override: int | None,
) -> ExperimentSelection:
    group_label = get_group_label(group_key)

    if group_key == "exp1":
        preset_name = variant_key or get_group1_default_variant()
        cfg = apply_preset(ExperimentConfig(), preset_name)
        variant_label = next(opt.label for opt in GROUP1_VARIANTS if opt.key == preset_name)
        return ExperimentSelection(
            group_key=group_key,
            group_label=group_label,
            variant_key=preset_name,
            variant_label=variant_label,
            preset_name=preset_name,
            base_preset_name=preset_name,
            run_name=preset_name,
            summary=f"{variant_label}; full pipeline with preset-defined settings.",
            supports_stage_demo=True,
        )

    if group_key == "exp2":
        rules = int(n_rules_override or 6)
        variant_label = group2_variant_label(base_preset_name)
        return ExperimentSelection(
            group_key=group_key,
            group_label=group_label,
            variant_key=f"EXP2_RULES_{rules}",
            variant_label=variant_label,
            preset_name=base_preset_name,
            base_preset_name=base_preset_name,
            run_name=f"{base_preset_name}_rules_{rules}",
            summary=f"Base preset {base_preset_name} with ANFIS rules overridden to {rules}.",
            supports_stage_demo=True,
            n_rules_override=rules,
        )

    if group_key == "exp3":
        preset_name = variant_key or get_group3_default_variant()
        variant_label = next(opt.label for opt in GROUP3_VARIANTS if opt.key == preset_name)
        supports_stage_demo = preset_name == "EXP3_FULL"
        return ExperimentSelection(
            group_key=group_key,
            group_label=group_label,
            variant_key=preset_name,
            variant_label=variant_label,
            preset_name=preset_name,
            base_preset_name=preset_name,
            run_name=preset_name,
            summary="Stage contribution experiment using the preset logic already defined in source.",
            supports_stage_demo=supports_stage_demo,
        )

    raise ValueError(f"Unsupported group_key: {group_key}")


def build_config_from_selection(
    *,
    data_path: str,
    selection: ExperimentSelection,
) -> ExperimentConfig:
    cfg = ExperimentConfig(data_path=data_path)
    cfg = apply_preset(cfg, selection.preset_name)

    overrides: dict[str, Any] = {
        "data_path": data_path,
        "experiment_group": selection.group_key.upper(),
        "variant_name": selection.variant_key,
    }
    if selection.group_key == "exp2" and selection.n_rules_override is not None:
        overrides["ablation"] = "exp1"
        overrides["n_rules"] = selection.n_rules_override
    return cfg.merged(**overrides)


def config_summary_rows(cfg: ExperimentConfig, selected_threshold: float) -> list[dict[str, Any]]:
    return [
        {"parameter": "MF type", "value": cfg.mf_type},
        {"parameter": "Number of fuzzy sets", "value": cfg.n_sets},
        {"parameter": "FTS normalization", "value": cfg.fts_norm},
        {"parameter": "ANFIS rules", "value": cfg.n_rules},
        {"parameter": "BN bins", "value": cfg.n_bins},
        {"parameter": "BN soft evidence", "value": "enabled" if cfg.bn_soft else "disabled"},
        {"parameter": "Reliability weighting", "value": "enabled" if cfg.bn_reliability else "disabled"},
        {"parameter": "Calibration", "value": cfg.calibrate},
        {"parameter": "Threshold metric", "value": cfg.threshold_metric},
        {"parameter": "Selected threshold", "value": round(float(selected_threshold), 4)},
    ]


def run_streamlit_experiment(
    *,
    project_root: Path,
    data_path: str,
    selection: ExperimentSelection,
) -> dict[str, Any]:
    cfg = build_config_from_selection(data_path=data_path, selection=selection)
    setup_seed(cfg.seed)

    dataset = load_dataset(project_root / cfg.data_path)

    oof_post, oof_y, fold_indices = run_cv(
        dataset.X_train,
        dataset.y_train,
        dataset.user_train,
        cfg,
    )

    best_thr, thr_score, thresholds, scores = find_best_threshold(
        oof_y,
        oof_post,
        metric=cfg.threshold_metric,
        n_grid=cfg.threshold_grid,
        return_sweep=True,
    )

    oof_metrics = compute_oof_metrics(oof_y, oof_post, best_thr)
    oof_pred = (oof_post >= best_thr).astype(int)

    test_pred, test_post, test_metrics, models = retrain_and_test(
        dataset.X_train,
        dataset.y_train,
        dataset.X_test,
        dataset.y_test,
        best_thr,
        cfg,
    )

    subject_rows = per_subject_metrics(
        dataset.y_test,
        test_pred,
        test_post,
        dataset.user_test,
    )

    try:
        artifact_results = export_bn_dashboard_artifacts(
            models=models,
            cfg=cfg.to_dict(),
            exp_name=selection.run_name,
            best_threshold=best_thr,
            test_metrics=test_metrics,
            test_probs=models["te_probs"],
            y_test=dataset.y_test,
            user_test=dataset.user_test,
            artifacts_dir=project_root / "artifacts",
        )
    except Exception as exc:
        artifact_results = {"error": str(exc)}

    return {
        "cfg": cfg,
        "dataset": dataset,
        "oof_post": oof_post,
        "oof_y": oof_y,
        "oof_pred": oof_pred,
        "fold_indices": fold_indices,
        "best_thr": best_thr,
        "thr_score": thr_score,
        "thresholds": thresholds,
        "scores": scores,
        "oof_metrics": oof_metrics,
        "test_pred": test_pred,
        "test_post": test_post,
        "test_metrics": test_metrics,
        "models": models,
        "subject_df": pd.DataFrame(subject_rows),
        "subject_rows": subject_rows,
        "exp_name": selection.run_name,
        "artifact_results": artifact_results,
        "selection": selection,
        "config_summary_rows": config_summary_rows(cfg, best_thr),
    }
