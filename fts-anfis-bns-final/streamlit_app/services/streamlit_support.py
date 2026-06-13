from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fbn_pipeline.ablation import Ablation
from fbn_pipeline.config import ExperimentConfig, setup_seed
from fbn_pipeline.data import load_dataset
from fbn_pipeline.metrics import find_best_threshold
from fbn_pipeline.presets import EXPERIMENT_PRESETS, apply_preset
from fbn_pipeline.runner import run_one_experiment


THESIS_FINAL_PRESET = "THESIS_FINAL"
A2_SOFT_PRESET = THESIS_FINAL_PRESET


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
    ExperimentOption("EXP1_MF_TRI", "Triangular FTS MF"),
    ExperimentOption("EXP1_MF_GAUSS", "Gaussian FTS MF (THESIS_FINAL baseline)"),
    ExperimentOption("EXP1_MF_TRAP", "Trapezoid FTS MF"),
    ExperimentOption("EXP1_MF_GBELL", "Generalized bell FTS MF"),
)


GROUP3_VARIANTS: tuple[ExperimentOption, ...] = (
    ExperimentOption("EXP3_FULL", "Full pipeline — FTS → ANFIS → BN"),
    ExperimentOption("EXP3_NO_BN", "Without BN — FTS → ANFIS only"),
    ExperimentOption("EXP3_NO_ANFIS", "Without ANFIS — FTS summary → BN"),
)


RULE_COUNT_OPTIONS: tuple[int, ...] = (3, 6, 8, 12)
EXP2_VARIANT_BY_RULES: dict[int, str] = {
    3: "EXP2_RULES_3",
    6: "EXP2_RULES_6",
    8: "EXP2_RULES_8",
    12: "EXP2_RULES_12",
}

STREAMLIT_EXPERIMENT_MAP: dict[str, tuple[str, ...]] = {
    "exp1": tuple(option.key for option in GROUP1_VARIANTS),
    "exp2": tuple(EXP2_VARIANT_BY_RULES.values()),
    "exp3": tuple(option.key for option in GROUP3_VARIANTS),
}


def get_a2_soft_config(data_path: str = "last_30s_segments") -> ExperimentConfig:
    if A2_SOFT_PRESET not in EXPERIMENT_PRESETS:
        raise ValueError(f"Required preset {A2_SOFT_PRESET} is not defined in EXPERIMENT_PRESETS.")
    return apply_preset(ExperimentConfig(data_path=data_path), A2_SOFT_PRESET)


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
    return "EXP1_MF_GAUSS"


def get_group2_default_base_preset() -> str:
    return A2_SOFT_PRESET


def get_group3_default_variant() -> str:
    return "EXP3_FULL"


def group2_variant_label(base_preset_name: str) -> str:
    if base_preset_name != A2_SOFT_PRESET:
        raise ValueError(f"Streamlit dashboard is locked to {A2_SOFT_PRESET}.")
    cfg = get_a2_soft_config()
    return (
        f"{base_preset_name} - {cfg.mf_type.capitalize()} FTS + "
        f"{cfg.anfis_mf_type.capitalize()} ANFIS + "
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
        variant_label = next(opt.label for opt in GROUP1_VARIANTS if opt.key == preset_name)
        return ExperimentSelection(
            group_key=group_key,
            group_label=group_label,
            variant_key=preset_name,
            variant_label=variant_label,
            preset_name=preset_name,
            base_preset_name=A2_SOFT_PRESET,
            run_name=preset_name,
            summary=f"{variant_label}; all other hyperparameters follow {A2_SOFT_PRESET}.",
            supports_stage_demo=True,
        )

    if group_key == "exp2":
        rules = int(n_rules_override or 6)
        if base_preset_name != A2_SOFT_PRESET:
            raise ValueError(f"Streamlit dashboard is locked to {A2_SOFT_PRESET}.")
        preset_name = EXP2_VARIANT_BY_RULES[rules]
        variant_label = f"{preset_name} - {rules} ANFIS rules"
        return ExperimentSelection(
            group_key=group_key,
            group_label=group_label,
            variant_key=preset_name,
            variant_label=variant_label,
            preset_name=preset_name,
            base_preset_name=A2_SOFT_PRESET,
            run_name=preset_name,
            summary=f"{A2_SOFT_PRESET} hyperparameters with only ANFIS n_rules changed to {rules}.",
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
            base_preset_name=A2_SOFT_PRESET,
            run_name=preset_name,
            summary="Stage contribution experiment using the EXP3 preset logic already defined in source.",
            supports_stage_demo=supports_stage_demo,
        )

    raise ValueError(f"Unsupported group_key: {group_key}")


def build_config_from_selection(
    *,
    data_path: str,
    selection: ExperimentSelection,
) -> ExperimentConfig:
    if selection.group_key not in STREAMLIT_EXPERIMENT_MAP:
        raise ValueError(f"Unsupported group_key: {selection.group_key}")
    if selection.preset_name not in STREAMLIT_EXPERIMENT_MAP[selection.group_key]:
        raise ValueError(
            f"Preset {selection.preset_name!r} is not valid for Streamlit group "
            f"{selection.group_key!r}."
        )

    cfg = ExperimentConfig(data_path=data_path)
    cfg = apply_preset(cfg, selection.preset_name)

    overrides: dict[str, Any] = {
        "data_path": data_path,
        "experiment_group": cfg.experiment_group or selection.group_key.upper(),
        "variant_name": cfg.variant_name or selection.variant_key,
    }
    return cfg.merged(**overrides)


def config_summary_rows(cfg: ExperimentConfig, selected_threshold: float) -> list[dict[str, Any]]:
    return [
        {"parameter": "FTS MF type", "value": cfg.mf_type},
        {"parameter": "ANFIS MF type", "value": cfg.anfis_mf_type},
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

    result = run_one_experiment(
        cfg,
        dataset,
        exp_name=selection.run_name,
        preset_name=selection.preset_name,
        run_name=selection.run_name,
        show_per_subject=True,
        dump_rules=cfg.dump_rules,
        export_artifacts=True,
        artifacts_dir=project_root / "artifacts",
        print_progress=False,
        print_configuration=False,
        pipeline_desc=selection.summary,
    )
    _best_thr, _thr_score, thresholds, scores = find_best_threshold(
        dataset.y_train,
        result.oof_post,
        metric=cfg.threshold_metric,
        n_grid=cfg.threshold_grid,
        return_sweep=True,
    )

    subject_rows = result.subject_rows or []
    models = result.models.as_dict()
    artifact = {
        "config": cfg,
        "experiment_name": selection.run_name,
        "variant": selection.variant_key,
        "oof_probabilities": result.oof_post,
        "selected_threshold": result.best_threshold,
        "test_probabilities": result.test_post,
        "test_predictions": result.test_pred,
        "metrics": {
            "oof": result.oof_metrics,
            "test": result.test_metrics,
        },
        "fitted_fts": models.get("fts"),
        "fitted_anfis_models": models.get("anfis_list"),
        "fitted_fusion_model": models.get("bn"),
    }

    return {
        "artifact": artifact,
        "cfg": cfg,
        "config": cfg,
        "dataset": dataset,
        "oof_post": result.oof_post,
        "oof_y": dataset.y_train.copy(),
        "oof_pred": result.oof_pred,
        "fold_indices": result.fold_indices,
        "best_thr": result.best_threshold,
        "thr_score": result.threshold_score,
        "thresholds": thresholds,
        "scores": scores,
        "oof_metrics": result.oof_metrics,
        "test_pred": result.test_pred,
        "test_post": result.test_post,
        "test_metrics": result.test_metrics,
        "models": models,
        "subject_df": pd.DataFrame(subject_rows),
        "subject_rows": subject_rows,
        "exp_name": selection.run_name,
        "experiment_name": selection.run_name,
        "variant": selection.variant_key,
        "artifact_results": result.artifact_results,
        "selection": selection,
        "config_summary_rows": config_summary_rows(cfg, result.best_threshold),
        "cache_key": make_cache_key(project_root, data_path, selection, cfg),
    }


def make_cache_key(
    project_root: Path,
    data_path: str,
    selection: ExperimentSelection,
    cfg: ExperimentConfig,
) -> tuple[Any, ...]:
    return (
        A2_SOFT_PRESET,
        selection.group_key,
        selection.variant_key,
        str((project_root / data_path).resolve()),
        cfg.mf_type,
        cfg.n_rules,
        "soft" if cfg.bn_soft else "none" if cfg.ablation == Ablation.NO_BN.value else "hard",
        cfg.n_bins,
        cfg.alpha,
        cfg.seed,
        cfg.k_folds,
    )
