from __future__ import annotations

import os
from dataclasses import asdict, dataclass


@dataclass(slots=True)
class ExperimentConfig:
    seed: int = 42
    k_folds: int = 5
    data_path: str = "last_30s_segments"

    n_sets: int = 3
    mf_type: str = "triangular"
    mf_sigma_scale: float = 0.5
    gbell_b: float = 2.0
    fts_norm: str = "total_sum"

    n_rules: int = 8
    sigma_scale: float = 1.5
    ridge_lambda: float = 0.1
    n_epochs: int = 15
    lr: float = 0.001

    n_bins: int = 3
    alpha: float = 1.0

    rank_metric: str = "auc"
    threshold_metric: str = "balanced_accuracy"
    threshold_grid: int = 181
    composite_score: bool = False

    scale_fts: bool = False
    robust_fts: bool = False
    bn_soft: bool = False
    bn_reliability: bool = False
    calibrate: str = "none"

    dump_rules: bool = False
    save_csv: str = "results_main.csv"

    # ------------------------------------------------------------------
    # Ablation experiment controls
    # ------------------------------------------------------------------
    # Which pipeline variant to run:
    #   exp1 = Stage1 (FTS) + Stage2 (ANFIS) + Stage3 (BN)  — full pipeline
    #   exp2 = Stage1 (FTS) + Stage2 (ANFIS)                — no BN, fuse by mean
    #   exp3 = Stage2 (ANFIS) + Stage3 (BN)                 — no FTS, use stat feats
    #   exp4 = Stage1 (FTS) + Stage3 (BN)                   — no ANFIS, FTS summary
    ablation: str = "exp1"

    # exp2 only: how to fuse 4 per-signal probs when BN is absent
    #   "mean"          → simple average of 4 ANFIS outputs
    #   "weighted_mean" → AUC-weighted average (weights from train fold)
    fusion_mode_when_no_bn: str = "mean"

    # exp3 only: feature type replacing Stage 1
    #   "stats" → 8 handcrafted statistical features per signal (currently only option)
    no_stage1_feature_mode: str = "stats"

    # exp4 only: how to summarise FTS features into one score per signal
    #   "diag"      → sum of self-transition probabilities
    #   "high"      → transitions involving the highest fuzzy state
    #   "diag_high" → normalised average of diag + high (default)
    #   "entropy"   → 1 − normalised Shannon entropy of transition dist.
    fts_summary_mode: str = "diag_high"

    def to_dict(self) -> dict:
        return asdict(self)

    def merged(self, **overrides) -> "ExperimentConfig":
        data = self.to_dict()
        data.update(overrides)
        return ExperimentConfig(**data)


def setup_seed(seed: int) -> None:
    import numpy as np

    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


EXPERIMENT_PRESETS: dict[str, dict] = {
    "A1": {
        "n_sets": 3, "mf_type": "triangular", "mf_sigma_scale": 0.5,
        "gbell_b": 2.0, "fts_norm": "total_sum",
        "n_rules": 8, "sigma_scale": 1.5, "ridge_lambda": 0.1,
        "n_epochs": 15, "lr": 0.001,
        "n_bins": 3, "alpha": 1.0, "k_folds": 5,
    },
    "A2": {
        "n_sets": 3, "mf_type": "gaussian", "mf_sigma_scale": 0.4,
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 6, "sigma_scale": 1.3, "ridge_lambda": 0.01,
        "n_epochs": 50, "lr": 0.003,
        "n_bins": 5, "alpha": 0.5, "k_folds": 5,
    },
    "A3": {
        "n_sets": 5, "mf_type": "trapezoid", "mf_sigma_scale": 0.5,
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 12, "sigma_scale": 1.0, "ridge_lambda": 0.0,
        "n_epochs": 50, "lr": 0.001,
        "n_bins": 5, "alpha": 0.5, "k_folds": 5,
    },
    "A4": {
        "n_sets": 3, "mf_type": "gbell", "mf_sigma_scale": 0.8,
        "gbell_b": 3.0, "fts_norm": "total_sum",
        "n_rules": 6, "sigma_scale": 0.8, "ridge_lambda": 0.1,
        "n_epochs": 10, "lr": 0.005,
        "n_bins": 3, "alpha": 2.0, "k_folds": 5,
    },
    "A5": {
        "n_sets": 4, "mf_type": "gaussian", "mf_sigma_scale": 0.4,
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 8, "sigma_scale": 1.5, "ridge_lambda": 0.1,
        "n_epochs": 20, "lr": 0.001,
        "n_bins": 5, "alpha": 1.0, "k_folds": 5,
    },
    "A2_plus": {
        "n_sets": 3, "mf_type": "gaussian", "mf_sigma_scale": 0.4,
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 6, "sigma_scale": 1.3, "ridge_lambda": 0.01,
        "n_epochs": 50, "lr": 0.003,
        "n_bins": 5, "alpha": 0.5, "k_folds": 5,
        "scale_fts": True, "robust_fts": True,
        "bn_soft": True, "bn_reliability": True,
        "calibrate": "isotonic",
    },
    "A2_soft": {
        "n_sets": 3, "mf_type": "gaussian", "mf_sigma_scale": 0.4,
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 6, "sigma_scale": 1.3, "ridge_lambda": 0.01,
        "n_epochs": 50, "lr": 0.003,
        "n_bins": 5, "alpha": 0.5, "k_folds": 5,
        "bn_soft": True,
    },
    "A2_cal": {
        "n_sets": 3, "mf_type": "gaussian", "mf_sigma_scale": 0.4,
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 6, "sigma_scale": 1.3, "ridge_lambda": 0.01,
        "n_epochs": 50, "lr": 0.003,
        "n_bins": 5, "alpha": 0.5, "k_folds": 5,
        "calibrate": "isotonic",
    },
    "A2_new": {
        "n_sets": 3, "mf_type": "gaussian", "mf_sigma_scale": 0.6,
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 6, "sigma_scale": 1.3, "ridge_lambda": 0.01,
        "n_epochs": 20, "lr": 0.003,
        "n_bins": 3, "alpha": 2.0, "k_folds": 5,
        "bn_soft": True,
    },
}


def apply_preset(config: ExperimentConfig, preset_name: str | None) -> ExperimentConfig:
    if not preset_name:
        return config
    preset = EXPERIMENT_PRESETS.get(preset_name)
    if preset is None:
        raise ValueError(
            f"Unknown preset: {preset_name}. Available: {list(EXPERIMENT_PRESETS)}"
        )
    return config.merged(**preset)
