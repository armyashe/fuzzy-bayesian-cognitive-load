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

    # MF type for ANFIS premise layer (Stage 2).
    # This is intentionally separate from mf_type, which belongs to the
    # FTS encoder and may be triangular / gaussian / trapezoid / gbell.
    # ANFIS_TSK currently supports only gaussian or gbell premise MFs.
    anfis_mf_type: str = "gaussian"

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
    # Bayesian Network structure variant
    # ------------------------------------------------------------------
    # Controls which BN fusion class is used in Stage 3:
    #   "naive"     → BNFusion  (Naive Bayes, default)
    #   "dependent" → TANFusion (Tree-Augmented NB, experimental)
    #   "tan"       → TANFusion (alias for "dependent")
    bn_structure: str = "naive"

    # ------------------------------------------------------------------
    # Ablation experiment controls
    # ------------------------------------------------------------------
    # Which pipeline variant to run:
    #   exp1 = Stage1 (FTS) + Stage2 (ANFIS) + Stage3 (BN)  — full pipeline
    #   exp2 = Stage1 (FTS) + Stage2 (ANFIS)                — no BN, fuse by mean
    #   exp3 = Stage1 (FTS) + Stage3 (BN)                   — no ANFIS, FTS summary → BN
    #   exp4 = Stage2 (ANFIS) + Stage3 (BN)                 — no FTS, stat features → ANFIS → BN
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

    # ------------------------------------------------------------------
    # Experiment identification (used by run_experiments.py)
    # ------------------------------------------------------------------
    # High-level group label: "EXP1", "EXP2", "EXP3", or "" for ad-hoc runs.
    experiment_group: str = ""
    # Short human-readable name identifying the specific variant being run.
    variant_name: str = ""

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
        "anfis_mf_type": "gaussian",
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
        "anfis_mf_type": "gaussian",
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 12, "sigma_scale": 1.0, "ridge_lambda": 0.0,
        "n_epochs": 50, "lr": 0.001,
        "n_bins": 5, "alpha": 0.5, "k_folds": 5,
    },
    "A4": {
        "n_sets": 3, "mf_type": "gbell", "mf_sigma_scale": 0.8,
        "anfis_mf_type": "gbell",
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
    "A2_gbell": {
        # A2_soft với gbell cho cả FTS encoder lẫn ANFIS premise MFs
        "n_sets": 3, "mf_type": "gbell", "mf_sigma_scale": 0.4,
        "anfis_mf_type": "gbell",
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 6, "sigma_scale": 1.3, "ridge_lambda": 0.01,
        "n_epochs": 50, "lr": 0.003,
        "n_bins": 5, "alpha": 0.5, "k_folds": 5,
        "bn_soft": True,
    },
    "A2_soft_anfis_gbell": {
        # FTS encoder dùng Gaussian, ANFIS premise dùng Gbell
        "n_sets": 3, "mf_type": "gaussian", "mf_sigma_scale": 0.4,
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 6, "sigma_scale": 1.3, "ridge_lambda": 0.01,
        "n_epochs": 50, "lr": 0.003,
        "n_bins": 5, "alpha": 0.5, "k_folds": 5,
        "bn_soft": True,
        "anfis_mf_type": "gbell",
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
     "A2_soft_bin3": {
        "n_sets": 3, "mf_type": "gaussian", "mf_sigma_scale": 0.4,
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 6, "sigma_scale": 1.3, "ridge_lambda": 0.01,
        "n_epochs": 50, "lr": 0.003,
        "n_bins": 3, "alpha": 0.5, "k_folds": 5,
        "bn_soft": True,
    },
      "A2_soft_nset4_bin3": {
        "n_sets": 4, "mf_type": "gaussian", "mf_sigma_scale": 0.4,
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 6, "sigma_scale": 1.3, "ridge_lambda": 0.01,
        "n_epochs": 50, "lr": 0.003,
        "n_bins": 3, "alpha": 0.5, "k_folds": 5,
        "bn_soft": True,
    },
    # ------------------------------------------------------------------
    # TAN (Tree-Augmented Naive Bayes) experiment
    # Identical hyperparameters to A2_soft but with TAN fusion structure.
    # Designed to compare against A2_soft (naive BN) in the same protocol.
    # ------------------------------------------------------------------
    "A2_soft_tan": {
        "n_sets": 3, "mf_type": "gaussian", "mf_sigma_scale": 0.4,
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 6, "sigma_scale": 1.3, "ridge_lambda": 0.01,
        "n_epochs": 50, "lr": 0.003,
        "n_bins": 5, "alpha": 0.5, "k_folds": 5,
        "bn_soft": True,    # Gaussian soft evidence over bins (same as A2_soft)
        "bn_structure": "tan",
    },
    # Hard-evidence TAN for ablation: compare soft vs hard within TAN
    "A2_hard_tan": {
        "n_sets": 3, "mf_type": "gaussian", "mf_sigma_scale": 0.4,
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 6, "sigma_scale": 1.3, "ridge_lambda": 0.01,
        "n_epochs": 50, "lr": 0.003,
        "n_bins": 5, "alpha": 0.5, "k_folds": 5,
        "bn_soft": False,
        "bn_structure": "tan",
    },

    # ==================================================================
    # EXP1 — Effect of FTS Membership Function Type
    # All variants use the full FTS → ANFIS → Naive BN (soft) pipeline.
    # Only mf_type changes between variants.
    # Base hyperparameters: A2_soft configuration.
    # ==================================================================
    "EXP1_MF_TRI": {
        "n_sets": 3, "mf_type": "triangular", "mf_sigma_scale": 0.4,
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 6, "sigma_scale": 1.3, "ridge_lambda": 0.01,
        "n_epochs": 50, "lr": 0.003,
        "n_bins": 5, "alpha": 0.5, "k_folds": 5,
        "bn_structure": "naive", "bn_soft": True,
        "ablation": "exp1",
        "anfis_mf_type": "gaussian",  # ANFIS premise fixed; only FTS encoder MF varies
        "experiment_group": "EXP1", "variant_name": "EXP1_MF_TRI",
    },
    "EXP1_MF_GAUSS": {
        "n_sets": 3, "mf_type": "gaussian", "mf_sigma_scale": 0.4,
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 6, "sigma_scale": 1.3, "ridge_lambda": 0.01,
        "n_epochs": 50, "lr": 0.003,
        "n_bins": 5, "alpha": 0.5, "k_folds": 5,
        "bn_structure": "naive", "bn_soft": True,
        "ablation": "exp1",
        "anfis_mf_type": "gaussian",  # ANFIS premise fixed; only FTS encoder MF varies
        "experiment_group": "EXP1", "variant_name": "EXP1_MF_GAUSS",
    },
    "EXP1_MF_TRAP": {
        "n_sets": 3, "mf_type": "trapezoid", "mf_sigma_scale": 0.4,
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 6, "sigma_scale": 1.3, "ridge_lambda": 0.01,
        "n_epochs": 50, "lr": 0.003,
        "n_bins": 5, "alpha": 0.5, "k_folds": 5,
        "bn_structure": "naive", "bn_soft": True,
        "ablation": "exp1",
        "anfis_mf_type": "gaussian",  # ANFIS premise fixed; only FTS encoder MF varies
        "experiment_group": "EXP1", "variant_name": "EXP1_MF_TRAP",
    },
    "EXP1_MF_GBELL": {
        "n_sets": 3, "mf_type": "gbell", "mf_sigma_scale": 0.4,
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 6, "sigma_scale": 1.3, "ridge_lambda": 0.01,
        "n_epochs": 50, "lr": 0.003,
        "n_bins": 5, "alpha": 0.5, "k_folds": 5,
        "bn_structure": "naive", "bn_soft": True,
        "ablation": "exp1",
        "anfis_mf_type": "gaussian",  # ANFIS premise fixed; only FTS encoder MF varies
        "experiment_group": "EXP1", "variant_name": "EXP1_MF_GBELL",
    },

    # ==================================================================
    # EXP2 — Effect of Number of ANFIS Rules
    # All variants use gaussian MF and the full FTS → ANFIS → Naive BN pipeline.
    # Only n_rules changes between variants.
    # ==================================================================
    "EXP2_RULES_3": {
        "n_sets": 3, "mf_type": "gaussian", "mf_sigma_scale": 0.4,
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 3, "sigma_scale": 1.3, "ridge_lambda": 0.01,
        "n_epochs": 50, "lr": 0.003,
        "n_bins": 5, "alpha": 0.5, "k_folds": 5,
        "bn_structure": "naive", "bn_soft": True,
        "ablation": "exp1",
        "experiment_group": "EXP2", "variant_name": "EXP2_RULES_3",
    },
    "EXP2_RULES_6": {
        "n_sets": 3, "mf_type": "gaussian", "mf_sigma_scale": 0.4,
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 6, "sigma_scale": 1.3, "ridge_lambda": 0.01,
        "n_epochs": 50, "lr": 0.003,
        "n_bins": 5, "alpha": 0.5, "k_folds": 5,
        "bn_structure": "naive", "bn_soft": True,
        "ablation": "exp1",
        "experiment_group": "EXP2", "variant_name": "EXP2_RULES_6",
    },
    "EXP2_RULES_8": {
        "n_sets": 3, "mf_type": "gaussian", "mf_sigma_scale": 0.4,
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 8, "sigma_scale": 1.3, "ridge_lambda": 0.01,
        "n_epochs": 50, "lr": 0.003,
        "n_bins": 5, "alpha": 0.5, "k_folds": 5,
        "bn_structure": "naive", "bn_soft": True,
        "ablation": "exp1",
        "experiment_group": "EXP2", "variant_name": "EXP2_RULES_8",
    },
    "EXP2_RULES_12": {
        "n_sets": 3, "mf_type": "gaussian", "mf_sigma_scale": 0.4,
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 12, "sigma_scale": 1.3, "ridge_lambda": 0.01,
        "n_epochs": 50, "lr": 0.003,
        "n_bins": 5, "alpha": 0.5, "k_folds": 5,
        "bn_structure": "naive", "bn_soft": True,
        "ablation": "exp1",
        "experiment_group": "EXP2", "variant_name": "EXP2_RULES_12",
    },

    # ==================================================================
    # EXP3 — Contribution of Pipeline Stages
    # EXP3_FULL     : FTS → ANFIS → BN       (proposed complete model)
    # EXP3_NO_BN    : FTS → ANFIS → Mean     (ANFIS only, no BN fusion)
    # EXP3_NO_ANFIS : FTS Summary → BN       (BN only, no per-signal ANFIS)
    # ==================================================================
    "EXP3_FULL": {
        "n_sets": 3, "mf_type": "gaussian", "mf_sigma_scale": 0.4,
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 6, "sigma_scale": 1.3, "ridge_lambda": 0.01,
        "n_epochs": 50, "lr": 0.003,
        "n_bins": 5, "alpha": 0.5, "k_folds": 5,
        "bn_structure": "naive", "bn_soft": True,
        "ablation": "exp1",
        "experiment_group": "EXP3", "variant_name": "EXP3_FULL",
    },
    "EXP3_NO_BN": {
        # FTS → ANFIS → Mean fusion (no BN CPT fitted).
        # fusion_mode_when_no_bn="mean" → p_final = (p_Temp+p_HR+p_GSR+p_RR)/4
        "n_sets": 3, "mf_type": "gaussian", "mf_sigma_scale": 0.4,
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 6, "sigma_scale": 1.3, "ridge_lambda": 0.01,
        "n_epochs": 50, "lr": 0.003,
        "n_bins": 5, "alpha": 0.5, "k_folds": 5,
        "bn_structure": "naive", "bn_soft": False,
        "ablation": "exp2",  # pipeline.py: no BN, fuse by mean
        "fusion_mode_when_no_bn": "mean",
        "experiment_group": "EXP3", "variant_name": "EXP3_NO_BN",
    },
    "EXP3_NO_ANFIS": {
        # FTS summary score per signal → BN (ANFIS bypassed).
        # fts_summary_mode="diag_high": normalised avg of diag + high-state transitions.
        # This is an interpretable heuristic baseline, not a learned model.
        "n_sets": 3, "mf_type": "gaussian", "mf_sigma_scale": 0.4,
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 6, "sigma_scale": 1.3, "ridge_lambda": 0.01,
        "n_epochs": 50, "lr": 0.003,
        "n_bins": 5, "alpha": 0.5, "k_folds": 5,
        "bn_structure": "naive", "bn_soft": True,
        "ablation": "exp3",  # pipeline.py: FTS summary → BN, ANFIS bypassed
        "fts_summary_mode": "diag_high",
        "experiment_group": "EXP3", "variant_name": "EXP3_NO_ANFIS",
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
