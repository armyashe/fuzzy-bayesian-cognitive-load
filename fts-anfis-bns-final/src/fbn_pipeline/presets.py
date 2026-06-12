from __future__ import annotations


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
        # A2_soft with gbell for both FTS encoder and ANFIS premise MFs.
        "n_sets": 3, "mf_type": "gbell", "mf_sigma_scale": 0.4,
        "anfis_mf_type": "gbell",
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 6, "sigma_scale": 1.3, "ridge_lambda": 0.01,
        "n_epochs": 50, "lr": 0.003,
        "n_bins": 5, "alpha": 0.5, "k_folds": 5,
        "bn_soft": True,
    },
    "A2_soft_anfis_gbell": {
        # Gaussian FTS encoder, gbell ANFIS premise.
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
    # EXP1 - Effect of FTS Membership Function Type
    # All variants use the full FTS -> ANFIS -> Naive BN (soft) pipeline.
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
    # EXP2 - Effect of Number of ANFIS Rules
    # All variants use gaussian MF and the full FTS -> ANFIS -> Naive BN pipeline.
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
    # EXP3 - Contribution of Pipeline Stages
    # EXP3_FULL     : FTS -> ANFIS -> BN       (proposed complete model)
    # EXP3_NO_BN    : FTS -> ANFIS -> Mean     (ANFIS only, no BN fusion)
    # EXP3_NO_ANFIS : FTS Summary -> BN       (BN only, no per-signal ANFIS)
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
        # FTS -> ANFIS -> Mean fusion (no BN CPT fitted).
        # fusion_mode_when_no_bn="mean" -> p_final = (p_Temp+p_HR+p_GSR+p_RR)/4
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
        # FTS summary score per signal -> BN (ANFIS bypassed).
        # fts_summary_mode="diag_high": normalised avg of diag + high-state transitions.
        # This is an interpretable heuristic baseline, not a learned model.
        "n_sets": 3, "mf_type": "gaussian", "mf_sigma_scale": 0.4,
        "gbell_b": 2.0, "fts_norm": "row",
        "n_rules": 6, "sigma_scale": 1.3, "ridge_lambda": 0.01,
        "n_epochs": 50, "lr": 0.003,
        "n_bins": 5, "alpha": 0.5, "k_folds": 5,
        "bn_structure": "naive", "bn_soft": True,
        "ablation": "exp3",  # pipeline.py: FTS summary -> BN, ANFIS bypassed
        "fts_summary_mode": "diag_high",
        "experiment_group": "EXP3", "variant_name": "EXP3_NO_ANFIS",
    },
}


def apply_preset(config, preset_name: str | None):
    if not preset_name:
        return config
    preset = EXPERIMENT_PRESETS.get(preset_name)
    if preset is None:
        raise ValueError(
            f"Unknown preset: {preset_name}. Available: {list(EXPERIMENT_PRESETS)}"
        )
    return config.merged(**preset)
