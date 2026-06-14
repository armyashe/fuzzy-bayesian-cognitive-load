from __future__ import annotations

import os
from dataclasses import asdict, dataclass

from .ablation import Ablation


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
    trap_plateau_scale: float = 0.15
    trap_width_scale: float = 1.0
    grid_margin_scale: float = 0.05

    # MF type for ANFIS premise layer (Stage 2).
    # This is intentionally separate from mf_type, which belongs to the
    # FTS encoder and may be triangular / gaussian / trapezoid / gbell.
    # ANFIS_TSK currently supports only gaussian or gbell premise MFs.
    anfis_mf_type: str = "gaussian"
    anfis_gbell_b: float = 2.0

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
    final_seed_offset: int = 0

    dump_rules: bool = False
    save_csv: str = "artifacts/A2_soft_results/results_main.csv"

    # ------------------------------------------------------------------
    # Bayesian Network structure variant
    # ------------------------------------------------------------------
    # Controls which BN fusion class is used in Stage 3:
    #   "naive"     -> BNFusion  (Naive Bayes, default)
    #   "dependent" -> TANFusion (Tree-Augmented NB, experimental)
    #   "tan"       -> TANFusion (alias for "dependent")
    bn_structure: str = "naive"

    # ------------------------------------------------------------------
    # Ablation experiment controls
    # ------------------------------------------------------------------
    # Which pipeline variant to run:
    #   exp1 = Stage1 (FTS) + Stage2 (ANFIS) + Stage3 (BN)  - full pipeline
    #   exp2 = Stage1 (FTS) + Stage2 (ANFIS)                - no BN, fuse by mean
    #   exp3 = Stage1 (FTS) + Stage3 (BN)                   - no ANFIS, FTS summary -> BN
    #   exp4 = Stage2 (ANFIS) + Stage3 (BN)                 - no FTS, stat features -> ANFIS -> BN
    ablation: str = Ablation.FULL.value

    # exp2 only: how to fuse 4 per-signal probs when BN is absent
    #   "mean"          -> simple average of 4 ANFIS outputs
    #   "weighted_mean" -> AUC-weighted average (weights from train fold)
    fusion_mode_when_no_bn: str = "mean"

    # exp4 only: feature type replacing Stage 1
    #   "stats" -> 8 handcrafted statistical features per signal (currently only option)
    no_stage1_feature_mode: str = "stats"

    # exp3 only: how to summarise FTS features into one score per signal
    #   "diag"      -> sum of self-transition probabilities
    #   "high"      -> transitions involving the highest fuzzy state
    #   "diag_high" -> normalised average of diag + high (default)
    #   "entropy"   -> 1 - normalised Shannon entropy of transition dist.
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
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


THESIS_CONFIG_EXPECTED = {
    "bn_soft": True,
    "anfis_mf_type": "gaussian",
    "anfis_gbell_b": 2.0,
    "mf_type": "gaussian",
    "n_rules": 6,
    "n_bins": 5,
    "alpha": 0.5,
    "calibrate": "none",
    "bn_reliability": False,
    "scale_fts": False,
    "robust_fts": False,
}


def assert_thesis_config(cfg: ExperimentConfig) -> None:
    """Fail fast if the locked thesis run drifts from the published config."""
    mismatches: list[str] = []
    for key, expected in THESIS_CONFIG_EXPECTED.items():
        actual = getattr(cfg, key)
        if actual != expected:
            mismatches.append(f"{key}={actual!r} (expected {expected!r})")

    if mismatches:
        joined = "; ".join(mismatches)
        raise ValueError(f"THESIS_FINAL configuration mismatch: {joined}")


from .presets import EXPERIMENT_PRESETS, apply_preset  # noqa: E402
