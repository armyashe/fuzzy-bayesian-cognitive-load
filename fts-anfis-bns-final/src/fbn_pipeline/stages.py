from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from .ablation import Ablation
from .config import ExperimentConfig
from .features import extract_stat_features
from .models.fts_encoder import FTSEncoder


def extract_features_for_ablation(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, FTSEncoder | None]:
    """Run Stage 1 or its ablation replacement for train/valid windows."""
    if cfg.ablation == Ablation.NO_FTS:
        F_train = extract_stat_features(X_train)
        F_valid = extract_stat_features(X_valid)
        fts = None
    else:
        fts = FTSEncoder(
            n_sets=cfg.n_sets,
            mf_type=cfg.mf_type,
            mf_sigma_scale=cfg.mf_sigma_scale,
            gbell_b=cfg.gbell_b,
            norm_mode=cfg.fts_norm,
            robust=cfg.robust_fts,
            trap_plateau_scale=cfg.trap_plateau_scale,
            trap_width_scale=cfg.trap_width_scale,
            grid_margin_scale=cfg.grid_margin_scale,
        )
        F_train = fts.fit_transform(X_train)
        F_valid = fts.transform(X_valid)

    if cfg.scale_fts:
        scaler = StandardScaler()
        F_train = scaler.fit_transform(F_train)
        F_valid = scaler.transform(F_valid)

    return F_train, F_valid, fts
