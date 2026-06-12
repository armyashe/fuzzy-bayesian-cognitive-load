from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from .config import ExperimentConfig
from .models.bn_dependent import TANFusion
from .models.bn_fusion import BNFusion
from .models.fusion_protocol import FusionModel


def compute_signal_weights(y_train: np.ndarray, train_probs: np.ndarray) -> np.ndarray:
    """Compute AUC-derived reliability weights for per-signal evidence."""
    weights = np.ones(train_probs.shape[1])
    for s in range(train_probs.shape[1]):
        try:
            auc_s = roc_auc_score(y_train, train_probs[:, s])
            auc_s = max(auc_s, 1 - auc_s)
            weights[s] = max(2 * (auc_s - 0.5), 0.01)
        except ValueError:
            weights[s] = 0.5
    return weights


def make_bn(cfg: ExperimentConfig) -> FusionModel:
    """Construct the configured Bayesian fusion model."""
    bn_structure = getattr(cfg, "bn_structure", "naive")
    if bn_structure in ("dependent", "tan"):
        return TANFusion(n_bins=cfg.n_bins, alpha=cfg.alpha, soft=cfg.bn_soft)
    if bn_structure == "naive":
        return BNFusion(n_bins=cfg.n_bins, alpha=cfg.alpha, soft=cfg.bn_soft)
    raise ValueError(
        f"Unknown bn_structure: {bn_structure!r}. "
        "Expected one of: 'naive', 'dependent', 'tan'."
    )


def fuse_without_bn(
    y_train: np.ndarray,
    train_probs: np.ndarray,
    valid_probs: np.ndarray,
    cfg: ExperimentConfig,
) -> np.ndarray:
    """Fuse four per-signal probabilities without fitting a BN."""
    if cfg.fusion_mode_when_no_bn == "weighted_mean":
        weights = compute_signal_weights(y_train, train_probs)
        weights = weights / weights.sum()
        return (valid_probs * weights).sum(axis=1)
    return valid_probs.mean(axis=1)
