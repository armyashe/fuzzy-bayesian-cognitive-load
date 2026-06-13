from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .config import ExperimentConfig
from .models.anfis_tsk import ANFIS_TSK


def calibrate_probs(
    train_probs: np.ndarray,
    valid_probs: np.ndarray,
    y_train: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate per-signal probability outputs using Platt or isotonic mode."""
    calibrated_train = train_probs.copy()
    calibrated_valid = valid_probs.copy()
    if mode == "none":
        return calibrated_train, calibrated_valid

    for s in range(train_probs.shape[1]):
        if mode == "platt":
            eps = 1e-7
            p_tr = np.clip(calibrated_train[:, s], eps, 1 - eps)
            p_va = np.clip(calibrated_valid[:, s], eps, 1 - eps)
            logit_tr = np.log(p_tr / (1 - p_tr))
            logit_va = np.log(p_va / (1 - p_va))
            cal = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            cal.fit(logit_tr.reshape(-1, 1), y_train)
            calibrated_train[:, s] = cal.predict_proba(logit_tr.reshape(-1, 1))[:, 1]
            calibrated_valid[:, s] = cal.predict_proba(logit_va.reshape(-1, 1))[:, 1]
        elif mode == "isotonic":
            cal = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            cal.fit(calibrated_train[:, s], y_train)
            calibrated_train[:, s] = cal.predict(calibrated_train[:, s])
            calibrated_valid[:, s] = cal.predict(calibrated_valid[:, s])

    return calibrated_train, calibrated_valid


def fit_signal_models(
    F_train: np.ndarray,
    F_valid: np.ndarray,
    y_train: np.ndarray,
    cfg: ExperimentConfig,
    *,
    seed_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[ANFIS_TSK]]:
    """Train one ANFIS per signal and collect per-signal posterior outputs."""
    n_signals = 4
    feat_per_sig = F_train.shape[1] // n_signals
    train_probs = np.zeros((len(y_train), n_signals))
    valid_probs = np.zeros((len(F_valid), n_signals))
    anfis_models: list[ANFIS_TSK] = []

    if cfg.anfis_mf_type not in {"gaussian", "gbell"}:
        raise ValueError(
            "ANFIS mf_type must be 'gaussian' or 'gbell', "
            f"got {cfg.anfis_mf_type!r}"
        )

    for s in range(n_signals):
        col_s = s * feat_per_sig
        col_e = col_s + feat_per_sig
        model = ANFIS_TSK(
            n_rules=cfg.n_rules,
            sigma_scale=cfg.sigma_scale,
            ridge_lambda=cfg.ridge_lambda,
            n_epochs=cfg.n_epochs,
            lr=cfg.lr,
            random_state=cfg.seed + seed_offset + s,
            mf_type=cfg.anfis_mf_type,
            gbell_b=cfg.anfis_gbell_b,
        )
        model.fit(F_train[:, col_s:col_e], y_train)
        train_probs[:, s] = model.predict_proba(F_train[:, col_s:col_e])
        valid_probs[:, s] = model.predict_proba(F_valid[:, col_s:col_e])
        anfis_models.append(model)

    return train_probs, valid_probs, anfis_models
