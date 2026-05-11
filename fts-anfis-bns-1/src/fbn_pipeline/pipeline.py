from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig
from .metrics import find_best_threshold
from .models.anfis_tsk import ANFIS_TSK
from .models.bn_fusion import BNFusion
from .models.fts_encoder import FTSEncoder


def compute_oof_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    try:
        auc = round(float(roc_auc_score(y_true, y_prob)), 4)
    except ValueError:
        auc = 0.0
    return {
        "acc": round(float(accuracy_score(y_true, y_pred)), 4),
        "bal_acc": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "auc": auc,
    }


def _calibrate_probs(
    train_probs: np.ndarray,
    valid_probs: np.ndarray,
    y_train: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
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


def _compute_signal_weights(y_train: np.ndarray, train_probs: np.ndarray) -> np.ndarray:
    weights = np.ones(train_probs.shape[1])
    for s in range(train_probs.shape[1]):
        try:
            auc_s = roc_auc_score(y_train, train_probs[:, s])
            auc_s = max(auc_s, 1 - auc_s)
            weights[s] = max(2 * (auc_s - 0.5), 0.01)
        except ValueError:
            weights[s] = 0.5
    return weights


def _fit_signal_models(
    F_train: np.ndarray,
    F_valid: np.ndarray,
    y_train: np.ndarray,
    cfg: ExperimentConfig,
    *,
    seed_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[ANFIS_TSK]]:
    n_signals = 4
    feat_per_sig = F_train.shape[1] // n_signals
    train_probs = np.zeros((len(y_train), n_signals))
    valid_probs = np.zeros((len(F_valid), n_signals))
    anfis_models: list[ANFIS_TSK] = []

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
        )
        model.fit(F_train[:, col_s:col_e], y_train)
        train_probs[:, s] = model.predict_proba(F_train[:, col_s:col_e])
        valid_probs[:, s] = model.predict_proba(F_valid[:, col_s:col_e])
        anfis_models.append(model)

    return train_probs, valid_probs, anfis_models


def run_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    user_train: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    gkf = GroupKFold(n_splits=cfg.k_folds)
    oof_post = np.full(len(y_train), np.nan)
    fold_indices: list[tuple[np.ndarray, np.ndarray]] = []

    for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, user_train)):
        fold_indices.append((tr_idx, va_idx))
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr = y_train[tr_idx]

        fts = FTSEncoder(
            n_sets=cfg.n_sets,
            mf_type=cfg.mf_type,
            mf_sigma_scale=cfg.mf_sigma_scale,
            gbell_b=cfg.gbell_b,
            norm_mode=cfg.fts_norm,
            robust=cfg.robust_fts,
        )
        F_tr = fts.fit_transform(X_tr)
        F_va = fts.transform(X_va)

        if cfg.scale_fts:
            scaler = StandardScaler()
            F_tr = scaler.fit_transform(F_tr)
            F_va = scaler.transform(F_va)

        tr_probs, va_probs, _ = _fit_signal_models(F_tr, F_va, y_tr, cfg, seed_offset=fold_i * 100)
        tr_probs, va_probs = _calibrate_probs(tr_probs, va_probs, y_tr, cfg.calibrate)

        signal_weights = _compute_signal_weights(y_tr, tr_probs) if cfg.bn_reliability else None

        bn = BNFusion(n_bins=cfg.n_bins, alpha=cfg.alpha, soft=cfg.bn_soft)
        bn.fit(tr_probs, y_tr, signal_weights=signal_weights)
        oof_post[va_idx] = bn.predict_proba(va_probs)

    return oof_post, y_train.copy(), fold_indices


def retrain_and_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict]:
    fts = FTSEncoder(
        n_sets=cfg.n_sets,
        mf_type=cfg.mf_type,
        mf_sigma_scale=cfg.mf_sigma_scale,
        gbell_b=cfg.gbell_b,
        norm_mode=cfg.fts_norm,
        robust=cfg.robust_fts,
    )
    F_tr = fts.fit_transform(X_train)
    F_te = fts.transform(X_test)

    if cfg.scale_fts:
        scaler = StandardScaler()
        F_tr = scaler.fit_transform(F_tr)
        F_te = scaler.transform(F_te)

    tr_probs, te_probs, anfis_models = _fit_signal_models(F_tr, F_te, y_train, cfg)
    tr_probs, te_probs = _calibrate_probs(tr_probs, te_probs, y_train, cfg.calibrate)

    signal_weights = _compute_signal_weights(y_train, tr_probs) if cfg.bn_reliability else None

    bn = BNFusion(n_bins=cfg.n_bins, alpha=cfg.alpha, soft=cfg.bn_soft)
    bn.fit(tr_probs, y_train, signal_weights=signal_weights)

    te_post = bn.predict_proba(te_probs)
    y_pred = (te_post >= threshold).astype(int)
    metrics = compute_oof_metrics(y_test, te_post, threshold)

    models = {
        "fts": fts,
        "bn": bn,
        "anfis_list": anfis_models,
        "te_probs": te_probs,
    }
    return y_pred, te_post, metrics, models


def inner_cv_score(
    X: np.ndarray,
    y: np.ndarray,
    users: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[float, float, dict[str, float]]:
    oof_post, oof_y, _ = run_cv(X, y, users, cfg)
    best_thr, _ = find_best_threshold(
        oof_y,
        oof_post,
        metric=cfg.threshold_metric,
        n_grid=cfg.threshold_grid,
    )
    metrics = compute_oof_metrics(oof_y, oof_post, best_thr)

    if cfg.composite_score:
        rank_score = round(0.5 * metrics["auc"] + 0.5 * metrics["bal_acc"], 4)
    elif cfg.rank_metric == "balanced_accuracy":
        rank_score = metrics["bal_acc"]
    elif cfg.rank_metric == "macro_f1":
        rank_score = metrics["macro_f1"]
    else:
        rank_score = metrics["auc"]

    return rank_score, best_thr, metrics
