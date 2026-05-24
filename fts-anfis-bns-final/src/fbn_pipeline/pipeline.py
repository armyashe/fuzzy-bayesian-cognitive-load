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
from .features import extract_stat_features, summarise_fts_per_signal
from .metrics import find_best_threshold
from .models.anfis_tsk import ANFIS_TSK
from .models.bn_fusion import BNFusion
from .models.bn_dependent import TANFusion
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
    """
    Train one ANFIS per signal and collect per-signal posterior probabilities.

    Parameters
    ----------
    F_train : (N_tr, n_signals * n_feat_per_signal)
    F_valid : (N_va, n_signals * n_feat_per_signal)
    y_train : (N_tr,)  binary labels
    cfg     : experiment config

    Returns
    -------
    train_probs : (N_tr, n_signals)  — P(task | signal_s) on train set
    valid_probs : (N_va, n_signals)  — P(task | signal_s) on valid set
    anfis_list  : list of fitted ANFIS_TSK, one per signal
    """
    n_signals = 4
    feat_per_sig = F_train.shape[1] // n_signals
    train_probs = np.zeros((len(y_train), n_signals))
    valid_probs = np.zeros((len(F_valid), n_signals))
    anfis_models: list[ANFIS_TSK] = []

    for s in range(n_signals):
        col_s = s * feat_per_sig
        col_e = col_s + feat_per_sig
        # anfis_mf_type overrides mf_type for ANFIS premise layer only.
        # Empty string means inherit the same MF as the FTS encoder.
        effective_anfis_mf = cfg.anfis_mf_type or cfg.mf_type
        model = ANFIS_TSK(
            n_rules=cfg.n_rules,
            sigma_scale=cfg.sigma_scale,
            ridge_lambda=cfg.ridge_lambda,
            n_epochs=cfg.n_epochs,
            lr=cfg.lr,
            random_state=cfg.seed + seed_offset + s,
            mf_type=effective_anfis_mf,
            gbell_b=cfg.gbell_b,
        )
        model.fit(F_train[:, col_s:col_e], y_train)
        train_probs[:, s] = model.predict_proba(F_train[:, col_s:col_e])
        valid_probs[:, s] = model.predict_proba(F_valid[:, col_s:col_e])
        anfis_models.append(model)

    return train_probs, valid_probs, anfis_models


# ------------------------------------------------------------------
# Stage 1 dispatch: FTS encoder vs. statistical features
# ------------------------------------------------------------------

def _extract_features_for_ablation(
    X_tr: np.ndarray,
    X_va: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, FTSEncoder | None]:
    """
    Dispatch Stage 1 feature extraction based on cfg.ablation.

    - exp1, exp2, exp3 → FTSEncoder (transition matrices)
    - exp4             → handcrafted statistical features (no FTSEncoder)

    Parameters
    ----------
    X_tr : (N_tr, 4, T)  raw train windows
    X_va : (N_va, 4, T)  raw validation / test windows
    cfg  : ExperimentConfig

    Returns
    -------
    F_tr : (N_tr, n_features)
    F_va : (N_va, n_features)
    fts  : FTSEncoder or None  (None for exp4; kept for rule dumping otherwise)
    """
    if cfg.ablation == "exp4":
        # Stage 1 bypass: use handcrafted statistical features
        F_tr = extract_stat_features(X_tr)
        F_va = extract_stat_features(X_va)
        fts = None
    else:
        # Standard FTSEncoder (exp1, exp2, exp3)
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

    return F_tr, F_va, fts


# ------------------------------------------------------------------
# exp2: BN-free fusion
# ------------------------------------------------------------------

def _make_bn(cfg: ExperimentConfig) -> BNFusion | TANFusion:
    """
    Construct the correct BN fusion object based on cfg.bn_structure.

    "naive"            -> BNFusion  (Naive Bayes, default)
    "dependent" | "tan"-> TANFusion (Tree-Augmented Naive Bayes, experimental)
    """
    bn_structure = getattr(cfg, "bn_structure", "naive")
    if bn_structure in ("dependent", "tan"):
        return TANFusion(n_bins=cfg.n_bins, alpha=cfg.alpha, soft=cfg.bn_soft)
    if bn_structure == "naive":
        return BNFusion(n_bins=cfg.n_bins, alpha=cfg.alpha, soft=cfg.bn_soft)
    raise ValueError(
        f"Unknown bn_structure: {bn_structure!r}. "
        "Expected one of: 'naive', 'dependent', 'tan'."
    )


def _fuse_without_bn(
    y_train: np.ndarray,
    tr_probs: np.ndarray,
    va_probs: np.ndarray,
    cfg: ExperimentConfig,
) -> np.ndarray:
    """
    Fuse 4 per-signal ANFIS probabilities without a Bayesian Network.

    Used in **exp2** where BNFusion is bypassed.

    Parameters
    ----------
    y_train  : (N_tr,)   — training labels, used to compute AUC weights
    tr_probs : (N_tr, 4) — ANFIS outputs on training fold
    va_probs : (N_va, 4) — ANFIS outputs on validation / test fold
    cfg      : ExperimentConfig (reads fusion_mode_when_no_bn)

    Returns
    -------
    posterior : (N_va,)  — fused probability estimate
    """
    if cfg.fusion_mode_when_no_bn == "weighted_mean":
        weights = _compute_signal_weights(y_train, tr_probs)
        weights = weights / weights.sum()          # normalise to sum=1
        return (va_probs * weights).sum(axis=1)
    else:  # "mean"
        return va_probs.mean(axis=1)


# ------------------------------------------------------------------
# Cross-validation
# ------------------------------------------------------------------

def run_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    user_train: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """
    GroupKFold cross-validation dispatcher.

    Dispatches the inner fold logic based on cfg.ablation:
      exp1 : FTS → ANFIS → BN
      exp2 : FTS → ANFIS → mean / weighted_mean fusion
      exp3 : stat-features → ANFIS → BN
      exp4 : FTS → per-signal summary → BN

    Threshold selection happens **outside** this function on the returned
    OOF posteriors to avoid any data leakage.

    Parameters
    ----------
    X_train    : (N, 4, T)
    y_train    : (N,)
    user_train : (N,)  subject IDs for GroupKFold

    Returns
    -------
    oof_post     : (N,)   out-of-fold posterior probabilities
    oof_y        : (N,)   copy of y_train
    fold_indices : list of (tr_idx, va_idx) arrays
    """
    gkf = GroupKFold(n_splits=cfg.k_folds)
    oof_post = np.full(len(y_train), np.nan)
    fold_indices: list[tuple[np.ndarray, np.ndarray]] = []

    header = f"  {'Fold':>4} | {'Split':>5} | {'Total':>5} | {'cls 0':>5} | {'cls 1':>5} | Users"
    print("-" * 80)
    print(header)
    print("-" * 80)

    for fold_i, (tr_idx, va_idx) in enumerate(
        gkf.split(X_train, y_train, user_train)
    ):
        fold_indices.append((tr_idx, va_idx))
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr = y_train[tr_idx]
        y_va = y_train[va_idx]

        users_tr = sorted(np.unique(user_train[tr_idx]).tolist())
        users_va = sorted(np.unique(user_train[va_idx]).tolist())
        n_tr_0, n_tr_1 = int(np.sum(y_tr == 0)), int(np.sum(y_tr == 1))
        n_va_0, n_va_1 = int(np.sum(y_va == 0)), int(np.sum(y_va == 1))

        print(f"  {fold_i + 1:>4} | {'Train':>5} | {len(tr_idx):>5} | {n_tr_0:>5} | {n_tr_1:>5} | {users_tr}")
        print(f"  {' ':>4} | {'Val':>5} | {len(va_idx):>5} | {n_va_0:>5} | {n_va_1:>5} | {users_va}")
        print("-" * 80)

        # --- Stage 1 (or bypass) ---
        F_tr, F_va, _fts = _extract_features_for_ablation(X_tr, X_va, cfg)

        if cfg.ablation == "exp3":
            # Stage 1 → continuous summary → Stage 3 (ANFIS bypassed)
            tr_scores = summarise_fts_per_signal(F_tr, cfg.n_sets, cfg.fts_summary_mode)
            va_scores = summarise_fts_per_signal(F_va, cfg.n_sets, cfg.fts_summary_mode)

            sw = _compute_signal_weights(y_tr, tr_scores) if cfg.bn_reliability else None

            bn = _make_bn(cfg)
            bn.fit(tr_scores, y_tr, signal_weights=sw)

            oof_post[va_idx] = bn.predict_proba(va_scores)

        else:
            # Stage 2: ANFIS per signal (exp1, exp2, exp4)
            tr_probs, va_probs, _ = _fit_signal_models(
                F_tr, F_va, y_tr, cfg, seed_offset=fold_i * 100
            )
            tr_probs, va_probs = _calibrate_probs(tr_probs, va_probs, y_tr, cfg.calibrate)

            if cfg.ablation == "exp2":
                # No BN: fuse probabilities directly
                oof_post[va_idx] = _fuse_without_bn(y_tr, tr_probs, va_probs, cfg)
            else:
                # Stage 3: BN fusion (exp1, exp4)
                sw = _compute_signal_weights(y_tr, tr_probs) if cfg.bn_reliability else None
                bn = _make_bn(cfg)
                bn.fit(tr_probs, y_tr, signal_weights=sw)
                oof_post[va_idx] = bn.predict_proba(va_probs)

    return oof_post, y_train.copy(), fold_indices


# ------------------------------------------------------------------
# Final retrain + test evaluation
# ------------------------------------------------------------------

def retrain_and_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict]:
    """
    Retrain on the full training set and evaluate on the held-out test set.

    Dispatches based on cfg.ablation (same logic as run_cv).

    Returns
    -------
    y_pred   : (N_te,)  binary predictions
    te_post  : (N_te,)  posterior probabilities
    metrics  : dict with acc, bal_acc, macro_f1, f1, auc
    models   : dict with keys:
                 "fts"        → FTSEncoder | None
                 "bn"         → BNFusion   | None
                 "anfis_list" → list[ANFIS_TSK] | None
                 "te_probs"   → (N_te, 4) per-signal evidence fed to BN/fusion
    """
    F_tr, F_te, fts = _extract_features_for_ablation(X_train, X_test, cfg)

    if cfg.ablation == "exp3":
        tr_scores = summarise_fts_per_signal(F_tr, cfg.n_sets, cfg.fts_summary_mode)
        te_scores = summarise_fts_per_signal(F_te, cfg.n_sets, cfg.fts_summary_mode)

        sw = _compute_signal_weights(y_train, tr_scores) if cfg.bn_reliability else None

        bn = _make_bn(cfg)
        bn.fit(tr_scores, y_train, signal_weights=sw)

        te_post = bn.predict_proba(te_scores)
        y_pred = (te_post >= threshold).astype(int)
        metrics = compute_oof_metrics(y_test, te_post, threshold)

        models = {
            "fts": fts,
            "bn": bn,
            "anfis_list": None,
            "te_probs": te_scores,   # continuous FTS summary scores (N_te, 4)
        }

    else:
        tr_probs, te_probs, anfis_models = _fit_signal_models(F_tr, F_te, y_train, cfg)
        tr_probs, te_probs = _calibrate_probs(tr_probs, te_probs, y_train, cfg.calibrate)

        if cfg.ablation == "exp2":
            te_post = _fuse_without_bn(y_train, tr_probs, te_probs, cfg)
            bn = None
        else:
            # exp1, exp4
            sw = _compute_signal_weights(y_train, tr_probs) if cfg.bn_reliability else None
            bn = _make_bn(cfg)
            bn.fit(tr_probs, y_train, signal_weights=sw)
            te_post = bn.predict_proba(te_probs)

        y_pred = (te_post >= threshold).astype(int)
        metrics = compute_oof_metrics(y_test, te_post, threshold)
        models = {
            "fts": fts,
            "bn": bn,
            "anfis_list": anfis_models,
            "te_probs": te_probs,    # ANFIS probability outputs (N_te, 4)
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
