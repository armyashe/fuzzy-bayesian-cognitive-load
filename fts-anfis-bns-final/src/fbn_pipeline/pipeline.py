from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .ablation import Ablation
from .config import ExperimentConfig
from .features import summarise_fts_per_signal
from .fusion import compute_signal_weights, fuse_without_bn, make_bn
from .metrics import compute_oof_metrics
from .results import CvResult, TestResult, TrainedModels
from .stages import extract_features_for_ablation
from .training import calibrate_probs, fit_signal_models


def run_cv_result(
    X_train: np.ndarray,
    y_train: np.ndarray,
    user_train: np.ndarray,
    cfg: ExperimentConfig,
) -> CvResult:
    """Run GroupKFold CV and return typed out-of-fold predictions."""
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

        F_tr, F_va, _fts = extract_features_for_ablation(X_tr, X_va, cfg)

        if cfg.ablation == Ablation.NO_ANFIS:
            tr_evidence = summarise_fts_per_signal(F_tr, cfg.n_sets, cfg.fts_summary_mode)
            va_evidence = summarise_fts_per_signal(F_va, cfg.n_sets, cfg.fts_summary_mode)
            sw = compute_signal_weights(y_tr, tr_evidence) if cfg.bn_reliability else None

            bn = make_bn(cfg)
            bn.fit(tr_evidence, y_tr, signal_weights=sw)
            oof_post[va_idx] = bn.predict_proba(va_evidence)
            tr_post_fold = bn.predict_proba(tr_evidence)
        else:
            tr_probs, va_probs, _ = fit_signal_models(
                F_tr, F_va, y_tr, cfg, seed_offset=fold_i * 100
            )
            tr_probs, va_probs = calibrate_probs(tr_probs, va_probs, y_tr, cfg.calibrate)

            if cfg.ablation == Ablation.NO_BN:
                oof_post[va_idx] = fuse_without_bn(y_tr, tr_probs, va_probs, cfg)
                tr_post_fold = fuse_without_bn(y_tr, tr_probs, tr_probs, cfg)
            else:
                sw = compute_signal_weights(y_tr, tr_probs) if cfg.bn_reliability else None
                bn = make_bn(cfg)
                bn.fit(tr_probs, y_tr, signal_weights=sw)
                oof_post[va_idx] = bn.predict_proba(va_probs)
                tr_post_fold = bn.predict_proba(tr_probs)

        _print_fold_metrics(y_tr, y_va, tr_post_fold, oof_post[va_idx], fold_i)

    return CvResult(oof_post=oof_post, oof_y=y_train.copy(), fold_indices=fold_indices)


def run_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    user_train: np.ndarray,
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Backward-compatible tuple wrapper around run_cv_result()."""
    result = run_cv_result(X_train, y_train, user_train, cfg)
    return result.oof_post, result.oof_y, result.fold_indices


def retrain_and_test_result(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    cfg: ExperimentConfig,
) -> TestResult:
    """Retrain on full training data and evaluate the held-out test set."""
    F_tr, F_te, fts = extract_features_for_ablation(X_train, X_test, cfg)

    if cfg.ablation == Ablation.NO_ANFIS:
        tr_evidence = summarise_fts_per_signal(F_tr, cfg.n_sets, cfg.fts_summary_mode)
        te_evidence = summarise_fts_per_signal(F_te, cfg.n_sets, cfg.fts_summary_mode)
        sw = compute_signal_weights(y_train, tr_evidence) if cfg.bn_reliability else None

        bn = make_bn(cfg)
        bn.fit(tr_evidence, y_train, signal_weights=sw)
        te_post = bn.predict_proba(te_evidence)
        models = TrainedModels(
            fts=fts,
            bn=bn,
            anfis_list=None,
            test_evidence=te_evidence,
            train_evidence=tr_evidence,
        )
    else:
        tr_probs, te_probs, anfis_models = fit_signal_models(
            F_tr,
            F_te,
            y_train,
            cfg,
            seed_offset=cfg.final_seed_offset,
        )
        tr_probs, te_probs = calibrate_probs(tr_probs, te_probs, y_train, cfg.calibrate)

        if cfg.ablation == Ablation.NO_BN:
            te_post = fuse_without_bn(y_train, tr_probs, te_probs, cfg)
            bn = None
        else:
            sw = compute_signal_weights(y_train, tr_probs) if cfg.bn_reliability else None
            bn = make_bn(cfg)
            bn.fit(tr_probs, y_train, signal_weights=sw)
            te_post = bn.predict_proba(te_probs)

        models = TrainedModels(
            fts=fts,
            bn=bn,
            anfis_list=anfis_models,
            test_evidence=te_probs,
            train_evidence=tr_probs,
        )

    y_pred = (te_post >= threshold).astype(int)
    metrics = compute_oof_metrics(y_test, te_post, threshold)
    return TestResult(y_pred=y_pred, posterior=te_post, metrics=metrics, models=models)


def retrain_and_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    cfg: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict]:
    """Backward-compatible tuple wrapper around retrain_and_test_result()."""
    result = retrain_and_test_result(X_train, y_train, X_test, y_test, threshold, cfg)
    return result.y_pred, result.posterior, result.metrics, result.models.as_dict()


def _print_fold_metrics(
    y_train: np.ndarray,
    y_valid: np.ndarray,
    train_post: np.ndarray,
    valid_post: np.ndarray,
    fold_i: int,
) -> None:
    tr_acc = round(float(accuracy_score(y_train, (train_post >= 0.5).astype(int))), 4)
    try:
        tr_auc = round(float(roc_auc_score(y_train, train_post)), 4)
    except ValueError:
        tr_auc = float("nan")
    tr_f1 = round(float(f1_score(y_train, (train_post >= 0.5).astype(int), zero_division=0)), 4)

    va_acc = round(float(accuracy_score(y_valid, (valid_post >= 0.5).astype(int))), 4)
    try:
        va_auc = round(float(roc_auc_score(y_valid, valid_post)), 4)
    except ValueError:
        va_auc = float("nan")
    va_f1 = round(float(f1_score(y_valid, (valid_post >= 0.5).astype(int), zero_division=0)), 4)

    print(f"  Fold {fold_i + 1:>2} | Train | Acc: {tr_acc:.4f} | AUC: {tr_auc:.4f} | F1: {tr_f1:.4f}")
    print(f"  {' ':>6} |  Val  | Acc: {va_acc:.4f} | AUC: {va_auc:.4f} | F1: {va_f1:.4f}")
    print("-" * 80)
