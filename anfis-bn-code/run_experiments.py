#!/usr/bin/env python
"""
run_experiments.py  --  CLI for FTS -> ANFIS (5-layer TSK) -> Bayesian Network
===============================================================================

Usage examples
--------------
    python run_experiments.py --mode single --exp A1
    python run_experiments.py --mode single --n_sets 4 --mf_type gaussian --lr 0.003
    python run_experiments.py --mode grid --primary_metric accuracy

Notes
-----
- In grid mode, this script (by default) ALSO evaluates TEST for each config
  (diagnostic). DO NOT use per-config test metrics to select the best model.
  Best config is still selected by OOF metric.
"""

import argparse
import csv
import importlib
import itertools
import json
import os
import sys
import time
import types
import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Import project modules  (load-data.py has a hyphen -> importlib)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from fts_encoder import FTSEncoder
from anfis_tsk import ANFIS_TSK
from bn_fusion import BNFusion
from eval_utils import find_best_threshold, print_metrics

_ld = importlib.import_module("load-data")
load_data = _ld.load_data


# ========================= REPRODUCIBILITY =========================
def setup_seed(seed: int) -> None:
    """Fix random seed for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ========================= EXPERIMENT PRESETS =========================
experiment_presets = {
    "A1": {
        "n_sets": 3, "mf_type": "triangular", "mf_sigma_scale": 0.5,
        "gbell_b": 2.0, "fts_norm": "total_sum",
        "n_rules": 8, "sigma_scale": 1.5, "ridge_lambda": 0.1,
        "n_epochs": 15, "lr": 0.001,
        "n_bins": 3, "alpha": 1.0, "k_folds": 5,
    },
    "A2": {
        "n_sets": 4, "mf_type": "gaussian", "mf_sigma_scale": 0.6,
        "gbell_b": 2.0, "fts_norm": "total_sum",
        "n_rules": 10, "sigma_scale": 1.3, "ridge_lambda": 0.01,
        "n_epochs": 20, "lr": 0.003,
        "n_bins": 3, "alpha": 1.0, "k_folds": 5,
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
}


# ========================= ARGPARSE =========================
def parse_args():
    p = argparse.ArgumentParser(
        description="FTS -> ANFIS -> BN experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Basic
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k_folds", type=int, default=5)
    p.add_argument(
        "--data_path", type=str,
        default=os.path.join(_SCRIPT_DIR, "last_30s_segments"),
    )

    # FTS
    p.add_argument("--n_sets", type=int, default=3)
    p.add_argument(
        "--mf_type", type=str, default="triangular",
        choices=["triangular", "gaussian", "trapezoid", "gbell"],
    )
    p.add_argument("--mf_sigma_scale", type=float, default=0.5)
    p.add_argument("--gbell_b", type=float, default=2.0)
    p.add_argument(
        "--fts_norm", type=str, default="total_sum",
        choices=["total_sum", "row"],
    )

    # ANFIS
    p.add_argument("--n_rules", type=int, default=8)
    p.add_argument("--sigma_scale", type=float, default=1.5)
    p.add_argument("--ridge_lambda", type=float, default=0.1)
    p.add_argument("--n_epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=0.001)

    # BN
    p.add_argument("--n_bins", type=int, default=3)
    p.add_argument("--alpha", type=float, default=1.0)

    # Threshold
    p.add_argument(
        "--primary_metric", type=str, default="accuracy",
        choices=["accuracy", "balanced_accuracy", "f1"],
    )
    p.add_argument("--threshold_grid", type=int, default=181)

    # Mode / output
    p.add_argument("--mode", type=str, default="single",
                   choices=["single", "grid"])
    p.add_argument("--save_csv", type=str, default="experiment_results.csv")
    p.add_argument("--save_best_json", type=str, default="best_config.json")
    p.add_argument("--save_test_report", type=str, default="test_report.txt")

    # Preset
    p.add_argument("--exp", type=str, default=None)

    # GRID controls
    p.add_argument("--max_configs", type=int, default=100,
                   help="Run only first N grid configs after optional shuffle (0 = all).")

    # default ON shuffle for better coverage
    p.add_argument("--grid_shuffle", dest="grid_shuffle", action="store_true",
                   help="Shuffle grid configs before slicing max_configs.")
    p.add_argument("--no_grid_shuffle", dest="grid_shuffle", action="store_false",
                   help="Disable shuffling grid configs.")
    p.set_defaults(grid_shuffle=True)

    # default ON: evaluate test for each config (diagnostic)
    p.add_argument("--eval_test_each", dest="eval_test_each", action="store_true",
                   help="Evaluate test for each config (diagnostic).")
    p.add_argument("--no_eval_test_each", dest="eval_test_each", action="store_false",
                   help="Disable per-config test evaluation (faster, cleaner).")
    p.set_defaults(eval_test_each=True)

    return p.parse_args()


def apply_preset(args):
    """Override args with preset values when --exp is given."""
    if args.exp is not None and args.exp in experiment_presets:
        preset = experiment_presets[args.exp]
        for k, v in preset.items():
            setattr(args, k, v)
        print(f"[INFO] Applied preset '{args.exp}': {preset}")
    elif args.exp is not None:
        print(
            f"[WARN] Preset '{args.exp}' not found.  "
            f"Available: {list(experiment_presets.keys())}.  Using CLI args."
        )
    return args


# ========================= CSV LOGGING =========================
CSV_COLUMNS = [
    "exp", "seed", "k_folds",
    "n_sets", "mf_type", "mf_sigma_scale", "gbell_b", "fts_norm",
    "n_rules", "anfis_sigma_scale", "ridge_lambda", "n_epochs", "lr",
    "n_bins", "alpha",
    "best_thr", "oof_acc", "oof_f1", "test_acc", "test_f1",
]


def log_to_csv(filepath: str, row: dict) -> None:
    """Append one result row to CSV (creates header on first write)."""
    file_exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


def make_row(cfg, exp_name, best_thr, oof_acc, oof_f1,
             test_acc="", test_f1=""):
    """Build a CSV row dict from a config (dict or namespace) + metrics."""
    c = cfg if isinstance(cfg, dict) else vars(cfg)
    return {
        "exp": exp_name,
        "seed": c.get("seed", ""),
        "k_folds": c.get("k_folds", ""),
        "n_sets": c["n_sets"],
        "mf_type": c["mf_type"],
        "mf_sigma_scale": c["mf_sigma_scale"],
        "gbell_b": c["gbell_b"],
        "fts_norm": c["fts_norm"],
        "n_rules": c["n_rules"],
        "anfis_sigma_scale": c["sigma_scale"],
        "ridge_lambda": c["ridge_lambda"],
        "n_epochs": c["n_epochs"],
        "lr": c["lr"],
        "n_bins": c["n_bins"],
        "alpha": c["alpha"],
        "best_thr": best_thr,
        "oof_acc": oof_acc,
        "oof_f1": oof_f1,
        "test_acc": test_acc,
        "test_f1": test_f1,
    }


# ===================================================================
# CORE PIPELINE
# ===================================================================
def _ns(base_args, override: dict):
    """Create a SimpleNamespace by merging *base_args* with *override*."""
    ns = types.SimpleNamespace(**vars(base_args))
    for k, v in override.items():
        setattr(ns, k, v)
    return ns


def run_cv(X_train, y_train, user_train, cfg):
    """
    Run GroupKFold CV.

    Returns
    -------
    oof_posteriors : ndarray (N,)
    oof_labels     : ndarray (N,)
    """
    n_signals = X_train.shape[1]  # expected 4

    gkf = GroupKFold(n_splits=cfg.k_folds)
    oof_post = np.full(len(y_train), np.nan)

    for fold_i, (tr_idx, va_idx) in enumerate(
        gkf.split(X_train, y_train, user_train)
    ):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr = y_train[tr_idx]

        # --- 1) FTS encoder (fit on train fold only) ---
        fts = FTSEncoder(
            n_sets=cfg.n_sets,
            mf_type=cfg.mf_type,
            mf_sigma_scale=cfg.mf_sigma_scale,
            gbell_b=cfg.gbell_b,
            norm_mode=cfg.fts_norm,
        )
        F_tr = fts.fit_transform(X_tr)
        F_va = fts.transform(X_va)

        # infer per-signal feature width robustly
        feat_per_sig = F_tr.shape[1] // n_signals

        # --- 2) One ANFIS per signal ---
        tr_probs = np.zeros((len(tr_idx), n_signals))
        va_probs = np.zeros((len(va_idx), n_signals))

        for s in range(n_signals):
            col_s = s * feat_per_sig
            col_e = col_s + feat_per_sig

            anfis = ANFIS_TSK(
                n_rules=cfg.n_rules,
                sigma_scale=cfg.sigma_scale,
                ridge_lambda=cfg.ridge_lambda,
                n_epochs=cfg.n_epochs,
                lr=cfg.lr,
                random_state=cfg.seed + fold_i * 100 + s,
            )
            anfis.fit(F_tr[:, col_s:col_e], y_tr)

            tr_probs[:, s] = anfis.predict_proba(F_tr[:, col_s:col_e])
            va_probs[:, s] = anfis.predict_proba(F_va[:, col_s:col_e])

        # --- 3) BN Fusion (fit on train fold probabilities) ---
        bn = BNFusion(n_bins=cfg.n_bins, alpha=cfg.alpha)
        bn.fit(tr_probs, y_tr)
        oof_post[va_idx] = bn.predict_proba(va_probs)

    return oof_post, y_train.copy()


def retrain_and_test(X_train, y_train, X_test, y_test, threshold, cfg):
    """Retrain on full train set and evaluate on held-out test set."""
    n_signals = X_train.shape[1]

    fts = FTSEncoder(
        n_sets=cfg.n_sets,
        mf_type=cfg.mf_type,
        mf_sigma_scale=cfg.mf_sigma_scale,
        gbell_b=cfg.gbell_b,
        norm_mode=cfg.fts_norm,
    )
    F_tr = fts.fit_transform(X_train)
    F_te = fts.transform(X_test)

    feat_per_sig = F_tr.shape[1] // n_signals

    tr_probs = np.zeros((len(y_train), n_signals))
    te_probs = np.zeros((len(y_test), n_signals))

    for s in range(n_signals):
        col_s = s * feat_per_sig
        col_e = col_s + feat_per_sig

        anfis = ANFIS_TSK(
            n_rules=cfg.n_rules,
            sigma_scale=cfg.sigma_scale,
            ridge_lambda=cfg.ridge_lambda,
            n_epochs=cfg.n_epochs,
            lr=cfg.lr,
            random_state=cfg.seed + s,
        )
        anfis.fit(F_tr[:, col_s:col_e], y_train)

        tr_probs[:, s] = anfis.predict_proba(F_tr[:, col_s:col_e])
        te_probs[:, s] = anfis.predict_proba(F_te[:, col_s:col_e])

    bn = BNFusion(n_bins=cfg.n_bins, alpha=cfg.alpha)
    bn.fit(tr_probs, y_train)

    te_post = bn.predict_proba(te_probs)
    y_pred = (te_post >= threshold).astype(int)

    t_acc = round(accuracy_score(y_test, y_pred), 4)
    t_f1 = round(f1_score(y_test, y_pred, zero_division=0), 4)
    return y_pred, te_post, t_acc, t_f1


# ===================================================================
# MODE: SINGLE
# ===================================================================
def run_single(args, X_train, y_train, user_train, X_test, y_test):
    print("\n" + "=" * 60)
    print("  MODE: SINGLE")
    print("=" * 60)

    # 1) GroupKFold CV
    print(f"\n[1] Running {args.k_folds}-fold GroupKFold CV ...")
    oof_post, oof_y = run_cv(X_train, y_train, user_train, args)

    # 2) Threshold tuning on OOF (NOT test)
    print(f"[2] Tuning threshold on OOF (metric={args.primary_metric}) ...")
    best_thr, best_score = find_best_threshold(
        oof_y, oof_post,
        metric=args.primary_metric,
        n_grid=args.threshold_grid,
    )

    oof_pred = (oof_post >= best_thr).astype(int)
    oof_acc = round(accuracy_score(oof_y, oof_pred), 4)
    oof_f1 = round(f1_score(oof_y, oof_pred, zero_division=0), 4)

    print(f"    Best threshold = {best_thr}")
    print(f"    OOF {args.primary_metric} = {best_score}")
    print(f"    OOF accuracy = {oof_acc},  OOF F1 = {oof_f1}")

    # 3) Retrain full train -> evaluate test
    print("[3] Retraining on full train -> evaluating test ...")
    y_pred, _, test_acc, test_f1 = retrain_and_test(
        X_train, y_train, X_test, y_test, best_thr, args,
    )

    print(f"[TEST] acc={test_acc} f1={test_f1}")
    print_metrics(y_test, y_pred, title="TEST SET RESULTS")

    # 4) Log CSV
    exp_name = args.exp if args.exp else "custom"
    row = make_row(vars(args), exp_name, best_thr, oof_acc, oof_f1,
                   test_acc, test_f1)
    log_to_csv(args.save_csv, row)
    print(f"[4] Results appended to {args.save_csv}")


# ===================================================================
# GRID SEARCH SPACE
# ===================================================================
def generate_grid_configs(base_args):
    """Generate list of config dicts for exhaustive grid search."""
    grid_n_sets = [3, 4, 5]
    grid_fts_norm = ["total_sum", "row"]
    grid_n_rules = [6, 8, 10, 12]
    grid_sigma_scale = [0.8, 1.0, 1.3, 1.5]
    grid_ridge = [0.0, 0.01, 0.1]
    grid_epochs = [10, 20, 50]
    grid_lr = [0.001, 0.003, 0.005]
    grid_n_bins = [3, 5]
    grid_alpha = [0.5, 1.0, 2.0]

    mf_specs = {
        "triangular": {"mf_sigma_scale": [0.5],              "gbell_b": [2.0]},
        "gaussian":   {"mf_sigma_scale": [0.4, 0.6, 0.8],   "gbell_b": [2.0]},
        "trapezoid":  {"mf_sigma_scale": [0.5],              "gbell_b": [2.0]},
        "gbell":      {"mf_sigma_scale": [0.4, 0.6, 0.8],   "gbell_b": [1.5, 2.0, 3.0]},
    }

    configs: list[dict] = []
    for mf_type, spec in mf_specs.items():
        for combo in itertools.product(
            grid_n_sets,
            spec["mf_sigma_scale"],
            spec["gbell_b"],
            grid_fts_norm,
            grid_n_rules,
            grid_sigma_scale,
            grid_ridge,
            grid_epochs,
            grid_lr,
            grid_n_bins,
            grid_alpha,
        ):
            configs.append({
                "mf_type": mf_type,
                "n_sets": combo[0],
                "mf_sigma_scale": combo[1],
                "gbell_b": combo[2],
                "fts_norm": combo[3],
                "n_rules": combo[4],
                "sigma_scale": combo[5],
                "ridge_lambda": combo[6],
                "n_epochs": combo[7],
                "lr": combo[8],
                "n_bins": combo[9],
                "alpha": combo[10],
                "seed": base_args.seed,
                "k_folds": base_args.k_folds,
            })

    return configs


# ===================================================================
# MODE: GRID
# ===================================================================
def run_grid(args, X_train, y_train, user_train, X_test, y_test):
    print("\n" + "=" * 60)
    print("  MODE: GRID SEARCH (limited)")
    print("=" * 60)

    configs = generate_grid_configs(args)

    # shuffle then slice to max_configs
    if args.grid_shuffle:
        rng = np.random.default_rng(args.seed)
        rng.shuffle(configs)

    if args.max_configs and args.max_configs > 0:
        configs = configs[:args.max_configs]

    total = len(configs)
    print(f"  Total configurations to run: {total}")
    print(f"  Per-config TEST eval: {'ON' if args.eval_test_each else 'OFF'} (diagnostic)\n")

    results: list[tuple] = []  # (primary, oof_f1, cfg_dict, thr, oof_acc, test_acc, test_f1)
    t0 = time.time()

    for i, cfg_dict in enumerate(configs):
        cfg = _ns(args, cfg_dict)

        elapsed = time.time() - t0
        eta = (elapsed / (i + 1)) * (total - i - 1) if i > 0 else 0
        print(
            f"[Grid {i + 1}/{total}] "
            f"mf={cfg.mf_type} n_sets={cfg.n_sets} "
            f"sigma_mf={cfg.mf_sigma_scale} gbell_b={cfg.gbell_b} "
            f"norm={cfg.fts_norm} rules={cfg.n_rules} "
            f"sig={cfg.sigma_scale} ridge={cfg.ridge_lambda} "
            f"ep={cfg.n_epochs} lr={cfg.lr} "
            f"bins={cfg.n_bins} alpha={cfg.alpha}"
            f"  | elapsed={elapsed:.0f}s ETA={eta:.0f}s"
        )

        try:
            setup_seed(cfg.seed)

            # ---- CV / OOF ----
            oof_post, oof_y = run_cv(X_train, y_train, user_train, cfg)

            best_thr, _ = find_best_threshold(
                oof_y, oof_post,
                metric=args.primary_metric,
                n_grid=args.threshold_grid,
            )

            oof_pred = (oof_post >= best_thr).astype(int)
            oof_acc = round(accuracy_score(oof_y, oof_pred), 4)
            oof_f1 = round(f1_score(oof_y, oof_pred, zero_division=0), 4)

            if args.primary_metric == "accuracy":
                primary = oof_acc
            elif args.primary_metric == "f1":
                primary = oof_f1
            else:
                primary = round(balanced_accuracy_score(oof_y, oof_pred), 4)

            # ---- TEST per-config (diagnostic) ----
            test_acc = ""
            test_f1 = ""
            if args.eval_test_each:
                _, _, test_acc, test_f1 = retrain_and_test(
                    X_train, y_train, X_test, y_test, best_thr, cfg
                )

            # store result
            results.append((primary, oof_f1, cfg_dict, best_thr, oof_acc, test_acc, test_f1))

            # log CSV (with test metrics if computed)
            row = make_row(
                cfg_dict, f"grid_{i + 1}",
                best_thr, oof_acc, oof_f1,
                test_acc, test_f1
            )
            log_to_csv(args.save_csv, row)

            print(f"         -> OOF acc={oof_acc} f1={oof_f1} thr={best_thr} | TEST acc={test_acc} f1={test_f1}")

        except Exception as e:
            print(f"         -> ERROR: {e}")
            continue

    if not results:
        print("[ERROR] No configurations completed successfully.")
        return

    # ---- Best config by OOF primary desc, then OOF f1 desc ----
    results.sort(key=lambda x: (x[0], x[1]), reverse=True)

    best_primary, best_f1, best_cfg, best_thr, best_oof_acc, best_test_acc, best_test_f1 = results[0]
    print("\n" + "=" * 60)
    print("  BEST CONFIG (selected by OOF)")
    print("=" * 60)
    for k, v in sorted(best_cfg.items()):
        print(f"  {k:25s}: {v}")
    print(f"  {'best_threshold':25s}: {best_thr}")
    print(f"  {'oof_' + args.primary_metric:25s}: {best_primary}")
    print(f"  {'oof_f1':25s}: {best_f1}")
    print(f"  {'oof_acc':25s}: {best_oof_acc}")
    print("=" * 60)

    # ---- Retrain best config on full train -> evaluate test (final) ----
    print("\n[Final] Retraining best config on full train -> test ...")
    setup_seed(args.seed)
    best_ns = _ns(args, best_cfg)
    y_pred, _, test_acc, test_f1 = retrain_and_test(
        X_train, y_train, X_test, y_test, best_thr, best_ns,
    )

    print(f"[TEST - BEST] acc={test_acc} f1={test_f1}")
    print_metrics(y_test, y_pred, title="TEST SET RESULTS (BEST CONFIG)")

    # Append best row WITH test metrics
    row = make_row(best_cfg, "grid_best", best_thr,
                   best_oof_acc, best_f1, test_acc, test_f1)
    log_to_csv(args.save_csv, row)

    # ---- Save best_config.json ----
    best_json = {
        **best_cfg,
        "primary_metric": args.primary_metric,
        "best_threshold": best_thr,
        "oof_accuracy": best_oof_acc,
        "oof_f1": best_f1,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
    }
    with open(args.save_best_json, "w", encoding="utf-8") as f:
        json.dump(best_json, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {args.save_best_json}")

    # ---- Save test_report.txt ----
    report = classification_report(y_test, y_pred, target_names=["rest", "task"])
    cm = confusion_matrix(y_test, y_pred)
    with open(args.save_test_report, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  TEST REPORT - BEST CONFIG FROM GRID SEARCH\n")
        f.write("=" * 60 + "\n\n")
        f.write("Config:\n")
        for k, v in sorted(best_json.items()):
            f.write(f"  {k}: {v}\n")
        f.write(f"\nThreshold: {best_thr}\n")
        f.write(f"Test Accuracy:  {test_acc}\n")
        f.write(f"Test F1-score:  {test_f1}\n\n")
        f.write("Confusion Matrix:\n")
        f.write("              Predicted\n")
        f.write("              rest  task\n")
        f.write(f"  Actual rest {cm[0, 0]:4d}  {cm[0, 1]:4d}\n")
        f.write(f"  Actual task {cm[1, 0]:4d}  {cm[1, 1]:4d}\n\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
    print(f"[Saved] {args.save_test_report}")


# ===================================================================
# MAIN
# ===================================================================
def main():
    args = parse_args()
    args = apply_preset(args)
    setup_seed(args.seed)

    print(f"\n[Loading data from {args.data_path}]")
    X_train, y_train, user_train, X_test, y_test, user_test = load_data(args.data_path)

    if args.mode == "single":
        run_single(args, X_train, y_train, user_train, X_test, y_test)
    elif args.mode == "grid":
        run_grid(args, X_train, y_train, user_train, X_test, y_test)

    print("\nDone.")


if __name__ == "__main__":
    main()