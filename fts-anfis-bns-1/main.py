from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# ============================================================
# USER SETTINGS - đổi ở đây rồi %run main.py
# ============================================================
MODE = "single"                    # "single"
PRESET_NAME = "A2_soft"           # None | "A1" | "A2" | ...
EXP_NAME = "main_run"
DATA_FOLDER = "last_30s_segments"
CSV_PATH = "results_main.csv"
SAVE_CSV = True
SHOW_PER_SUBJECT = True
DUMP_RULES = False

OVERRIDES: dict[str, Any] = {
    # "seed": 42,
    # "k_folds": 5,
    # "n_epochs": 20,
}


def _resolve_project_root() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd().resolve()


PROJECT_ROOT = _resolve_project_root()
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fbn_pipeline import ExperimentConfig, apply_preset, setup_seed
from fbn_pipeline.data import load_dataset
from fbn_pipeline.metrics import (
    find_best_threshold,
    per_subject_metrics,
    print_config,
    print_metrics,
    print_per_subject_table,
)
from fbn_pipeline.pipeline import compute_oof_metrics, retrain_and_test, run_cv
from fbn_pipeline.reporting import (
    dump_anfis_rules,
    dump_bn_cpt,
    dump_fts_transitions,
    log_to_csv,
    make_result_row,
)


def build_config() -> ExperimentConfig:
    cfg = ExperimentConfig(data_path=DATA_FOLDER)
    cfg = apply_preset(cfg, PRESET_NAME)
    if OVERRIDES:
        cfg = cfg.merged(**OVERRIDES)
    if DUMP_RULES:
        cfg = cfg.merged(dump_rules=True)
    return cfg


def run_single() -> dict[str, Any]:
    cfg = build_config()
    setup_seed(cfg.seed)
    data_path = PROJECT_ROOT / cfg.data_path
    dataset = load_dataset(data_path)

    print("=" * 72)
    print("FBN PIPELINE - MAIN RUNNER")
    print("=" * 72)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data folder  : {data_path}")
    print(f"Preset       : {PRESET_NAME}")
    print(f"Experiment   : {EXP_NAME}")
    print("=" * 72)
    print(
        f"Train: X={dataset.X_train.shape}, y={dataset.y_train.shape}, users={dataset.user_train.shape}"
    )
    print(
        f"Test : X={dataset.X_test.shape}, y={dataset.y_test.shape}, users={dataset.user_test.shape}"
    )
    print_config(cfg.to_dict())

    print("\n[1/3] Running GroupKFold CV...")
    oof_post, oof_y, fold_indices = run_cv(
        dataset.X_train,
        dataset.y_train,
        dataset.user_train,
        cfg,
    )
    best_thr, thr_score = find_best_threshold(
        oof_y,
        oof_post,
        metric=cfg.threshold_metric,
        n_grid=cfg.threshold_grid,
    )
    oof_metrics = compute_oof_metrics(oof_y, oof_post, best_thr)
    oof_pred = (oof_post >= best_thr).astype(int)

    print(f"Best threshold ({cfg.threshold_metric}) = {best_thr:.4f}")
    print(f"Best threshold score                = {thr_score:.4f}")
    print_metrics(oof_y, oof_pred, oof_post, title="OOF / Cross-Validation")

    print("\n[2/3] Retraining on full training set and evaluating on test set...")
    test_pred, test_post, test_metrics, models = retrain_and_test(
        dataset.X_train,
        dataset.y_train,
        dataset.X_test,
        dataset.y_test,
        best_thr,
        cfg,
    )
    print_metrics(dataset.y_test, test_pred, test_post, title="Held-out Test")

    print("\n[3/3] Reporting...")
    subject_rows = None
    if SHOW_PER_SUBJECT:
        subject_rows = per_subject_metrics(
            dataset.y_test,
            test_pred,
            test_post,
            dataset.user_test,
        )
        print_per_subject_table(subject_rows)

    if cfg.dump_rules:
        dump_anfis_rules(models["anfis_list"], n_sets=cfg.n_sets)
        dump_fts_transitions(models["fts"], dataset.X_train[0])
        if len(dataset.y_test) > 0:
            dump_bn_cpt(models["bn"], sample_probs=models["te_probs"][0])

    if SAVE_CSV:
        csv_path = PROJECT_ROOT / CSV_PATH
        row = make_result_row(
            cfg.to_dict(),
            exp_name=EXP_NAME,
            best_thr=best_thr,
            oof=oof_metrics,
            test=test_metrics,
            eval_protocol="subject-independent hold-out + GroupKFold OOF",
            rank_metric=cfg.rank_metric,
            thr_metric=cfg.threshold_metric,
        )
        log_to_csv(str(csv_path), row)
        print(f"Saved CSV row to: {csv_path}")

    print("\nDone.")
    return {
        "config": cfg,
        "dataset": dataset,
        "fold_indices": fold_indices,
        "best_threshold": best_thr,
        "oof_post": oof_post,
        "oof_pred": oof_pred,
        "oof_metrics": oof_metrics,
        "test_post": test_post,
        "test_pred": test_pred,
        "test_metrics": test_metrics,
        "subject_rows": subject_rows,
        "models": models,
    }


if MODE != "single":
    raise ValueError("Phiên bản main.py này đang chạy MODE='single'.")

RUN_OUTPUT = run_single()
