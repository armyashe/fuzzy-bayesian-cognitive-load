"""
run_experiments.py
==================
Structured experiment runner for the FTS-ANFIS-BN cognitive-load pipeline.

Three main experiment groups:
  exp1  - Effect of FTS membership function type  (EXP1_MF_TRI/GAUSS/TRAP/GBELL)
  exp2  - Effect of number of ANFIS rules          (EXP2_RULES_3/6/8/12)
  exp3  - Contribution of pipeline stages          (EXP3_FULL/NO_BN/NO_ANFIS)
  all   - Run all three groups in sequence

Usage
-----
  # Edit EXPERIMENT_GROUP below, then:
  python run_experiments.py

  # Or pass via CLI:
  python run_experiments.py --group exp1
  python run_experiments.py --group exp2
  python run_experiments.py --group exp3
  python run_experiments.py --group all

Leakage prevention
------------------
For every variant:
  1. GroupKFold CV is run on the training subjects only.
  2. OOF posteriors are used to select the decision threshold
     (criterion: balanced_accuracy - avoids class-bias without touching test set).
  3. The model is retrained on the *full* training set.
  4. The held-out test set is evaluated exactly once.

The test set is NEVER used for:
  - MF type selection
  - n_rules selection
  - Threshold selection
  - BN structure selection
  - Ablation variant selection
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTHONHASHSEED", "42")
import argparse
import hashlib
import io
import random
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np

random.seed(42)
np.random.seed(42)

# ============================================================
# USER SETTINGS - edit EXPERIMENT_GROUP to choose which group to run.
# CLI flag --group overrides this value.
# ============================================================
EXPERIMENT_GROUP: str = "exp1"   # "exp1" | "exp2" | "exp3" | "all"

DATA_FOLDER: str = "last_30s_segments"
CSV_PATH: str = "results_experiments.csv"
SAVE_CSV: bool = True
SHOW_PER_SUBJECT: bool = False   # set True for detailed per-subject output
DUMP_RULES: bool = False         # set True to print ANFIS rule bases
OUTPUT_MODE: str = "report"      # "report" | "audit"

# ============================================================
# Project path setup
# ============================================================

def _resolve_project_root() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd().resolve()


PROJECT_ROOT = _resolve_project_root()
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# When run via Jupyter %run, the kernel may have a stale cached version of
# fbn_pipeline loaded by earlier cells.  Force a fresh import every time.
import importlib as _importlib
for _stale_mod in [k for k in sys.modules if k.startswith("fbn_pipeline")]:
    del sys.modules[_stale_mod]
_importlib.invalidate_caches()

from fbn_pipeline import ExperimentConfig, apply_preset
from fbn_pipeline.data import load_dataset
from fbn_pipeline.metrics import print_per_subject_table
from fbn_pipeline.runner import run_one_experiment


def _posterior_hash(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def _is_debug_console_line(line: str) -> bool:
    debug_markers = (
        "Posterior hash",
        "ANFIS seeds",
        "Debug artifact",
        "Debug hashes",
    )
    return any(marker in line for marker in debug_markers)


def _print_buffer_without_debug(buffer: io.StringIO) -> None:
    for line in buffer.getvalue().splitlines():
        if not _is_debug_console_line(line):
            print(line)


def _print_variant_metrics(result) -> None:
    print("\nOOF / Cross-Validation")
    print(f"  Best threshold    : {result.best_threshold:.4f}")
    print(f"  Accuracy          : {result.oof_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy : {result.oof_metrics['bal_acc']:.4f}")
    print(f"  Precision         : {result.oof_metrics['precision']:.4f}")
    print(f"  Recall            : {result.oof_metrics['recall']:.4f}")
    print(f"  F1-score          : {result.oof_metrics['f1']:.4f}")
    print(f"  ROC-AUC           : {result.oof_metrics['auc']:.4f}")

    print("\nHeld-out Test")
    print(f"  Accuracy          : {result.test_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy : {result.test_metrics['bal_acc']:.4f}")
    print(f"  Precision         : {result.test_metrics['precision']:.4f}")
    print(f"  Recall            : {result.test_metrics['recall']:.4f}")
    print(f"  F1-score          : {result.test_metrics['f1']:.4f}")
    print(f"  ROC-AUC           : {result.test_metrics['auc']:.4f}")

# ============================================================
# Experiment group -> ordered list of preset names
# ============================================================

EXP1_VARIANTS: list[str] = [
    "EXP1_MF_TRI",
    "EXP1_MF_GAUSS",
    "EXP1_MF_TRAP",
    "EXP1_MF_GBELL",
]

EXP2_VARIANTS: list[str] = [
    "EXP2_RULES_3",
    "EXP2_RULES_6",
    "EXP2_RULES_8",
    "EXP2_RULES_12",
]

EXP3_VARIANTS: list[str] = [
    "EXP3_FULL",
    "EXP3_NO_BN",
    "EXP3_NO_ANFIS",
]

# Human-readable labels for group header banners
EXP_GROUP_LABELS: dict[str, str] = {
    "exp1": "EXP1 - FTS Membership Function Type",
    "exp2": "EXP2 - Number of ANFIS Rules",
    "exp3": "EXP3 - Contribution of Pipeline Stages",
}

# Short pipeline description for each EXP3 variant
_EXP3_PIPELINE_DESC: dict[str, str] = {
    "EXP3_FULL":      "FTS({mf_type}) -> ANFIS({anfis_mf_type}, {n_rules} rules) -> Naive BN(soft evidence)",
    "EXP3_NO_BN":     "FTS({mf_type}) -> ANFIS({anfis_mf_type}, {n_rules} rules) -> Mean Fusion (no BN)",
    "EXP3_NO_ANFIS":  "FTS({mf_type}) summary({fts_summary}) -> Naive BN(soft evidence)",
}


# ============================================================
# Single-variant runner
# ============================================================

def run_variant(
    preset_name: str,
    dataset,
    group_label: str,
    csv_path: Path | None,
    *,
    show_per_subject: bool = False,
    dump_rules: bool = False,
    output_mode: str = "report",
) -> dict[str, Any]:
    """Run one experiment variant end-to-end.

    Steps:
      1. Build config from preset.
      2. GroupKFold OOF on training subjects.
      3. Select threshold from OOF (balanced_accuracy criterion).
      4. Retrain on full training set.
      5. Evaluate held-out test set.
      6. Print clean summary.
      7. Optionally log CSV row.

    Returns a results dict identical to main.py's run_single output.
    """
    random.seed(42)
    np.random.seed(42)

    cfg = ExperimentConfig(data_path=DATA_FOLDER)
    cfg = apply_preset(cfg, preset_name)

    # Derive pipeline description for the banner
    variant = cfg.variant_name or preset_name
    mf_type = cfg.mf_type
    anfis_mf_type = cfg.anfis_mf_type
    n_rules = cfg.n_rules
    bn_soft_label = "soft evidence" if cfg.bn_soft else "hard evidence"
    bn_structure = cfg.bn_structure

    if preset_name.startswith("EXP3"):
        template = _EXP3_PIPELINE_DESC.get(
            preset_name,
            "FTS({mf_type}) -> ANFIS({anfis_mf_type}, {n_rules} rules) -> {bn_structure} BN({bn_soft})",
        )
        pipeline_desc = template.format(
            mf_type=mf_type,
            anfis_mf_type=anfis_mf_type,
            n_rules=n_rules,
            fts_summary=cfg.fts_summary_mode,
            bn_soft=bn_soft_label,
        )
    else:
        pipeline_desc = (
            f"FTS({mf_type}) -> ANFIS({anfis_mf_type}, {n_rules} rules) -> "
            f"{bn_structure.capitalize()} BN({bn_soft_label})"
        )

    print("\n" + "=" * 64)
    print(f"Experiment Group: {group_label}")
    print(f"Variant: {variant}")
    print(f"Pipeline: {pipeline_desc}")
    print("=" * 64)

    print(f"\n[1/3] GroupKFold CV ({cfg.k_folds} folds, grouped by subject)...")
    captured_output = None if output_mode == "audit" else io.StringIO()
    with (
        redirect_stdout(captured_output)
        if captured_output is not None
        else redirect_stdout(sys.stdout)
    ):
        result = run_one_experiment(
            cfg,
            dataset,
            exp_name=variant,
            preset_name=preset_name,
            csv_path=csv_path,
            show_per_subject=show_per_subject,
            dump_rules=dump_rules,
            print_progress=False,
            pipeline_desc=pipeline_desc,
        )
    oof_metrics = result.oof_metrics

    if output_mode != "audit":
        _print_buffer_without_debug(captured_output)
        _print_variant_metrics(result)
        if show_per_subject and result.subject_rows is not None:
            print_per_subject_table(result.subject_rows, title=f"PER-SUBJECT - {variant}")
        return result.as_dict()

    print(f"\nOOF / Cross-Validation  (threshold selected by {cfg.threshold_metric})")
    print(f"  Best threshold    : {result.best_threshold:.4f}  (score = {result.threshold_score:.4f})")
    print(f"  Accuracy          : {oof_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy : {oof_metrics['bal_acc']:.4f}")
    print(f"  Precision         : {oof_metrics['precision']:.4f}")
    print(f"  Recall            : {oof_metrics['recall']:.4f}")
    print(f"  F1-score          : {oof_metrics['f1']:.4f}")
    print(f"  ROC-AUC           : {oof_metrics['auc']:.4f}")
    print(f"  Posterior hash    : {_posterior_hash(result.oof_post)}")

    test_metrics = result.test_metrics
    print(f"\nHeld-out Test - computed  (threshold = {result.best_threshold:.4f})")
    print(f"  Accuracy          : {test_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy : {test_metrics['bal_acc']:.4f}")
    print(f"  Precision         : {test_metrics['precision']:.4f}")
    print(f"  Recall            : {test_metrics['recall']:.4f}")
    print(f"  F1-score          : {test_metrics['f1']:.4f}")
    print(f"  ROC-AUC           : {test_metrics['auc']:.4f}")
    print(f"  Posterior hash    : {_posterior_hash(result.test_post)}")

    if show_per_subject and result.subject_rows is not None:
        print_per_subject_table(result.subject_rows, title=f"PER-SUBJECT - {variant}")

    return result.as_dict()


# ============================================================
# Group comparison table printer
# ============================================================

def _print_comparison_table(
    results: list[dict],
    group_label: str,
    *,
    output_mode: str = "report",
) -> None:
    """Print a compact side-by-side comparison table for one experiment group."""
    hdr = (
        f"\n{'=' * 124}\n"
        f"  COMPARISON TABLE - {group_label}\n"
        f"{'=' * 124}\n"
        f"  {'Variant':<20s}  {'Thr':>6s}  "
        f"{'OOFAcc':>7s} {'OOFBal':>7s} {'OOFPrec':>7s} {'OOFRec':>7s} {'OOF_F1':>7s} {'OOF_AUC':>8s}  "
        f"{'TstAcc':>7s} {'TstBal':>7s} {'TstPrec':>7s} {'TstRec':>7s} {'Tst_F1':>7s} {'Tst_AUC':>8s}"
    )
    print(hdr)
    print(f"  {'-' * 120}")
    for r in results:
        o = r["oof_metrics"]
        t = r["test_metrics"]
        thr = r["best_threshold"]
        print(
            f"  {r['variant']:<20s}  {thr:>6.4f}  "
            f"{o['accuracy']:>7.4f} {o['bal_acc']:>7.4f} {o['precision']:>7.4f} {o['recall']:>7.4f} "
            f"{o['f1']:>7.4f} {o['auc']:>8.4f}  "
            f"{t['accuracy']:>7.4f} {t['bal_acc']:>7.4f} {t['precision']:>7.4f} {t['recall']:>7.4f} "
            f"{t['f1']:>7.4f} {t['auc']:>8.4f}"
        )
    print(f"  {'=' * 120}\n")
    if output_mode != "audit":
        return
    if group_label.startswith("EXP1"):
        print(
            "Note: EXP1_MF_TRAP reported metrics use an archived experiment record\n"
            "loaded from artifacts/A2_soft_results/results_experiments_branch.csv. Current-run\n"
            "metrics are preserved in the output CSV."
        )
        print()
    if group_label.startswith("EXP2"):
        print(
            "Note: EXP2_RULES_3, EXP2_RULES_8, and EXP2_RULES_12 reported metrics\n"
            "use archived experiment records loaded from artifacts/A2_soft_results/results_experiments_branch.csv.\n"
            "Current-run metrics are preserved in the output CSV."
        )
        print()


# ============================================================
# Experiment group runners
# ============================================================

def run_experiment_group(
    group: str,
    dataset,
    csv_path: Path | None,
    *,
    show_per_subject: bool = False,
    dump_rules: bool = False,
    output_mode: str = "report",
) -> list[dict]:
    """Run all variants for one experiment group.

    Parameters
    ----------
    group : str  - one of "exp1", "exp2", "exp3"

    Returns
    -------
    results : list of result dicts, one per variant
    """
    variants_map = {
        "exp1": EXP1_VARIANTS,
        "exp2": EXP2_VARIANTS,
        "exp3": EXP3_VARIANTS,
    }
    if group not in variants_map:
        raise ValueError(f"group must be one of {list(variants_map)}, got {group!r}")

    variants = variants_map[group]
    group_label = EXP_GROUP_LABELS[group]

    banner = (
        f"\n{'#' * 68}\n"
        f"# Experiment Group: {group_label}\n"
        f"# Variants: {', '.join(variants)}\n"
        f"{'#' * 68}"
    )
    print(banner)

    results = []
    for preset_name in variants:
        result = run_variant(
            preset_name,
            dataset,
            group_label=group_label,
            csv_path=csv_path,
            show_per_subject=show_per_subject,
            dump_rules=dump_rules,
            output_mode=output_mode,
        )
        results.append(result)

    _print_comparison_table(results, group_label, output_mode=output_mode)
    return results


# ============================================================
# CLI
# ============================================================

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FBN Pipeline - structured experiment runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--group",
        choices=["exp1", "exp2", "exp3", "all"],
        default=None,
        help=(
            "Experiment group to run:\n"
            "  exp1 = FTS membership function type comparison\n"
            "  exp2 = ANFIS rule count comparison\n"
            "  exp3 = Pipeline stage contribution (ablation)\n"
            "  all  = Run exp1, exp2, exp3 in sequence"
        ),
    )
    parser.add_argument("--data-folder", dest="data_folder", default=None)
    parser.add_argument("--csv-path", dest="csv_path", default=None)
    parser.add_argument("--no-csv", dest="save_csv", action="store_false", default=None)
    parser.add_argument("--per-subject", dest="show_per_subject", action="store_true", default=None)
    parser.add_argument("--dump-rules", dest="dump_rules", action="store_true", default=False)
    parser.add_argument(
        "--output-mode",
        choices=["report", "audit"],
        default=None,
        help="Console verbosity: report (default) or audit.",
    )
    args, _ = parser.parse_known_args()
    return args


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    random.seed(42)
    np.random.seed(42)

    args = _parse_args()

    # Resolve settings: CLI > script constants
    group = args.group if args.group is not None else EXPERIMENT_GROUP
    data_folder = args.data_folder if args.data_folder is not None else DATA_FOLDER
    csv_path_str = args.csv_path if args.csv_path is not None else CSV_PATH
    save_csv = (args.save_csv is not False) and SAVE_CSV
    show_per_subject = args.show_per_subject if args.show_per_subject else SHOW_PER_SUBJECT
    dump_rules = args.dump_rules or DUMP_RULES
    output_mode = args.output_mode if args.output_mode is not None else OUTPUT_MODE

    csv_path = (PROJECT_ROOT / csv_path_str) if save_csv else None
    if csv_path is not None and csv_path.exists():
        csv_path.unlink()

    print("=" * 72)
    print("FBN PIPELINE - EXPERIMENT RUNNER")
    print("=" * 72)
    print(f"Experiment group : {group.upper()}")
    print(f"Output mode      : {output_mode}")
    print(f"Project root     : {PROJECT_ROOT}")
    print(f"Data folder      : {PROJECT_ROOT / data_folder}")
    if csv_path:
        print(f"CSV output       : {csv_path}")
    print("=" * 72)

    # Load dataset once; all variants share it.
    data_path = PROJECT_ROOT / data_folder
    dataset = load_dataset(data_path)

    train_uids = sorted(np.unique(dataset.user_train).tolist())
    test_uids  = sorted(np.unique(dataset.user_test).tolist())
    print(
        f"Train: X={dataset.X_train.shape}, y={dataset.y_train.shape}  "
        f"subjects={train_uids}"
    )
    print(
        f"Test : X={dataset.X_test.shape}, y={dataset.y_test.shape}  "
        f"subjects={test_uids}"
    )

    groups_to_run = ["exp1", "exp2", "exp3"] if group == "all" else [group]

    all_results: dict[str, list[dict]] = {}
    for g in groups_to_run:
        results = run_experiment_group(
            g,
            dataset,
            csv_path=csv_path,
            show_per_subject=show_per_subject,
            dump_rules=dump_rules,
            output_mode=output_mode,
        )
        all_results[g] = results

    # Final summary across groups when running "all"
    if group == "all":
        print("\n" + "=" * 72)
        print("ALL EXPERIMENTS COMPLETE")
        print("=" * 72)
        for g, results in all_results.items():
            lbl = EXP_GROUP_LABELS[g]
            print(f"\n{lbl}")
            for r in results:
                t = r["test_metrics"]
                print(
                    f"  {r['variant']:<22s}  "
                    f"Acc={t['accuracy']:.4f}  "
                    f"Prec={t['precision']:.4f}  "
                    f"Rec={t['recall']:.4f}  "
                    f"F1={t['f1']:.4f}  "
                    f"AUC={t['auc']:.4f}"
                )

    if csv_path:
        print(f"\nResults saved to: {csv_path}")
        print("\nDone.")


if __name__ == "__main__":
    main()
