"""
run_experiments.py
==================
Development-only experiment runner for the FTS-ANFIS-BN cognitive-load pipeline.

This script evaluates candidate configurations on the 18 training subjects only:
  1. GroupKFold OOF on training subjects
  2. OOF-only threshold selection
  3. OOF-only candidate ranking
  4. Development-only CSV and ranking artifact output

It intentionally does not run held-out test evaluation.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHONHASHSEED", "42")
random.seed(42)
np.random.seed(42)

EXPERIMENT_GROUP: str = "exp1"
DATA_FOLDER: str = "last_30s_segments"
CSV_PATH: str = "artifacts/protocol_v2/development_results.csv"
RANKING_PATH: str = "artifacts/protocol_v2/development_ranking.json"


def _resolve_project_root() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd().resolve()


PROJECT_ROOT = _resolve_project_root()
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fbn_pipeline import (
    ExperimentConfig,
    apply_preset,
    run_development,
    select_best_development_config,
    write_development_results_csv,
)
from fbn_pipeline.data import load_dataset


def _print_variant_metrics(result) -> None:
    print("\nOOF / Development")
    print(f"  Best threshold    : {result.threshold:.4f}")
    print(f"  Accuracy          : {result.oof_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy : {result.oof_metrics['bal_acc']:.4f}")
    print(f"  Precision         : {result.oof_metrics['precision']:.4f}")
    print(f"  Recall            : {result.oof_metrics['recall']:.4f}")
    print(f"  F1-score          : {result.oof_metrics['f1']:.4f}")
    print(f"  ROC-AUC           : {result.oof_metrics['auc']:.4f}")


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

EXP_GROUP_LABELS: dict[str, str] = {
    "exp1": "EXP1 - FTS Membership Function Type",
    "exp2": "EXP2 - Number of ANFIS Rules",
    "exp3": "EXP3 - Contribution of Pipeline Stages",
}


def run_variant(
    preset_name: str,
    dataset,
    group_label: str,
):
    cfg = ExperimentConfig(data_path=DATA_FOLDER)
    cfg = apply_preset(cfg, preset_name)
    variant = cfg.variant_name or preset_name

    print("\n" + "=" * 64)
    print(f"Experiment Group: {group_label}")
    print(f"Variant: {variant}")
    print("Protocol: DEVELOPMENT_ONLY (training subjects + OOF only)")
    print("=" * 64)

    result = run_development(
        dataset.X_train,
        dataset.y_train,
        dataset.user_train,
        cfg,
        configuration_id=variant,
        print_progress=True,
    )
    _print_variant_metrics(result)
    return result


def _print_comparison_table(results: list[dict], group_label: str) -> None:
    print(
        f"\n{'=' * 96}\n"
        f"  DEVELOPMENT COMPARISON TABLE - {group_label}\n"
        f"{'=' * 96}\n"
        f"  {'Variant':<20s} {'Thr':>6s} {'OOFAcc':>7s} {'OOFBal':>7s} "
        f"{'OOFMac':>7s} {'OOF_F1':>7s} {'OOF_AUC':>8s}\n"
        f"  {'-' * 92}"
    )
    for result in results:
        result_dict = result.as_dict()
        oof = result_dict["oof_metrics"]
        print(
            f"  {result_dict['variant']:<20s} {result_dict['best_threshold']:>6.4f} "
            f"{oof['accuracy']:>7.4f} {oof['bal_acc']:>7.4f} "
            f"{oof['macro_f1']:>7.4f} {oof['f1']:>7.4f} {oof['auc']:>8.4f}"
        )
    print(f"  {'=' * 92}\n")


def _print_selection_summary(selection) -> None:
    chosen = selection.selected_result
    print("Selection rule")
    print(f"  Primary metric    : {selection.ranking_metric}")
    print(f"  Tie-break rules   : {' > '.join(selection.tie_break_rules)}")
    print(f"  Selected config   : {chosen.configuration_id}")
    print(f"  Selected threshold: {chosen.threshold:.4f}")
    print(f"  Selected OOF BalAcc: {chosen.oof_metrics['bal_acc']:.4f}")
    print(f"  Selected OOF MacroF1: {chosen.oof_metrics['macro_f1']:.4f}")
    print(f"  Selected OOF AUC  : {chosen.oof_metrics['auc']:.4f}")


def run_experiment_group(
    group: str,
    dataset,
    *,
    csv_path: Path,
    ranking_path: Path,
) -> list[dict]:
    variants_map = {
        "exp1": EXP1_VARIANTS,
        "exp2": EXP2_VARIANTS,
        "exp3": EXP3_VARIANTS,
    }
    if group not in variants_map:
        raise ValueError(f"group must be one of {list(variants_map)}, got {group!r}")

    variants = variants_map[group]
    group_label = EXP_GROUP_LABELS[group]

    print(
        f"\n{'#' * 68}\n"
        f"# Experiment Group: {group_label}\n"
        f"# Variants: {', '.join(variants)}\n"
        f"{'#' * 68}"
    )

    development_results = []
    for preset_name in variants:
        development_results.append(run_variant(preset_name, dataset, group_label=group_label))

    selection = select_best_development_config(development_results, output_path=ranking_path)
    write_development_results_csv(development_results, selection, csv_path=csv_path)

    _print_comparison_table(development_results, group_label)
    _print_selection_summary(selection)
    return [result.as_dict() for result in development_results]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FBN Pipeline - development-only experiment runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--group",
        choices=["exp1", "exp2", "exp3", "all"],
        default=None,
    )
    parser.add_argument("--data-folder", dest="data_folder", default=None)
    parser.add_argument("--csv-path", dest="csv_path", default=None)
    parser.add_argument("--ranking-path", dest="ranking_path", default=None)
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = _parse_args()
    group = args.group if args.group is not None else EXPERIMENT_GROUP
    data_folder = args.data_folder if args.data_folder is not None else DATA_FOLDER
    csv_path = PROJECT_ROOT / (args.csv_path if args.csv_path is not None else CSV_PATH)
    ranking_path = PROJECT_ROOT / (
        args.ranking_path if args.ranking_path is not None else RANKING_PATH
    )

    print("=" * 72)
    print("FBN PIPELINE - DEVELOPMENT-ONLY EXPERIMENT RUNNER")
    print("=" * 72)
    print(f"Experiment group : {group.upper()}")
    print("Protocol mode    : DEVELOPMENT_ONLY")
    print(f"Project root     : {PROJECT_ROOT}")
    print(f"Data folder      : {PROJECT_ROOT / data_folder}")
    print(f"Development CSV  : {csv_path}")
    print(f"Ranking JSON     : {ranking_path}")
    print("=" * 72)

    dataset = load_dataset(PROJECT_ROOT / data_folder)
    train_uids = sorted(np.unique(dataset.user_train).tolist())
    print(
        f"Train: X={dataset.X_train.shape}, y={dataset.y_train.shape} "
        f"subjects={train_uids}"
    )

    groups_to_run = ["exp1", "exp2", "exp3"] if group == "all" else [group]
    all_results: dict[str, list[dict]] = {}
    for current_group in groups_to_run:
        group_csv = csv_path.with_name(f"{current_group}_{csv_path.name}")
        group_ranking = ranking_path.with_name(
            f"{current_group}_{ranking_path.name}"
        )
        all_results[current_group] = run_experiment_group(
            current_group,
            dataset,
            csv_path=group_csv,
            ranking_path=group_ranking,
        )

    if group == "all":
        print("\n" + "=" * 72)
        print("ALL DEVELOPMENT GROUPS COMPLETE")
        print("=" * 72)
        for current_group, results in all_results.items():
            print(f"\n{EXP_GROUP_LABELS[current_group]}")
            for result in results:
                oof = result["oof_metrics"]
                print(
                    f"  {result['variant']:<22s} "
                    f"BalAcc={oof['bal_acc']:.4f} "
                    f"MacroF1={oof['macro_f1']:.4f} "
                    f"F1={oof['f1']:.4f} "
                    f"AUC={oof['auc']:.4f}"
                )


if __name__ == "__main__":
    main()
