from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# ============================================================
# USER SETTINGS - edit here when running as a notebook / plain script.
# CLI arguments (--ablation, --preset, etc.) override these values.
# ============================================================
MODE = "single"                    # "single"
PRESET_NAME = "THESIS_FINAL"       # None | "THESIS_FINAL" | "A1" | "A2" | ...
ABLATION = "exp1"                 # "exp1" | "exp2" | "exp3" | "exp4"
EXP_NAME = "main_run"
DATA_FOLDER = "last_30s_segments"
CSV_PATH = "artifacts/A2_soft_results/results_main.csv"
SAVE_CSV = True
SHOW_PER_SUBJECT = True
DUMP_RULES = True
FUSION_MODE = "mean"              # "mean" | "weighted_mean"  (exp2 only)
FTS_SUMMARY_MODE = "diag_high"    # "diag"|"high"|"diag_high"|"entropy" (exp3 only)
NO_STAGE1_FEATURE_MODE = "stats"  # "stats"                   (exp4 only)


OVERRIDES: dict[str, Any] = {}


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

from fbn_pipeline import ABLATION_CHOICES, ExperimentConfig, apply_preset, assert_thesis_config
from fbn_pipeline.data import load_dataset
from fbn_pipeline.runner import run_one_experiment


# ============================================================
# CLI argument parser
# ============================================================

def _parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments.  All arguments are optional; defaults fall through
    to the script-level constants above.  Compatible with IPython / Jupyter
    (unknown kernel args are silently ignored via parse_known_args).
    """
    parser = argparse.ArgumentParser(
        description="FBN Pipeline - ablation experiment runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--ablation",
        choices=ABLATION_CHOICES,
        default=None,
        help=(
            "Ablation experiment to run:\n"
            "  exp1 = Stage1 FTS + Stage2 ANFIS + Stage3 BN  (full pipeline)\n"
            "  exp2 = Stage1 FTS + Stage2 ANFIS               (no BN)\n"
            "  exp3 = Stage1 FTS + Stage3 BN                  (no ANFIS, FTS summary)\n"
            "  exp4 = Stage2 ANFIS + Stage3 BN                (no FTS, stat features)"
        ),
    )
    parser.add_argument(
        "--preset",
        default=None,
        help="Config preset name (e.g. A2_soft, A1, A2_plus, ...).",
    )
    parser.add_argument(
        "--data-folder",
        dest="data_folder",
        default=None,
        help="Path to data folder (relative to project root or absolute).",
    )
    parser.add_argument(
        "--csv-path",
        dest="csv_path",
        default=None,
        help="Path to output CSV file.",
    )
    parser.add_argument(
        "--save-csv",
        dest="save_csv",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-save-csv",
        dest="save_csv",
        action="store_false",
    )
    parser.add_argument(
        "--show-per-subject",
        dest="show_per_subject",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--dump-rules",
        dest="dump_rules",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--fusion",
        dest="fusion_mode",
        choices=["mean", "weighted_mean"],
        default=None,
        help="(exp2 only) Fusion strategy when BN is disabled.",
    )
    parser.add_argument(
        "--fts-summary-mode",
        dest="fts_summary_mode",
        choices=["diag", "high", "diag_high", "entropy"],
        default=None,
        help="(exp3 only) How to summarise FTS features into per-signal evidence.",
    )
    parser.add_argument(
        "--anfis-mf-type",
        dest="anfis_mf_type",
        choices=["gaussian", "gbell"],
        default=None,
        help="MF type for ANFIS premise layer. Overrides the FTS mf_type for ANFIS only.\n"
             "  gaussian = standard Gaussian (default, same as before)\n"
             "  gbell    = generalized bell 1/(1+|(x-c)/\u03c3|^{2b})",
    )
    parser.add_argument(
        "--no-stage1-feature-mode",
        dest="no_stage1_feature_mode",
        choices=["stats"],
        default=None,
        help="(exp4 only) Feature extractor replacing Stage 1.",
    )
    parser.add_argument(
        "--explain-sample-index",
        dest="explain_sample_index",
        type=int,
        default=None,
        help="Print inference trace for a specific test sample index.",
    )
    parser.add_argument(
        "--explain-user-id",
        dest="explain_user_id",
        default=None,
        help="Print inference trace for a test sample belonging to this user_id.",
    )
    parser.add_argument(
        "--explain-user-k",
        dest="explain_user_k",
        type=int,
        default=0,
        help="When --explain-user-id is used, select the k-th sample of that user.",
    )
    parser.add_argument(
        "--print-bn-details",
        dest="print_bn_details",
        action="store_true",
        default=False,
        help="Print BN prior, signal weights, and CPT tables after training.",
    )

    # parse_known_args silently drops Jupyter / IPython kernel flags
    args, _ = parser.parse_known_args()
    return args


# ============================================================
# Config builder
# ============================================================

def build_config(args: argparse.Namespace | None = None) -> ExperimentConfig:
    """
    Build ExperimentConfig by merging preset -> script constants -> CLI args.

    Priority (high -> low): CLI args > script-level constants > preset > defaults.
    """
    # Resolve effective values from constants (may be overridden by args below)
    preset_name = PRESET_NAME
    ablation = ABLATION
    fusion_mode = FUSION_MODE
    no_stage1_feature_mode = NO_STAGE1_FEATURE_MODE
    fts_summary_mode = FTS_SUMMARY_MODE
    dump_rules = DUMP_RULES

    if args is not None:
        if args.preset is not None:
            preset_name = args.preset
        if args.ablation is not None:
            ablation = args.ablation
        if args.fusion_mode is not None:
            fusion_mode = args.fusion_mode
        if args.no_stage1_feature_mode is not None:
            no_stage1_feature_mode = args.no_stage1_feature_mode
        if args.fts_summary_mode is not None:
            fts_summary_mode = args.fts_summary_mode
        if args.dump_rules:
            dump_rules = True

    anfis_mf_type_override = getattr(args, "anfis_mf_type", None) if args is not None else None

    cfg = ExperimentConfig(data_path=DATA_FOLDER)
    cfg = apply_preset(cfg, preset_name)

    if OVERRIDES:
        cfg = cfg.merged(**OVERRIDES)

    cfg = cfg.merged(
        ablation=ablation,
        fusion_mode_when_no_bn=fusion_mode,
        no_stage1_feature_mode=no_stage1_feature_mode,
        fts_summary_mode=fts_summary_mode,
        dump_rules=dump_rules,
        **(({"anfis_mf_type": anfis_mf_type_override}) if anfis_mf_type_override else {}),
    )
    if preset_name == "THESIS_FINAL":
        assert_thesis_config(cfg)
    return cfg


# ============================================================
# Main runner
# ============================================================

def run_single(args: argparse.Namespace | None = None) -> dict[str, Any]:
    cfg = build_config(args)

    # Resolve runtime settings from args or script constants
    data_folder = DATA_FOLDER
    csv_path_str = CSV_PATH
    save_csv = SAVE_CSV
    show_per_subject = SHOW_PER_SUBJECT
    preset_name = PRESET_NAME
    exp_name = EXP_NAME

    if args is not None:
        if args.data_folder is not None:
            data_folder = args.data_folder
        if args.csv_path is not None:
            csv_path_str = args.csv_path
        if args.save_csv is not None:
            save_csv = args.save_csv
        if args.show_per_subject is not None:
            show_per_subject = args.show_per_subject
        if args.preset is not None:
            preset_name = args.preset

    #  ---- build unified run/artifact name ----
    preset_part = preset_name or "no_preset"
    ablation_part = getattr(cfg, "ablation", None) or "main"
    run_name = f"{preset_part}_{ablation_part}"
    
    # Keep the displayed experiment name aligned with preset + ablation.
    exp_name = run_name

    # Store all dashboard artifacts for this run.
    artifacts_dir = PROJECT_ROOT / "artifacts" / run_name
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    data_path = PROJECT_ROOT / data_folder
    dataset = load_dataset(data_path)

    print("=" * 72)
    print("FBN PIPELINE - MAIN RUNNER")
    print("=" * 72)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data folder  : {data_path}")
    print(f"Preset       : {PRESET_NAME if (args is None or args.preset is None) else args.preset}")
    print(f"Ablation     : {cfg.ablation}")
    print(f"Experiment   : {exp_name}")
    print(f"Artifacts dir : {artifacts_dir}")
    print("=" * 72)
    import numpy as _np
    train_uids = sorted(_np.unique(dataset.user_train).tolist())
    test_uids  = sorted(_np.unique(dataset.user_test).tolist())
    print(
        f"Train: X={dataset.X_train.shape}, y={dataset.y_train.shape}, "
        f"users={dataset.user_train.shape}  user_ids={train_uids}"
    )
    print(
        f"Test : X={dataset.X_test.shape}, y={dataset.y_test.shape}, "
        f"users={dataset.user_test.shape}  user_ids={test_uids}"
    )
    result = run_one_experiment(
        cfg,
        dataset,
        exp_name=exp_name,
        preset_name=preset_name or "",
        run_name=run_name,
        csv_path=(PROJECT_ROOT / csv_path_str) if save_csv else None,
        show_per_subject=show_per_subject,
        dump_rules=cfg.dump_rules,
        export_artifacts=True,
        artifacts_dir=artifacts_dir.parent,
        print_configuration=True,
        print_progress=True,
    )
    return result.as_dict()



def explain_test_sample(i: int) -> dict[str, Any]:
    """
    Explain one test sample by index.

    Usage after running main.py:
        explain_test_sample(0)
        explain_test_sample(10)
    """
    dataset = RUN_OUTPUT["dataset"]
    models = RUN_OUTPUT["models"]

    bn = models["bn"]
    te_probs = models["te_probs"]

    trace = bn.explain_sample(te_probs[i])

    true_label = "task" if dataset.y_test[i] == 1 else "rest"
    pred_label = "task" if RUN_OUTPUT["test_pred"][i] == 1 else "rest"

    result = {
        "sample_index": int(i),
        "user_id": str(dataset.user_test[i]),
        "true_label": true_label,
        "pred_label": pred_label,
        "decision_threshold": float(RUN_OUTPUT["best_threshold"]),
        "final_posterior_task_from_model": float(RUN_OUTPUT["test_post"][i]),
        "anfis_probs": {
            "Temp": float(te_probs[i, 0]),
            "HR": float(te_probs[i, 1]),
            "GSR": float(te_probs[i, 2]),
            "RR": float(te_probs[i, 3]),
        },
        "sample_inference_trace": trace,
        "cpt_bins": bn.get_cpt_table(),
    }

    print("=" * 72)
    print(f"SAMPLE INFERENCE TRACE | index={i} | user_id={result['user_id']}")
    print("=" * 72)
    print(f"True label      : {true_label}")
    print(f"Pred label      : {pred_label}")
    print(f"Threshold       : {result['decision_threshold']:.4f}")
    print(f"Final P(task)   : {result['final_posterior_task_from_model']:.4f}")

    print("\nANFIS probabilities:")
    for sig, p in result["anfis_probs"].items():
        print(f"  {sig:<5s}: {p:.4f}")

    mode = trace.get("mode", "hard")
    if mode == "soft":
        print("\nDominant bins (visualization only - not actual soft evidence):")
        for sig, b in trace.get("dominant_bins", trace.get("evidence_bins", {})).items():
            print(f"  {sig:<5s}: {b}")

        sm = trace.get("soft_memberships", {})
        if sm:
            print("\nSoft memberships:")
            for sig, bins in sm.items():
                parts = ", ".join(f"{k}={v:.4f}" for k, v in bins.items())
                print(f"  {sig:<5s}: {parts}")

        llr = trace.get("log_likelihood_ratio", {})
        if llr:
            print("\nLog-likelihood ratio task/rest (+ supports TASK, - supports REST):")
            for sig, val in llr.items():
                print(f"  {sig:<5s}: {val:+.4f}")

        print("\nBN posterior from explain_sample (soft evidence):")
        print(f"  P(task | soft evidence): {trace['posterior_task']:.4f}")
        print(f"  P(rest | soft evidence): {trace['posterior_rest']:.4f}")

    else:
        print("\nBN evidence bins:")
        for sig, b in trace["evidence_bins"].items():
            print(f"  {sig:<5s}: {b}")

        print("\nBN posterior from explain_sample:")
        print(f"  P(task | evidence): {trace['posterior_task']:.4f}")
        print(f"  P(rest | evidence): {trace['posterior_rest']:.4f}")

        print("\nLog-likelihood contributions:")
        for cls_id, contrib in trace["log_likelihoods"].items():
            cls_name = "rest" if int(cls_id) == 0 else "task"
            print(f"  Class = {cls_name}")
            for sig, val in contrib.items():
                print(f"    {sig:<5s}: {val:.4f}")

    return result

def explain_sample_of_user(user_id: str, k: int = 0) -> dict:
    """
    Explain the k-th test sample of a given user.

    Example:
        explain_sample_of_user("3caqi", 0)
        explain_sample_of_user("3caqi", 5)
        explain_sample_of_user("6frz4", 0)
    """
    import numpy as np

    users = RUN_OUTPUT["dataset"].user_test
    idx = np.where(users == user_id)[0]

    if len(idx) == 0:
        raise ValueError(f"user_id={user_id} not found in test set.")

    if k < 0 or k >= len(idx):
        raise IndexError(
            f"user_id={user_id} has {len(idx)} samples, but k={k} was requested."
        )

    return explain_test_sample(int(idx[k]))

def print_bn_details() -> dict[str, Any]:
    """
    Print Bayesian Network details: structure, prior, signal weights, CPT.
    """
    import json

    bn = RUN_OUTPUT["models"]["bn"]
    cfg = RUN_OUTPUT["config"]

    if bn is None:
        raise ValueError("No BN model found. This ablation may not use BN.")

    details = {
        "bn_structure": {
            "root": "CL",
            "root_states": ["rest", "task"],
            "evidence_nodes": ["Temp", "HR", "GSR", "RR"],
            "n_bins": int(bn.n_bins),
            "bin_names": list(bn.bin_names_),
            "soft_evidence": bool(getattr(bn, "soft", False)),
            "structure_type": getattr(cfg, "bn_structure", "naive"),
        },
        "prior": {
            "rest": round(float(bn.prior_[0]), 4),
            "task": round(float(bn.prior_[1]), 4),
        },
        "signal_weights": bn.get_signal_weights(),
        "cpt_tables": bn.get_cpt_table(),
    }

    print("=" * 72)
    print("BAYESIAN NETWORK DETAILS")
    print("=" * 72)
    print(json.dumps(details, indent=2, ensure_ascii=False))

    return details


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    _args = _parse_args()
    if MODE != "single":
        raise ValueError("This main.py only supports MODE='single'.")
    RUN_OUTPUT = run_single(_args)
else:
    # Notebook / %run context: parse args but tolerate empty argv
    _args = _parse_args()
    if MODE != "single":
        raise ValueError("This main.py only supports MODE='single'.")
    RUN_OUTPUT = run_single(_args)
    
# ============================================================
# Optional CLI-driven explainability
# ============================================================

if getattr(_args, "print_bn_details", False):
    print_bn_details()

if getattr(_args, "explain_sample_index", None) is not None:
    explain_test_sample(_args.explain_sample_index)

if getattr(_args, "explain_user_id", None) is not None:
    explain_sample_of_user(
        user_id=_args.explain_user_id,
        k=_args.explain_user_k,
    )
