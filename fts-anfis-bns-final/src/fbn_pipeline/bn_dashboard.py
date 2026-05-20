"""
Bayesian Network Export Module
===============================
Provides utilities to export BN artifacts for Streamlit dashboard:
  - JSON export of BN structure, CPT, prior, weight, sample inference trace
  - ANFIS rules export
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .models.bn_fusion import BNFusion, SIGNAL_NAMES
from .models.anfis_tsk import ANFIS_TSK


def export_bn_graph_image(
    bn: BNFusion,
    output_path: Path,
    dpi: int = 150,
) -> bool:
    """
    BN graph export skipped (pyAgrum removed).
    """
    return False


def export_bn_details_json(
    bn: BNFusion,
    cfg: dict,
    exp_name: str,
    best_threshold: float,
    test_metrics: dict,
    test_probs: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    user_test: np.ndarray | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """
    Export BN structure and metrics as JSON.

    Parameters
    ----------
    bn : BNFusion
        Fitted Bayesian Network.
    cfg : dict
        Configuration dict (from ExperimentConfig.to_dict()).
    exp_name : str
        Experiment name.
    best_threshold : float
        Chosen threshold.
    test_metrics : dict
        Test metrics (acc, bal_acc, macro_f1, f1, auc).
    test_probs : np.ndarray, optional
        Test evidence probabilities (for sample trace).
    y_test : np.ndarray, optional
        Test labels (for sample selection).
    output_path : Path, optional
        If provided, write JSON to this file.

    Returns
    -------
    dict
        JSON-serializable dict with BN details.
    """
    # Select a positive sample for trace if available
    sample_trace = None
    if test_probs is not None and len(test_probs) > 0:
        # Pick first sample of class "task" if available, else first sample
        if y_test is not None and np.sum(y_test == 1) > 0:
            pos_idx = np.where(y_test == 1)[0][0]
        else:
            pos_idx = 0
        
        sample_probs = test_probs[pos_idx]
        sample_trace = bn.explain_sample(sample_probs)
        sample_trace["sample_index"] = int(pos_idx)
        if user_test is not None and pos_idx < len(user_test):
            sample_trace["user_id"] = str(user_test[pos_idx])
        # Store raw ANFIS probs so the notebook can recompute soft posterior
        # without needing to re-run training.
        sample_trace["anfis_probs"] = {
            (SIGNAL_NAMES[s] if s < len(SIGNAL_NAMES) else f"S{s}"): round(float(sample_probs[s]), 8)
            for s in range(bn.n_signals_)
        }
        # final_posterior_task uses predict_proba (soft evidence when bn_soft=True),
        # matching the value shown in Streamlit — differs from posterior_task above
        # which is the hard-bin explanatory trace posterior.
        final_post = float(bn.predict_proba(sample_probs.reshape(1, -1))[0])
        sample_trace["final_posterior_task"] = round(final_post, 8)
        sample_trace["final_posterior_rest"] = round(1.0 - final_post, 8)

    # CPT tables
    cpt_tables = bn.get_cpt_table()
    cpt_dict = {}
    for sig_name, table in cpt_tables.items():
        cpt_dict[sig_name] = table

    # Signal weights
    signal_weights = bn.get_signal_weights()

    # Build full JSON
    details = {
        "experiment": exp_name,
        "configuration": cfg,
        "threshold": best_threshold,
        "test_metrics": test_metrics,
        "bn_structure": {
            "root": "CL",
            "root_states": ["rest", "task"],
            "evidence_nodes": [
                SIGNAL_NAMES[s] if s < len(SIGNAL_NAMES) else f"S{s}"
                for s in range(bn.n_signals_)
            ],
            "n_bins": bn.n_bins,
            "bin_names": bn.bin_names_,
            # bin_mids and bin_sigmas allow notebooks to reproduce soft-evidence
            # posteriors from JSON alone (no model object needed).
            "bin_mids": {
                (SIGNAL_NAMES[s] if s < len(SIGNAL_NAMES) else f"S{s}"): bn.bin_mids_[s].tolist()
                for s in range(bn.n_signals_)
            } if hasattr(bn, "bin_mids_") else None,
            "bin_sigmas": {
                (SIGNAL_NAMES[s] if s < len(SIGNAL_NAMES) else f"S{s}"): float(bn.bin_sigmas_[s])
                for s in range(bn.n_signals_)
            } if hasattr(bn, "bin_sigmas_") else None,
        },
        "prior": {
            "rest": float(bn.prior_[0]),
            "task": float(bn.prior_[1]),
        },
        "signal_weights": signal_weights,
        "cpt_tables": cpt_dict,
        "sample_inference_trace": sample_trace,
    }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(details, f, indent=2, ensure_ascii=False)

    return details


def export_anfis_rules_json(
    anfis_list: list[ANFIS_TSK] | None,
    n_sets: int,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """
    Export ANFIS rules from all signals as JSON.

    Parameters
    ----------
    anfis_list : list[ANFIS_TSK] or None
        List of fitted ANFIS models, one per signal.
    n_sets : int
        Number of fuzzy sets (for feature naming).
    output_path : Path, optional
        If provided, write JSON to this file.

    Returns
    -------
    dict
        JSON-serializable dict with rules per signal.
    """
    rules_dict: dict[str, list[str]] = {}

    if anfis_list is None:
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(rules_dict, f, indent=2)
        return rules_dict

    # Generate feature names based on n_sets
    if n_sets == 3:
        labels = ["L", "M", "H"]
    elif n_sets == 4:
        labels = ["VL", "L", "M", "H"]
    elif n_sets == 5:
        labels = ["VL", "L", "M", "H", "VH"]
    else:
        labels = [f"S{k}" for k in range(n_sets)]

    feature_names = [f"P({to_}|{from_})" for from_ in labels for to_ in labels]

    for s, sig_name in enumerate(SIGNAL_NAMES if len(anfis_list) == len(SIGNAL_NAMES) else [f"S{s}" for s in range(len(anfis_list))]):
        if s < len(anfis_list):
            anfis = anfis_list[s]
            rules = anfis.rule_strings_full(
                feature_names=feature_names,
            )
            rules_dict[sig_name] = rules

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rules_dict, f, indent=2, ensure_ascii=False)

    return rules_dict


def export_bn_dashboard_artifacts(
    models: dict[str, Any],
    cfg: dict,
    exp_name: str,
    best_threshold: float,
    test_metrics: dict,
    test_probs: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    user_test: np.ndarray | None = None,
    artifacts_dir: Path | str = "artifacts",
) -> dict[str, Path | None]:
    """
    Export all BN dashboard artifacts.

    Parameters
    ----------
    models : dict
        Dict from retrain_and_test(): {"fts": ..., "bn": ..., "anfis_list": ...}
    cfg : dict
        Configuration dict.
    exp_name : str
        Experiment name.
    best_threshold : float
        Best threshold selected.
    test_metrics : dict
        Test metrics.
    test_probs : np.ndarray, optional
        Test evidence probabilities.
    y_test : np.ndarray, optional
        Test labels.
    artifacts_dir : Path or str
        Root directory for artifacts. Will create:
          {artifacts_dir}/{exp_name}/bn_graph.png
          {artifacts_dir}/{exp_name}/bn_details.json
          {artifacts_dir}/{exp_name}/anfis_rules.json

    Returns
    -------
    dict[str, Path or None]
        Dict with keys: "bn_graph", "bn_details", "anfis_rules"
        Values are file paths if successful, None otherwise.
    """
    artifacts_dir = Path(artifacts_dir)
    exp_dir = artifacts_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "bn_graph": None,
        "bn_details": None,
        "anfis_rules": None,
        "exp_dir": exp_dir,
    }

    bn = models.get("bn")
    anfis_list = models.get("anfis_list")

    try:
        # Export BN graph
        if bn is not None:
            graph_path = exp_dir / "bn_graph.png"
            success = export_bn_graph_image(bn, graph_path)
            if success:
                results["bn_graph"] = graph_path
                print(f"[OK] Exported BN graph: {graph_path}")
            else:
                print(f"[WARN] BN graph export failed (pyAgrum or graphviz may not be installed)")
        else:
            print(f"[WARN] No BN model (likely exp2: fusion without BN)")

        # Export BN details JSON
        if bn is not None:
            details_path = exp_dir / "bn_details.json"
            export_bn_details_json(
                bn,
                cfg,
                exp_name,
                best_threshold,
                test_metrics,
                test_probs=test_probs,
                y_test=y_test,
                user_test=user_test,
                output_path=details_path,
            )
            results["bn_details"] = details_path
            print(f"[OK] Exported BN details: {details_path}")

        # Export ANFIS rules JSON
        if anfis_list is not None:
            rules_path = exp_dir / "anfis_rules.json"
            export_anfis_rules_json(
                anfis_list,
                n_sets=cfg.get("n_sets", 3),
                output_path=rules_path,
            )
            results["anfis_rules"] = rules_path
            print(f"[OK] Exported ANFIS rules: {rules_path}")
        else:
            print(f"[WARN] No ANFIS models (likely exp4: FTS summary)")

    except Exception as e:
        print(f"[ERROR] Error during artifact export: {e}")
        import traceback
        traceback.print_exc()

    return results
