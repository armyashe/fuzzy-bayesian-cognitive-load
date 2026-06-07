"""
Bayesian Network Export Module
===============================
Provides utilities to export BN artifacts for Streamlit dashboard:
  - JSON export of BN structure, CPT, prior, weight, sample inference trace
  - ANFIS rules export
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts.anfis_serializer import build_anfis_rules_artifact
from .artifacts.bn_serializer import build_bn_details_artifact
from .models.anfis_tsk import ANFIS_TSK
from .models.fusion_protocol import FusionModel


def export_bn_graph_image(
    bn: FusionModel,
    output_path: Path,
    dpi: int = 150,
) -> bool:
    """
    BN graph export skipped (pyAgrum removed).
    """
    return False


def export_bn_details_json(
    bn: FusionModel,
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
    details = build_bn_details_artifact(
        bn,
        cfg,
        exp_name,
        best_threshold,
        test_metrics,
        test_probs=test_probs,
        y_test=y_test,
        user_test=user_test,
    )

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
    rules_dict = build_anfis_rules_artifact(anfis_list, n_sets)

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
            print(f"[WARN] No ANFIS models (likely exp3: FTS summary)")

    except Exception as e:
        print(f"[ERROR] Error during artifact export: {e}")
        import traceback
        traceback.print_exc()

    return results
