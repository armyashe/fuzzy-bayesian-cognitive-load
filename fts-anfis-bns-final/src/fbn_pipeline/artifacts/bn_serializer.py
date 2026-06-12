from __future__ import annotations

from typing import Any

import numpy as np

from ..models.bn_fusion import SIGNAL_NAMES
from ..models.fusion_protocol import FusionModel


def build_bn_details_artifact(
    bn: FusionModel,
    cfg: dict,
    exp_name: str,
    best_threshold: float,
    test_metrics: dict,
    test_probs: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    user_test: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable BN artifact without writing it to disk."""
    sample_trace = _build_sample_trace(bn, test_probs, y_test, user_test)
    n_signals = getattr(bn, "n_signals_", len(SIGNAL_NAMES))

    return {
        "experiment": exp_name,
        "configuration": cfg,
        "threshold": best_threshold,
        "test_metrics": test_metrics,
        "bn_structure": {
            "root": "CL",
            "root_states": ["rest", "task"],
            "evidence_nodes": [
                SIGNAL_NAMES[s] if s < len(SIGNAL_NAMES) else f"S{s}"
                for s in range(n_signals)
            ],
            "n_bins": bn.n_bins,
            "bin_names": bn.bin_names_,
            "soft_evidence": bool(getattr(bn, "soft", False)),
            "bin_mids": _serialize_bin_mids(bn, n_signals),
            "bin_sigmas": _serialize_bin_sigmas(bn, n_signals),
        },
        "prior": {
            "rest": float(bn.prior_[0]),
            "task": float(bn.prior_[1]),
        },
        "signal_weights": bn.get_signal_weights(),
        "cpt_tables": bn.get_cpt_table(),
        "sample_inference_trace": sample_trace,
    }


def _build_sample_trace(
    bn: FusionModel,
    test_probs: np.ndarray | None,
    y_test: np.ndarray | None,
    user_test: np.ndarray | None,
) -> dict[str, Any] | None:
    if test_probs is None or len(test_probs) == 0:
        return None

    if y_test is not None and np.sum(y_test == 1) > 0:
        sample_idx = int(np.where(y_test == 1)[0][0])
    else:
        sample_idx = 0

    sample_probs = test_probs[sample_idx]
    trace = bn.explain_sample(sample_probs)
    trace["sample_index"] = sample_idx
    if user_test is not None and sample_idx < len(user_test):
        trace["user_id"] = str(user_test[sample_idx])

    n_signals = getattr(bn, "n_signals_", len(sample_probs))
    trace["anfis_probs"] = {
        (SIGNAL_NAMES[s] if s < len(SIGNAL_NAMES) else f"S{s}"): round(float(sample_probs[s]), 8)
        for s in range(n_signals)
    }

    final_post = float(bn.predict_proba(sample_probs.reshape(1, -1))[0])
    trace["final_posterior_task"] = round(final_post, 8)
    trace["final_posterior_rest"] = round(1.0 - final_post, 8)
    return trace


def _serialize_bin_mids(bn: FusionModel, n_signals: int) -> dict[str, list[float]] | None:
    if not hasattr(bn, "bin_mids_"):
        return None
    return {
        (SIGNAL_NAMES[s] if s < len(SIGNAL_NAMES) else f"S{s}"): bn.bin_mids_[s].tolist()
        for s in range(n_signals)
    }


def _serialize_bin_sigmas(bn: FusionModel, n_signals: int) -> dict[str, float] | None:
    if not hasattr(bn, "bin_sigmas_"):
        return None
    return {
        (SIGNAL_NAMES[s] if s < len(SIGNAL_NAMES) else f"S{s}"): float(bn.bin_sigmas_[s])
        for s in range(n_signals)
    }
