import numpy as np
import pandas as pd

from fbn_pipeline.models.bn_fusion import SIGNAL_NAMES
from streamlit_app.components.soft_evidence import compute_soft_membership_table
from streamlit_app.utils import build_raw_df, get_state_names, get_transition_feature_names


def extract_sample_stage_outputs(results: dict, sample_idx: int) -> dict:
    dataset = results["dataset"]
    models = results["models"]

    x_sample = dataset.X_test[sample_idx]
    y_true = int(dataset.y_test[sample_idx])
    y_pred = int(results["test_pred"][sample_idx])
    user_id = dataset.user_test[sample_idx]

    fts = models["fts"]
    bn = models["bn"]
    anfis_list = models["anfis_list"]
    test_evidence = models.get("te_probs")

    raw_df = build_raw_df(x_sample)

    f_sample = fts.transform(x_sample[np.newaxis, ...])[0]
    feat_per_sig = len(f_sample) // len(SIGNAL_NAMES)
    feature_names = get_transition_feature_names(fts.n_sets)
    state_names = get_state_names(fts.n_sets)

    transition_tables = {}
    fts_feature_tables = {}

    for s, signal_name in enumerate(SIGNAL_NAMES):
        start = s * feat_per_sig
        end = start + feat_per_sig
        f_sig = f_sample[start:end]
        trans = fts.get_transition_matrix(x_sample, s)

        transition_tables[signal_name] = pd.DataFrame(
            trans,
            index=state_names,
            columns=state_names,
        )

        fts_feature_tables[signal_name] = pd.DataFrame(
            {"feature": feature_names, "value": np.round(f_sig, 4)}
        )

    canonical_probs = None
    if test_evidence is not None and sample_idx < len(test_evidence):
        canonical_probs = np.asarray(test_evidence[sample_idx], dtype=float)

    signal_prob_rows = []
    firing_tables = {}
    top_rule_tables = {}

    te_probs_manual = []

    for s, signal_name in enumerate(SIGNAL_NAMES):
        start = s * feat_per_sig
        end = start + feat_per_sig
        f_sig = f_sample[start:end].reshape(1, -1)

        anfis = anfis_list[s]
        prob_s = float(anfis.predict_proba(f_sig)[0])
        firing = anfis._normalised_firing(f_sig)[0]
        te_probs_manual.append(prob_s)
        display_prob_s = (
            float(canonical_probs[s])
            if canonical_probs is not None and s < len(canonical_probs)
            else prob_s
        )

        signal_prob_rows.append(
            {
                "signal": signal_name,
                "anfis_p_task": round(display_prob_s, 4),
                "raw_model_p_task": round(prob_s, 4),
            }
        )

        firing_tables[signal_name] = pd.DataFrame(
            {
                "rule": [f"R{i+1}" for i in range(len(firing))],
                "firing_strength": np.round(firing, 4),
            }
        )

        rules = anfis.rule_strings(
            feature_names=feature_names,
            top_k=min(4, len(feature_names)),
        )
        top_idx = np.argsort(firing)[::-1][:3]
        top_rows = []
        for ridx in top_idx:
            top_rows.append(
                {
                    "rule": f"R{ridx+1}",
                    "firing_strength": round(float(firing[ridx]), 4),
                    "text": rules[ridx],
                }
            )
        top_rule_tables[signal_name] = pd.DataFrame(top_rows)

    te_probs_manual = np.array(te_probs_manual)
    bn_input_probs = canonical_probs if canonical_probs is not None else te_probs_manual
    posterior_from_bn = float(bn.predict_proba(bn_input_probs.reshape(1, -1))[0])
    posterior_task = (
        float(results["test_post"][sample_idx])
        if "test_post" in results and sample_idx < len(results["test_post"])
        else posterior_from_bn
    )
    bn_trace = bn.explain_sample(bn_input_probs)
    bn_trace["batch_posterior_task"] = posterior_task
    bn_trace["trace_matches_batch"] = bool(np.isclose(bn_trace["posterior_task"], posterior_task, atol=1e-8))
    soft_df = compute_soft_membership_table(bn, bn_input_probs)

    return {
        "sample_idx": sample_idx,
        "user_id": user_id,
        "y_true": y_true,
        "y_pred": y_pred,
        "threshold": float(results["best_thr"]),
        "posterior_task": posterior_task,
        "posterior_from_bn": posterior_from_bn,
        "evidence_mode": bn_trace.get("evidence_mode", bn_trace.get("mode", "hard")),
        "raw_df": raw_df,
        "transition_tables": transition_tables,
        "fts_feature_tables": fts_feature_tables,
        "signal_prob_df": pd.DataFrame(signal_prob_rows),
        "bn_input_probs": bn_input_probs,
        "manual_anfis_probs": te_probs_manual,
        "firing_tables": firing_tables,
        "top_rule_tables": top_rule_tables,
        "bn_trace": bn_trace,
        "soft_membership_df": soft_df,
    }
