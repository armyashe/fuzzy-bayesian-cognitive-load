from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from fbn_pipeline.models.bn_fusion import SIGNAL_NAMES
from streamlit_app.components.tables import (
    compute_soft_influence_table,
    cpt_dict_to_df,
    signal_weights_df,
)
from streamlit_app.services.artifacts import (
    get_bn_graph_image_path,
    load_anfis_rules_json,
    load_bn_artifact_json,
)
from streamlit_app.utils import get_transition_feature_names, signal_weights_are_uniform
def render_bn_rules_tab(results: dict, root: Path) -> None:
    cfg = results["cfg"]
    models = results["models"]
    bn = models.get("bn")
    anfis_list = models.get("anfis_list")
    has_bn = bn is not None
    has_anfis = anfis_list is not None

    st.subheader("Model rules and Bayesian CPTs")

    exp_name = results.get("exp_name", "main_run")

    bn_details = load_bn_artifact_json(exp_name, artifacts_root=root / "artifacts")
    anfis_rules_data = load_anfis_rules_json(exp_name, artifacts_root=root / "artifacts")
    bn_graph_path = get_bn_graph_image_path(exp_name, artifacts_root=root / "artifacts")

    render_bn_section(bn, has_bn, bn_details, bn_graph_path)
    render_anfis_rules_section(cfg, anfis_list, has_anfis, anfis_rules_data)
    render_missing_artifacts_warning(exp_name, bn_details, anfis_rules_data, bn_graph_path)


def render_bn_section(bn, has_bn: bool, bn_details: dict | None, bn_graph_path: Path | None) -> None:

    # =========================
    # BN section
    # =========================
    st.markdown("## Bayesian Network")

    if not has_bn:
        st.info("Experiment này không dùng Bayesian Network nên không có BN graph, prior, CPT hoặc inference trace.")
    else:
        if bn_graph_path is not None:
            st.markdown("**Bayesian Network Structure**")
            st.image(str(bn_graph_path), use_container_width=True)

        st.markdown("**BN Prior**")
        if bn_details is not None and "prior" in bn_details:
            prior_df = pd.DataFrame(
                [
                    {"class": "rest", "prior": round(bn_details["prior"].get("rest", 0.0), 4)},
                    {"class": "task", "prior": round(bn_details["prior"].get("task", 0.0), 4)},
                ]
            )
        else:
            prior_df = pd.DataFrame(
                [
                    {"class": "rest", "prior": round(float(bn.prior_[0]), 4)},
                    {"class": "task", "prior": round(float(bn.prior_[1]), 4)},
                ]
            )
        st.dataframe(prior_df, use_container_width=True, hide_index=True)

        if bn_details is not None and "signal_weights" in bn_details:
            weights_df = signal_weights_df(bn_details["signal_weights"])
        else:
            weights_df = signal_weights_df(bn.get_signal_weights())
        weights_dict = dict(zip(weights_df["signal"], weights_df["weight"]))
        if signal_weights_are_uniform(weights_dict):
            with st.expander("Debug: uniform BN signal weights", expanded=False):
                st.caption(
                    "Uniform weights are shown for completeness; they do not change the relative signal weighting."
                )
                st.dataframe(weights_df, use_container_width=True, hide_index=True)
        else:
            st.markdown("**BN Signal Weights Used in Soft Evidence Fusion**")
            st.dataframe(weights_df, use_container_width=True, hide_index=True)

        st.markdown("**BN CPT Tables**")
        cpt_tables = bn.get_cpt_table()
        cols = st.columns(4)
        for i, sig in enumerate(SIGNAL_NAMES):
            with cols[i]:
                st.markdown(f"**{sig}**")
                st.dataframe(cpt_dict_to_df(cpt_tables[sig]), use_container_width=True)

        if bn_details is not None and "sample_inference_trace" in bn_details:
            trace = bn_details["sample_inference_trace"]
            if trace is not None:
                with st.expander("Sample Inference Trace", expanded=False):
                    if trace.get("soft_memberships"):
                        st.markdown("**Soft evidence membership**")
                        st.dataframe(
                            pd.DataFrame(
                                [
                                    {
                                        "signal": signal,
                                        "p_task_from_ANFIS": round(
                                            float(trace.get("anfis_probs", {}).get(signal, np.nan)),
                                            4,
                                        ),
                                        **{
                                            level: round(float(value), 4)
                                            for level, value in memberships.items()
                                        },
                                    }
                                    for signal, memberships in trace["soft_memberships"].items()
                                ]
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )
                    elif trace.get("evidence_bins"):
                        with st.expander("Debug: hard-bin approximation", expanded=False):
                            st.caption(
                                "This is only a debug approximation and is not the inference mechanism "
                                "used in the main soft-evidence experiment."
                            )
                            st.json(trace.get("evidence_bins", {}))

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("P(task | soft evidence)", round(trace.get("posterior_task", 0.0), 4))
                    with col2:
                        st.metric("P(rest | soft evidence)", round(trace.get("posterior_rest", 0.0), 4))

                    influence_df = compute_soft_influence_table(trace)
                    if influence_df is not None:
                        st.markdown("**Soft evidence influence by class**")
                        st.dataframe(influence_df, use_container_width=True, hide_index=True)

    # =========================
    # ANFIS section
    # =========================

def render_anfis_rules_section(cfg, anfis_list, has_anfis: bool, anfis_rules_data: dict | None) -> None:
    st.markdown("## ANFIS Rule Bases")

    if not has_anfis:
        st.info("Experiment này không dùng ANFIS nên không có ANFIS rule base.")
    else:
        if anfis_rules_data is not None:
            for sig_name, rules in anfis_rules_data.items():
                with st.expander(f"{sig_name} rules", expanded=False):
                    for rule_text in rules:
                        st.caption(rule_text)
        else:
            feature_names = get_transition_feature_names(cfg.n_sets)

            for sig, anfis in zip(SIGNAL_NAMES, anfis_list):
                with st.expander(f"{sig} rules", expanded=False):
                    rules = anfis.rule_strings(
                        feature_names=feature_names,
                        top_k=min(4, len(feature_names)),
                    )
                    rule_df = pd.DataFrame(
                        [{"rule_id": f"R{i+1}", "text": rule} for i, rule in enumerate(rules)]
                    )
                    st.dataframe(rule_df, use_container_width=True, hide_index=True)

def render_missing_artifacts_warning(
    exp_name: str,
    bn_details: dict | None,
    anfis_rules_data: dict | None,
    bn_graph_path: Path | None,
) -> None:
    if bn_details is None and anfis_rules_data is None and bn_graph_path is None:
        st.warning(
            "⚠️ **No artifacts found.** Run the model first to generate graph, rules, and metrics.\n"
            f"Artifacts should be saved in: `artifacts/{exp_name}/`"
        )
