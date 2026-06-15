import time

import numpy as np
import pandas as pd
import streamlit as st

from fbn_pipeline.models.bn_fusion import SIGNAL_NAMES
from streamlit_app.components.charts import plot_rule_firing, plot_signal_window
from streamlit_app.components.soft_evidence import render_soft_membership_heatmap
from streamlit_app.components.tables import (
    compute_soft_influence_table,
    signal_weights_df,
)
from streamlit_app.services.sample_explainer import extract_sample_stage_outputs
from streamlit_app.utils import signal_weights_are_uniform


def render_sample_explanation_tab(results: dict) -> None:
    dataset = results["dataset"]
    models = results["models"]
    selection = results["selection"]
    has_fts = models.get("fts") is not None
    has_bn = models.get("bn") is not None
    has_anfis = models.get("anfis_list") is not None

    st.subheader("Sample explanation")
    if not selection.supports_stage_demo or not (has_fts and has_anfis and has_bn):
        st.warning(
            "Detailed stage-by-stage explanation is available only when the current variant keeps the full FTS → ANFIS → BN pipeline."
        )
        return

    sample_idx = render_sample_selector(dataset)

    run_sample_btn = st.button(
        "Run selected sample step-by-step",
        use_container_width=True,
    )
    if not run_sample_btn:
        st.info("Chọn sample index rồi bấm 'Run selected sample step-by-step'.")
        return

    sample = run_stage_progress(results, sample_idx)
    render_sample_summary(sample)
    render_stage1_raw_window(sample)
    render_stage2_fts(sample)
    render_stage3_anfis(sample)
    render_stage4_bn(sample)


def render_sample_selector(dataset) -> int:
    st.caption("Các stage bên dưới dùng đúng mô hình đã fit trên toàn bộ tập train trong lần chạy hiện tại.")

           # Chọn user trước, sau đó chọn sample index cục bộ trong user đó
    test_users = sorted(pd.unique(dataset.user_test))

    selected_user = st.selectbox(
        "Test user",
        test_users,
        index=0,
        key="stage_demo_user",
    )

    user_indices = np.where(dataset.user_test == selected_user)[0]

    local_sample_idx = st.number_input(
        f"Sample index within user {selected_user}",
        min_value=0,
        max_value=len(user_indices) - 1,
        value=0,
        step=1,
        key=f"stage_demo_local_idx_{selected_user}",
    )

    sample_idx = int(user_indices[int(local_sample_idx)])

    st.caption(
        f"Selected user = `{selected_user}` | "
        f"local index = `{int(local_sample_idx)}` | "
        f"global test index = `{sample_idx}`"
    )
    return int(sample_idx)

def run_stage_progress(results: dict, sample_idx: int) -> dict:
    progress = st.progress(0)
    status = st.empty()

    status.info("Stage 1/4 — Loading raw physiological window...")
    sample = extract_sample_stage_outputs(results, int(sample_idx))
    time.sleep(0.15)
    progress.progress(25)

    status.info("Stage 2/4 — Computing FTS transition features...")
    time.sleep(0.15)
    progress.progress(50)

    status.info("Stage 3/4 — Running per-signal ANFIS inference...")
    time.sleep(0.15)
    progress.progress(75)

    status.info("Stage 4/4 — Running Bayesian Network soft-evidence fusion...")
    time.sleep(0.15)
    progress.progress(100)
    status.success("Completed.")
    return sample

def render_sample_summary(sample: dict) -> None:
    true_label_text = "task" if sample["y_true"] == 1 else "rest"
    pred_label_text = "task" if sample["y_pred"] == 1 else "rest"

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("User", str(sample["user_id"]))
    c2.metric("True", true_label_text)
    c3.metric("Pred", pred_label_text)
    c4.metric("Posterior P(task)", f"{sample['posterior_task']:.4f}")
    c5.metric("Threshold", f"{sample['threshold']:.4f}")
    c6.metric("Evidence mode", str(sample.get("evidence_mode", "unknown")))

def render_stage1_raw_window(sample: dict) -> None:
    with st.expander("Stage 1 — Raw physiological window", expanded=True):
        st.markdown("**Input:** one test sample with four synchronous physiological signals.")
        st.dataframe(sample["raw_df"], use_container_width=True)
        plot_signal_window(sample["raw_df"])

def render_stage2_fts(sample: dict) -> None:
    with st.expander("Stage 2 — FTS / fuzzy transition encoding", expanded=True):
        st.markdown("**Output:** one transition matrix and one flattened transition vector per signal.")
        cols = st.columns(4)
        for i, sig in enumerate(SIGNAL_NAMES):
            with cols[i]:
                st.markdown(f"**{sig} transition matrix**")
                st.dataframe(sample["transition_tables"][sig], use_container_width=True)

        st.markdown("**FTS feature vectors by signal**")
        for sig in SIGNAL_NAMES:
            with st.expander(f"{sig} FTS features", expanded=False):
                st.dataframe(sample["fts_feature_tables"][sig], use_container_width=True, hide_index=True)

def render_stage3_anfis(sample: dict) -> None:
    with st.expander("Stage 3 — Per-signal ANFIS inference", expanded=True):
        st.markdown("**Output:** one ANFIS probability per signal.")
        st.dataframe(sample["signal_prob_df"], use_container_width=True, hide_index=True)
        st.bar_chart(sample["signal_prob_df"].set_index("signal"))

        for sig in SIGNAL_NAMES:
            st.markdown(f"**{sig}**")
            left, right = st.columns([1, 1])
            with left:
                st.markdown("Top activated rules")
                st.dataframe(sample["top_rule_tables"][sig], use_container_width=True, hide_index=True)
            with right:
                st.markdown("Rule firing strengths")
                st.dataframe(sample["firing_tables"][sig], use_container_width=True, hide_index=True)
                plot_rule_firing(sample["firing_tables"][sig]["firing_strength"].values)

def render_stage4_bn(sample: dict) -> None:
    with st.expander("Stage 4 — Bayesian Network Fusion with Soft Evidence", expanded=True):
        st.markdown("**Input:** four ANFIS probability outputs.")
        st.markdown(
            "**Soft evidence:** each probability is converted into a membership vector "
            "over evidence levels."
        )
        st.markdown("**Output:** posterior probability over {rest, task}.")
        st.caption(
            "Unlike hard evidence, no signal is assigned to a single discrete bin. "
            "Each signal contributes as a probability distribution across multiple evidence levels."
        )
        bn_trace = sample["bn_trace"]
        post_df = pd.DataFrame(
            [
                {"class": "rest", "posterior": bn_trace["posterior_rest"]},
                {"class": "task", "posterior": sample["posterior_task"]},
            ]
        )

        st.markdown("**ANFIS probabilities per signal**")
        st.dataframe(sample["signal_prob_df"], use_container_width=True, hide_index=True)

        render_bn_score_summary(sample)

        if sample["soft_membership_df"] is not None:
            st.markdown("**Soft evidence membership**")
            st.dataframe(sample["soft_membership_df"], use_container_width=True, hide_index=True)
            render_soft_evidence_trace_table(bn_trace)
            pred_label = "TASK" if sample["y_pred"] == 1 else "REST"
            render_soft_membership_heatmap(
                sample["soft_membership_df"],
                caption=(
                    f"user={sample['user_id']} | idx={sample['sample_idx']} | "
                    f"P(TASK)={sample['posterior_task']:.4f} | "
                    f"P(REST)={bn_trace['posterior_rest']:.4f} | "
                    f"thr={sample['threshold']:.4f} -> {pred_label}"
                ),
            )
        else:
            st.info(
                "This selected variant is not running BN soft evidence, "
                "so no soft membership matrix is available."
            )

        st.markdown("**Posterior from BN inference**")
        st.dataframe(post_df, use_container_width=True, hide_index=True)
        st.bar_chart(post_df.set_index("class"))

        weights = bn_trace.get("signal_weights", {})
        if weights and not signal_weights_are_uniform(weights):
            st.markdown("**Signal weights used in soft evidence fusion**")
            st.dataframe(signal_weights_df(weights), use_container_width=True, hide_index=True)

        influence_df = compute_soft_influence_table(bn_trace)
        if influence_df is not None:
            st.markdown("**Soft evidence influence by class**")
            st.caption(
                "Each contribution is computed from the soft effective likelihood used by the BN: "
                "log(sum over evidence levels of membership * P(level | class)), with signal weighting applied."
            )
            st.dataframe(influence_df, use_container_width=True, hide_index=True)


def render_bn_score_summary(sample: dict) -> None:
    bn_trace = sample["bn_trace"]
    pred_label = "task" if sample["y_pred"] == 1 else "rest"
    prior = bn_trace.get("prior", {})
    total_scores = bn_trace.get("total_scores", {})
    summary_df = pd.DataFrame(
        [
            {"item": "Prior REST", "value": prior.get("rest", np.nan)},
            {"item": "Prior TASK", "value": prior.get("task", np.nan)},
            {"item": "Total score REST", "value": total_scores.get("rest", np.nan)},
            {"item": "Total score TASK", "value": total_scores.get("task", np.nan)},
            {"item": "Normalized posterior P(task)", "value": sample["posterior_task"]},
            {"item": "Selected threshold", "value": sample["threshold"]},
            {"item": "Final prediction", "value": pred_label},
            {"item": "Evidence mode", "value": sample.get("evidence_mode", "unknown")},
        ]
    )
    st.markdown("**BN score summary**")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    if not bn_trace.get("trace_matches_batch", True):
        st.warning(
            "The live BN trace posterior differs from the stored batch posterior. "
            "The displayed decision uses the stored batch result from the current experiment."
        )


def render_soft_evidence_trace_table(bn_trace: dict) -> None:
    rows = []
    for item in bn_trace.get("soft_evidence_trace", []):
        row = {
            "signal": item["signal"],
            "ANFIS probability": round(float(item["anfis_probability"]), 4),
            "likelihood rest": round(float(item["likelihood_rest"]), 6),
            "likelihood task": round(float(item["likelihood_task"]), 6),
            "dominant bin for interpretation only": item.get(
                "dominant_bin_for_interpretation_only", ""
            ),
        }
        for bin_name, value in item.get("bin_memberships", {}).items():
            row[f"mu {bin_name}"] = round(float(value), 4)
        for bin_name, value in item.get("cpt_rest", {}).items():
            row[f"P({bin_name}|rest)"] = round(float(value), 4)
        for bin_name, value in item.get("cpt_task", {}).items():
            row[f"P({bin_name}|task)"] = round(float(value), 4)
        rows.append(row)

    if not rows:
        return

    st.markdown("**Soft evidence inference trace by signal**")
    st.caption(
        "The dominant bin is displayed for interpretation only; soft BN inference uses "
        "the full membership vector and all CPT columns."
    )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
