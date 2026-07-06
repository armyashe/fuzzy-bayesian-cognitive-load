import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from streamlit_app.components.charts import (
    plot_threshold_sweep,
    render_confusion_matrix,
    render_cpt_llr_heatmap,
)
from streamlit_app.components.tables import confusion_df
from streamlit_app.utils import format_metric


PIPELINE_DISPLAY_NAMES = {
    "EXP3_FULL": "FTS–ANFIS–BN",
    "EXP3_NO_BN": "FTS–ANFIS–NoBN",
    "EXP3_NO_ANFIS": "FTS–NoANFIS–BN",
}


def render_results_tab(results: dict) -> None:
    dataset = results["dataset"]

    st.subheader("OOF / Cross-validation result")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", format_metric(results["oof_metrics"]["accuracy"]))
    c2.metric("Balanced Accuracy", format_metric(results["oof_metrics"]["bal_acc"]))
    c3.metric("AUC", format_metric(results["oof_metrics"]["auc"]))
    c4.metric("Selected threshold", format_metric(results["best_thr"]))

    plot_threshold_sweep(results["thresholds"], results["scores"])

    left, right = st.columns(2)
    with left:
        st.markdown("**OOF metric table**")
        st.dataframe(pd.DataFrame([results["oof_metrics"]]), use_container_width=True, hide_index=True)

        st.markdown("**OOF confusion matrix**")
        st.dataframe(confusion_df(results["oof_y"], results["oof_pred"]), use_container_width=True)

    with right:
        st.markdown("**Held-out test metric table**")
        st.dataframe(pd.DataFrame([results["test_metrics"]]), use_container_width=True, hide_index=True)

        st.markdown("**Held-out test confusion matrix**")
        st.dataframe(confusion_df(dataset.y_test, results["test_pred"]), use_container_width=True)

    st.subheader("Held-out test result")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", format_metric(results["test_metrics"]["accuracy"]))
    c2.metric("Balanced Accuracy", format_metric(results["test_metrics"]["bal_acc"]))
    c3.metric("AUC", format_metric(results["test_metrics"]["auc"]))

    st.subheader("Test Confusion Matrix")
    render_selected_variant_confusion_matrix(results)

    render_selected_variant_cpt_llr_heatmap(results)

    st.markdown("**Per-subject test metrics**")
    st.dataframe(results["subject_df"], use_container_width=True, hide_index=True)


def render_selected_variant_confusion_matrix(results: dict) -> None:
    selection = results["selection"]
    selected_variant = selection.variant_key
    stored_variant = st.session_state.get("exp3_selected_variant")

    if stored_variant != selected_variant:
        st.info(
            "Click 'Run / refresh experiment' "
            "to generate results for the selected variant."
        )
        return

    y_test = np.asarray(results.get("y_test", results["dataset"].y_test), dtype=int).reshape(-1)
    test_pred_value = results.get("test_pred")
    if test_pred_value is None and results.get("test_post") is not None:
        test_pred_value = (
            np.asarray(results["test_post"], dtype=float)
            >= float(results["best_thr"])
        ).astype(int)
    if test_pred_value is None:
        st.error("Cannot display confusion matrix: missing test predictions.")
        st.stop()
    test_pred = np.asarray(test_pred_value, dtype=int).reshape(-1)

    if len(y_test) != len(test_pred):
        st.error(
            "Cannot display confusion matrix: "
            "y_test and test_pred have different lengths."
        )
        st.stop()

    valid_labels = {0, 1}

    if not set(np.unique(y_test)).issubset(valid_labels):
        st.error("Unexpected labels found in y_test.")
        st.stop()

    if not set(np.unique(test_pred)).issubset(valid_labels):
        st.error("Unexpected labels found in test_pred.")
        st.stop()

    pipeline_name = PIPELINE_DISPLAY_NAMES.get(selected_variant, selected_variant)
    threshold = float(results["best_thr"])
    cm = render_confusion_matrix(
        y_test,
        test_pred,
        pipeline_name=pipeline_name,
        threshold=threshold,
    )
    if cm is None:
        return

    tn, fp, fn, tp = cm.ravel()
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("True REST", int(tn))
    col2.metric("False TASK", int(fp))
    col3.metric("False REST", int(fn))
    col4.metric("True TASK", int(tp))

    st.caption("Rows represent true classes and columns represent predicted classes.")

    cm_metrics = {
        "accuracy": float(accuracy_score(y_test, test_pred)),
        "precision": float(
            precision_score(
                y_test,
                test_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                test_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                test_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
    }
    test_metrics = results["test_metrics"]
    metric_keys = {
        "accuracy": ("accuracy", "acc"),
        "precision": ("precision",),
        "recall": ("recall",),
        "f1": ("f1",),
    }
    metrics_match = True
    for metric_name, keys in metric_keys.items():
        displayed_value = next(
            (test_metrics[key] for key in keys if key in test_metrics),
            None,
        )
        if displayed_value is not None and not np.isclose(
            cm_metrics[metric_name],
            float(displayed_value),
            atol=1e-4,
        ):
            metrics_match = False
            break

    if not metrics_match:
        st.warning(
            "The confusion-matrix predictions do not match "
            "the displayed test metrics."
        )


def render_selected_variant_cpt_llr_heatmap(results: dict) -> None:
    selection = results["selection"]
    selected_variant = selection.variant_key
    stored_variant = st.session_state.get("exp3_selected_variant")

    if stored_variant != selected_variant:
        return

    st.subheader("Bayesian Network CPT Log-Likelihood Ratio")

    if selected_variant == "EXP3_NO_BN":
        st.info(
            "Not applicable for this ablation variant because "
            "the Bayesian Network is removed and the four "
            "ANFIS outputs are combined using Mean Fusion."
        )
        return

    pipeline_name = PIPELINE_DISPLAY_NAMES.get(selected_variant, selected_variant)
    bn_model = results.get("bn_model") or results.get("models", {}).get("bn")

    if bn_model is None:
        st.warning(
            "The fitted Bayesian Network model is not "
            "available for CPT visualization."
        )
        return

    if selected_variant == "EXP3_FULL":
        st.caption(
            "Evidence nodes are obtained from per-signal ANFIS output probabilities. "
            "The CPTs describe the class-conditional distributions of the discretized "
            "per-signal ANFIS output probabilities."
        )
    elif selected_variant == "EXP3_NO_ANFIS":
        st.caption(
            "Evidence nodes are obtained from per-signal FTS summary scores. "
            "The CPTs describe the class-conditional distributions of the discretized "
            "per-signal FTS summary scores."
        )

    render_cpt_llr_heatmap(
        bn_model,
        pipeline_name=pipeline_name,
    )

    st.caption(
        "This heatmap summarizes the learned CPTs globally. During soft-evidence "
        "inference, a sample may contribute to multiple bins through its membership vector."
    )
