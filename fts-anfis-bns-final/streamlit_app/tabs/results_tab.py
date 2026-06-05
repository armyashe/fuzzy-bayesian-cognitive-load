import pandas as pd
import streamlit as st

from streamlit_app.components.charts import plot_threshold_sweep
from streamlit_app.components.tables import confusion_df
from streamlit_app.utils import format_metric


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

    st.markdown("**Per-subject test metrics**")
    st.dataframe(results["subject_df"], use_container_width=True, hide_index=True)
