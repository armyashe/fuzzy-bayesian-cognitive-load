import pandas as pd
import streamlit as st

from streamlit_app.components.tables import label_distribution


def render_overview_tab(results: dict) -> None:
    cfg = results["cfg"]
    dataset = results["dataset"]
    selection = results["selection"]

    st.subheader("Experiment")

    st.markdown(f"**Preset / Base configuration:** `{selection.base_preset_name}`")
    st.markdown(f"**Experiment group:** {selection.group_label}")
    if selection.group_key == "exp1":
        st.markdown(f"**Variant / preset:** {selection.variant_label}")
    elif selection.group_key == "exp2":
        st.markdown(f"**Variant:** {selection.variant_label}")
    else:
        st.markdown(f"**Ablation variant:** {selection.variant_label}")

    st.markdown("**Current configuration summary**")
    st.dataframe(
        pd.DataFrame(results["config_summary_rows"]),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Dataset overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train samples", dataset.X_train.shape[0])
    c2.metric("Test samples", dataset.X_test.shape[0])
    c3.metric("Signals", dataset.X_train.shape[1])
    c4.metric("Timesteps", dataset.X_train.shape[2])

    left, right = st.columns(2)

    with left:
        st.markdown("**Shapes**")
        shape_df = pd.DataFrame(
            [
                {"split": "X_train", "shape": str(dataset.X_train.shape)},
                {"split": "y_train", "shape": str(dataset.y_train.shape)},
                {"split": "user_train", "shape": str(dataset.user_train.shape)},
                {"split": "X_test", "shape": str(dataset.X_test.shape)},
                {"split": "y_test", "shape": str(dataset.y_test.shape)},
                {"split": "user_test", "shape": str(dataset.user_test.shape)},
            ]
        )
        st.dataframe(shape_df, use_container_width=True, hide_index=True)

        st.markdown("**Train label distribution**")
        st.dataframe(label_distribution(dataset.y_train), use_container_width=True, hide_index=True)

        st.markdown("**Test label distribution**")
        st.dataframe(label_distribution(dataset.y_test), use_container_width=True, hide_index=True)

    with right:
        st.markdown("**Pipeline structure**")
        st.markdown(
            """
            `Raw physiological window`
            -> `FTS / fuzzy transition encoding`
            -> `4 per-signal ANFIS models`
            -> `Bayesian Network fusion`
            -> `Final cognitive-load prediction`
            """
        )

        st.markdown("**Resolved configuration**")
        cfg_df = pd.DataFrame(
            [{"parameter": k, "value": v} for k, v in cfg.to_dict().items()]
        )
        st.dataframe(cfg_df, use_container_width=True, hide_index=True)
