from pathlib import Path

import streamlit as st

from streamlit_app.services.streamlit_support import (
    A2_SOFT_PRESET,
    get_group2_default_base_preset,
    resolve_selection,
    run_streamlit_experiment,
)


ABLATION_VARIANTS = {
    "Full pipeline — FTS → ANFIS → BN": "EXP3_FULL",
    "Without BN — FTS → ANFIS only": "EXP3_NO_BN",
    "Without ANFIS — FTS summary → BN": "EXP3_NO_ANFIS",
}

EXPERIMENT_GROUP = "exp3"


def _force_exp3_session_state() -> None:
    st.session_state["experiment_group"] = EXPERIMENT_GROUP
    st.session_state["selected_group"] = EXPERIMENT_GROUP

    results = st.session_state.get("results")
    selection = results.get("selection") if isinstance(results, dict) else None
    cfg = results.get("cfg") if isinstance(results, dict) else None
    result_group = getattr(selection, "group_key", None) or str(
        getattr(cfg, "experiment_group", EXPERIMENT_GROUP)
    ).lower()
    if result_group != EXPERIMENT_GROUP:
        st.session_state.pop("results", None)


def render_sidebar(root: Path) -> None:
    _force_exp3_session_state()

    with st.sidebar:
        st.header("Experiment")

        default_base_preset = get_group2_default_base_preset()
        if default_base_preset != A2_SOFT_PRESET:
            raise ValueError(f"Streamlit dashboard requires the {A2_SOFT_PRESET} preset.")

        base_preset_name = A2_SOFT_PRESET
        st.selectbox(
            "Preset / Base configuration",
            [A2_SOFT_PRESET],
            index=0,
            disabled=True,
        )

        experiment_group = EXPERIMENT_GROUP
        st.caption(
            "Compare the contribution of major pipeline stages using the "
            "ablation logic already defined in source."
        )

        selected_label = st.selectbox(
            "Ablation variant",
            list(ABLATION_VARIANTS),
            index=0,
        )
        selected_variant = ABLATION_VARIANTS[selected_label]
        stored_variant = st.session_state.get("exp3_selected_variant")
        if stored_variant is not None and stored_variant != selected_variant:
            st.session_state.pop("results", None)
            st.session_state.pop("exp3_result", None)

        data_path = st.text_input("Data path", "last_30s_segments")

        run_btn = st.button("Run / refresh experiment", use_container_width=True)

        if run_btn:
            previous_key = st.session_state.get("results", {}).get("cache_key")
            selection = resolve_selection(
                group_key=experiment_group,
                variant_key=selected_variant,
                base_preset_name=base_preset_name,
                n_rules_override=None,
            )
            if previous_key is not None:
                st.session_state.pop("results", None)
            with st.spinner(f"Running {selection.run_name}..."):
                result = run_streamlit_experiment(
                    project_root=root,
                    data_path=data_path,
                    selection=selection,
                )
                st.session_state["exp3_selected_variant"] = selected_variant
                st.session_state["exp3_result"] = result
                st.session_state["results"] = result
