from pathlib import Path

import streamlit as st

from fbn_pipeline.config import EXPERIMENT_PRESETS
from fbn_pipeline.streamlit_support import (
    RULE_COUNT_OPTIONS,
    get_group1_default_variant,
    get_group2_default_base_preset,
    get_group3_default_variant,
    get_group_help,
    get_group_options,
    get_variant_options,
    group2_variant_label,
    resolve_selection,
    run_streamlit_experiment,
)


GROUP_OPTIONS = get_group_options()
GROUP_LABELS = {option.key: option.label for option in GROUP_OPTIONS}


def render_sidebar(root: Path) -> None:
    with st.sidebar:
        st.header("Experiment")

        base_preset_keys = [
            key for key in EXPERIMENT_PRESETS.keys()
            if not key.startswith(("EXP1_", "EXP2_", "EXP3_"))
        ]
        default_base_preset = get_group2_default_base_preset()
        default_base_idx = (
            base_preset_keys.index(default_base_preset)
            if default_base_preset in base_preset_keys
            else 0
        )

        base_preset_name = st.selectbox(
            "Preset / Base configuration",
            base_preset_keys,
            index=default_base_idx,
        )

        group_key = st.selectbox(
            "Experiment group",
            [option.key for option in GROUP_OPTIONS],
            index=0,
            format_func=lambda key: GROUP_LABELS[key],
        )

        st.caption(get_group_help(group_key))

        variant_key: str | None = None
        n_rules_override: int | None = None

        if group_key == "exp1":
            group1_options = get_variant_options("exp1")
            group1_keys = [option.key for option in group1_options]
            default_group1 = get_group1_default_variant()
            variant_key = st.selectbox(
                "Variant / preset",
                group1_keys,
                index=group1_keys.index(default_group1),
                format_func=lambda key: next(option.label for option in group1_options if option.key == key),
            )
        elif group_key == "exp2":
            st.selectbox(
                "Variant",
                [base_preset_name],
                index=0,
                format_func=lambda _: group2_variant_label(base_preset_name),
                disabled=True,
            )
            n_rules_override = st.selectbox(
                "Number of ANFIS rules",
                list(RULE_COUNT_OPTIONS),
                index=list(RULE_COUNT_OPTIONS).index(6),
            )
        else:
            group3_options = get_variant_options("exp3")
            group3_keys = [option.key for option in group3_options]
            default_group3 = get_group3_default_variant()
            variant_key = st.selectbox(
                "Ablation variant",
                group3_keys,
                index=group3_keys.index(default_group3),
                format_func=lambda key: next(option.label for option in group3_options if option.key == key),
            )

        data_path = st.text_input("Data path", "last_30s_segments")

        run_btn = st.button("Run / refresh experiment", use_container_width=True)

    if run_btn:
        selection = resolve_selection(
            group_key=group_key,
            variant_key=variant_key,
            base_preset_name=base_preset_name,
            n_rules_override=n_rules_override,
        )
        with st.spinner(f"Running {selection.run_name}..."):
            st.session_state["results"] = run_streamlit_experiment(
                project_root=root,
                data_path=data_path,
                selection=selection,
            )
