from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fbn_pipeline.config import EXPERIMENT_PRESETS, ExperimentConfig
from fbn_pipeline.models.bn_fusion import SIGNAL_NAMES
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


st.set_page_config(page_title="Fuzzy–ANFIS–BN Dashboard", layout="wide")
st.title("Fuzzy–ANFIS–Bayesian Network Dashboard")

GROUP_OPTIONS = get_group_options()
GROUP_LABELS = {option.key: option.label for option in GROUP_OPTIONS}


def get_state_names(n_sets: int) -> list[str]:
    if n_sets == 3:
        return ["L", "M", "H"]
    if n_sets == 5:
        return ["VL", "L", "M", "H", "VH"]
    return [f"S{i}" for i in range(n_sets)]


def get_transition_feature_names(n_sets: int) -> list[str]:
    states = get_state_names(n_sets)
    return [f"{a}→{b}" for a in states for b in states]


def label_distribution(y: np.ndarray) -> pd.DataFrame:
    values, counts = np.unique(y, return_counts=True)
    mapping = {0: "rest", 1: "task"}
    total = len(y)
    rows = []
    for v, c in zip(values, counts):
        rows.append(
            {
                "label": mapping.get(int(v), str(v)),
                "count": int(c),
                "ratio": round(float(c / total), 4),
            }
        )
    return pd.DataFrame(rows)


def confusion_df(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=["Actual rest", "Actual task"],
        columns=["Pred rest", "Pred task"],
    )


def plot_threshold_sweep(thresholds: np.ndarray, scores: np.ndarray):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, scores)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold sweep")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


def plot_signal_window(raw_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 4))
    for col in raw_df.columns:
        ax.plot(raw_df.index, raw_df[col], label=col)
    ax.set_xlabel("t")
    ax.set_ylabel("Value")
    ax.set_title("Raw physiological window")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


def plot_rule_firing(firing: np.ndarray):
    df = pd.DataFrame(
        {"rule": [f"R{i+1}" for i in range(len(firing))], "firing_strength": firing}
    ).set_index("rule")
    st.bar_chart(df)


def cpt_dict_to_df(cpt_signal_dict: dict) -> pd.DataFrame:
    return pd.DataFrame(cpt_signal_dict).T


def signal_weights_df(signal_weights: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [{"signal": k, "weight": round(float(v), 4)} for k, v in signal_weights.items()]
    )


def signal_weights_are_uniform(signal_weights: dict) -> bool:
    if not signal_weights:
        return True
    values = [float(v) for v in signal_weights.values()]
    return bool(np.allclose(values, np.ones(len(values))))


def load_bn_artifact_json(exp_name: str, artifacts_root: Path | str = "artifacts") -> dict | None:
    """Load BN details from JSON artifact if it exists."""
    artifact_root = Path(artifacts_root) if isinstance(artifacts_root, str) else artifacts_root
    artifact_path = artifact_root / exp_name / "bn_details.json"
    if artifact_path.exists():
        try:
            import json
            with open(artifact_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Failed to load BN JSON: {e}")
            return None
    return None


def load_anfis_rules_json(exp_name: str, artifacts_root: Path | str = "artifacts") -> dict | None:
    """Load ANFIS rules from JSON artifact if it exists."""
    artifact_root = Path(artifacts_root) if isinstance(artifacts_root, str) else artifacts_root
    artifact_path = artifact_root / exp_name / "anfis_rules.json"
    if artifact_path.exists():
        try:
            import json
            with open(artifact_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Failed to load ANFIS rules JSON: {e}")
            return None
    return None


def get_bn_graph_image_path(exp_name: str, artifacts_root: Path | str = "artifacts") -> Path | None:
    """Get BN graph image path if it exists."""
    artifact_root = Path(artifacts_root) if isinstance(artifacts_root, str) else artifacts_root
    img_path = artifact_root / exp_name / "bn_graph.png"
    if img_path.exists():
        return img_path
    return None


def build_raw_df(x_sample: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(x_sample.T, columns=SIGNAL_NAMES).rename_axis("t")


def render_method_diagram(cfg: ExperimentConfig, timesteps: int):
    bn_mode = "Soft Evidence" if cfg.bn_soft else "Hard Evidence"
    html = f"""
    <style>
    .method-wrap {{margin-top: 0.25rem;}}
    .method-title {{font-size: 1.05rem; font-weight: 700; margin-bottom: 0.5rem;}}
    .method-note {{font-size: 0.92rem; color: #666; margin: 0.35rem 0 0.8rem 0;}}
    .method-box {{
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 10px;
        background: rgba(250, 250, 250, 0.85);
        color: #111;
    }}
    .input-box {{background: #eef2f7;}}
    .layer1 {{background: #e7f1dc;}}
    .layer2 {{background: #f8edcf;}}
    .layer3 {{background: #dfeaf8;}}
    .final-box {{background: #f2f2f2;}}
    .box-head {{font-size: 1rem; font-weight: 800; margin-bottom: 8px;}}
    .chip-row {{display: flex; gap: 8px; flex-wrap: wrap; margin-top: 6px;}}
    .chip {{
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.9);
        border: 1px solid rgba(49, 51, 63, 0.18);
        font-size: 0.86rem;
        font-weight: 600;
    }}
    .stage-grid {{display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px;}}
    .stage-grid-3 {{display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px;}}
    .mini {{
        background: rgba(255,255,255,0.92);
        border: 1px solid rgba(49, 51, 63, 0.18);
        border-radius: 12px;
        padding: 10px 12px;
        min-height: 82px;
    }}
    .mini-head {{font-size: 0.9rem; font-weight: 800; margin-bottom: 4px;}}
    .mini-body {{font-size: 0.83rem; line-height: 1.35;}}
    .arrow {{text-align: center; font-size: 1.4rem; font-weight: 800; color: #555; margin: 0.1rem 0 0.5rem 0;}}
    .caption {{font-size: 0.85rem; color: #444; margin-top: 8px;}}
    </style>

    <div class="method-wrap">
      <div class="method-title">Proposed Method</div>

      <div class="method-box input-box">
        <div class="box-head">Input Data</div>
        <div class="chip-row">
          <span class="chip">Channel 0: Temp x₀(t)</span>
          <span class="chip">Channel 1: HR x₁(t)</span>
          <span class="chip">Channel 2: GSR x₂(t)</span>
          <span class="chip">Channel 3: RR x₃(t)</span>
          <span class="chip">Window T = {timesteps}</span>
        </div>
      </div>

      <div class="arrow">↓</div>

      <div class="method-box layer1">
        <div class="box-head">Layer 1 — FTS Encoding</div>
        <div class="stage-grid">
          <div class="mini">
            <div class="mini-head">Gaussian / Fuzzy Partition</div>
            <div class="mini-body">{cfg.mf_type} membership functions with K = {cfg.n_sets} fuzzy sets per signal.</div>
          </div>
          <div class="mini">
            <div class="mini-head">Membership Sequence</div>
            <div class="mini-body">Convert each raw time point into a normalized fuzzy membership vector.</div>
          </div>
          <div class="mini">
            <div class="mini-head">Transition Extraction</div>
            <div class="mini-body">Compute fuzzy transition counts from consecutive time steps for each channel.</div>
          </div>
          <div class="mini">
            <div class="mini-head">Transition Vector</div>
            <div class="mini-body">Normalize transitions with mode = {cfg.fts_norm} and flatten to a {cfg.n_sets * cfg.n_sets}-D vector per signal.</div>
          </div>
        </div>
        <div class="caption">Each channel is encoded independently, then concatenated into one FTS feature representation.</div>
      </div>

      <div class="arrow">↓</div>

      <div class="method-box layer2">
        <div class="box-head">Layer 2 — Per-signal ANFIS-TSK</div>
        <div class="stage-grid">
          <div class="mini">
            <div class="mini-head">Rule Firing</div>
            <div class="mini-body">Premise layer computes normalized firing strengths for R = {cfg.n_rules} fuzzy rules.</div>
          </div>
          <div class="mini">
            <div class="mini-head">TSK Consequent</div>
            <div class="mini-body">Each rule maps the FTS vector to a local linear consequent.</div>
          </div>
          <div class="mini">
            <div class="mini-head">Aggregate Output</div>
            <div class="mini-body">Weighted sum across active rules yields a raw per-signal score.</div>
          </div>
          <div class="mini">
            <div class="mini-head">Sigmoid Probability</div>
            <div class="mini-body">Transform the raw score into one probability pₛ = P(Y = 1) for each signal.</div>
          </div>
        </div>
        <div class="caption">Outputs a 4-D evidence vector: [p₀, p₁, p₂, p₃] for Temp, HR, GSR, and RR.</div>
      </div>

      <div class="arrow">↓</div>

      <div class="method-box layer3">
        <div class="box-head">Layer 3 — Bayesian Network Fusion</div>
        <div class="stage-grid-3">
          <div class="mini">
            <div class="mini-head">Evidence Modeling</div>
            <div class="mini-body">Discretize or softly distribute each ANFIS probability into B = {cfg.n_bins} evidence bins using {bn_mode.lower()}.</div>
          </div>
          <div class="mini">
            <div class="mini-head">Naive Bayes DAG</div>
            <div class="mini-body">Task node Y points to evidence nodes E<sub>Temp</sub>, E<sub>HR</sub>, E<sub>GSR</sub>, E<sub>RR</sub>.</div>
          </div>
          <div class="mini">
            <div class="mini-head">Posterior Inference</div>
            <div class="mini-body">Combine prior and CPT likelihoods to estimate P(Y = 1 | evidence).</div>
          </div>
        </div>
      </div>

      <div class="arrow">↓</div>

      <div class="method-box final-box">
        <div class="box-head">Final Output</div>
        <div class="chip-row">
          <span class="chip">Posterior: P(Y = 1 | p)</span>
          <span class="chip">Binary decision: Task (1) vs Rest (0)</span>
        </div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def compute_soft_membership_table(bn, probs_single: np.ndarray) -> pd.DataFrame | None:
    if not getattr(bn, "soft", False):
        return None
    if not hasattr(bn, "bin_mids_") or not hasattr(bn, "bin_sigmas_"):
        return None

    rows = []
    for s in range(len(probs_single)):
        mids = bn.bin_mids_[s]
        sigma = bn.bin_sigmas_[s]
        mem = np.exp(-0.5 * ((probs_single[s] - mids) / max(sigma, 1e-10)) ** 2)
        mem = mem / np.maximum(mem.sum(), 1e-10)
        row = {
            "signal": SIGNAL_NAMES[s],
            "p_task_from_ANFIS": round(float(probs_single[s]), 4),
        }
        for b, value in enumerate(mem):
            row[bn.bin_names_[b]] = round(float(value), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def render_soft_membership_heatmap(membership_df: pd.DataFrame, caption: str | None = None) -> None:
    level_cols = [
        col for col in membership_df.columns
        if col not in {"signal", "p_task_from_ANFIS"}
    ]
    if not level_cols:
        return

    heatmap_df = membership_df.set_index("signal")[level_cols].astype(float)
    values = heatmap_df.values

    fig_width = max(7.2, 1.2 * len(level_cols))
    fig_height = max(3.8, 0.7 * len(heatmap_df.index) + 1.6)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Use white background like the reference figure
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Orange/yellow/red colormap like the notebook figure
    image = ax.imshow(
        values,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )

    ax.set_title(
        "Soft Evidence Memberships for a Test Sample",
        color="black",
        fontsize=13,
        pad=12,
        weight="bold",
    )

    ax.set_xlabel("Bayesian evidence bins", color="black", labelpad=8)
    ax.set_ylabel("Physiological signals", color="black", labelpad=8)

    ax.set_xticks(np.arange(len(level_cols)))
    ax.set_xticklabels(level_cols, rotation=30, ha="right", color="black")

    ax.set_yticks(np.arange(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index, color="black")

    ax.tick_params(colors="black")

    # Annotate values and highlight dominant membership per signal
    for row_idx in range(values.shape[0]):
        max_col_idx = int(np.argmax(values[row_idx]))

        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            is_max = col_idx == max_col_idx

            text_color = "white" if value > 0.55 else "black"
            label = f"{value:.4f}\n★" if is_max else f"{value:.4f}"

            ax.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
                fontweight="bold" if is_max else "normal",
            )

        # Black border around dominant membership cell
        ax.add_patch(
            plt.Rectangle(
                (max_col_idx - 0.5, row_idx - 0.5),
                1,
                1,
                fill=False,
                edgecolor="black",
                linewidth=2.0,
                zorder=3,
            )
        )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Membership degree μ", color="black")
    colorbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(colorbar.ax.get_yticklabels(), color="black")
    colorbar.outline.set_edgecolor("black")

    for spine in ax.spines.values():
        spine.set_edgecolor("black")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    if caption:
        st.caption(caption)

    st.caption(
        "★ indicates the dominant membership level for visualization only; "
        "BN inference still uses the full soft membership vector."
    )


def compute_soft_influence_table(bn_trace: dict) -> pd.DataFrame | None:
    if bn_trace.get("mode") != "soft":
        return None

    rows = []
    for class_id, class_name in [(0, "rest"), (1, "task")]:
        for sig, value in bn_trace.get("log_likelihoods", {}).get(class_id, {}).items():
            rows.append(
                {
                    "class": class_name,
                    "signal": sig,
                    "soft_log_contribution": round(float(value), 4),
                    "effective_likelihood": round(
                        float(
                            bn_trace.get("effective_likelihoods", {})
                            .get(class_name, {})
                            .get(sig, np.nan)
                        ),
                        4,
                    ),
                }
            )
    return pd.DataFrame(rows) if rows else None


def compute_hard_debug_contribution_table(bn_trace: dict) -> pd.DataFrame:
    contrib_rows = []
    for class_id, class_name in [(0, "rest"), (1, "task")]:
        for sig, value in bn_trace.get("log_likelihoods", {}).get(class_id, {}).items():
            contrib_rows.append(
                {
                    "class": class_name,
                    "signal": sig,
                    "log_contribution": round(float(value), 4),
                }
            )
    return pd.DataFrame(contrib_rows)


def format_metric(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{float(value):.4f}"


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

        signal_prob_rows.append(
            {"signal": signal_name, "anfis_p_task": round(prob_s, 4)}
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
    posterior_task = float(bn.predict_proba(te_probs_manual.reshape(1, -1))[0])
    bn_trace = bn.explain_sample(te_probs_manual)
    soft_df = compute_soft_membership_table(bn, te_probs_manual)

    return {
        "sample_idx": sample_idx,
        "user_id": user_id,
        "y_true": y_true,
        "y_pred": y_pred,
        "threshold": float(results["best_thr"]),
        "posterior_task": posterior_task,
        "raw_df": raw_df,
        "transition_tables": transition_tables,
        "fts_feature_tables": fts_feature_tables,
        "signal_prob_df": pd.DataFrame(signal_prob_rows),
        "firing_tables": firing_tables,
        "top_rule_tables": top_rule_tables,
        "bn_trace": bn_trace,
        "soft_membership_df": soft_df,
    }


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
            project_root=ROOT,
            data_path=data_path,
            selection=selection,
        )

if "results" not in st.session_state:
    st.info("Select an experiment group and run the pipeline from the sidebar.")
    st.stop()

results = st.session_state["results"]
cfg = results["cfg"]
dataset = results["dataset"]
models = results["models"]
selection = results["selection"]

fts = models.get("fts")
bn = models.get("bn")
anfis_list = models.get("anfis_list")

has_fts = fts is not None
has_bn = bn is not None
has_anfis = anfis_list is not None

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Results", "Sample Explanation", "BN / Rules"]
)

with tab1:
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
            → `FTS / fuzzy transition encoding`
            → `4 per-signal ANFIS models`
            → `Bayesian Network fusion`
            → `Final cognitive-load prediction`
            """
        )

        st.markdown("**Resolved configuration**")
        cfg_df = pd.DataFrame(
            [{"parameter": k, "value": v} for k, v in cfg.to_dict().items()]
        )
        st.dataframe(cfg_df, use_container_width=True, hide_index=True)

with tab2:
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

with tab3:
    st.subheader("Sample explanation")
    if not selection.supports_stage_demo or not (has_fts and has_anfis and has_bn):
        st.warning(
            "Detailed stage-by-stage explanation is available only when the current variant keeps the full FTS → ANFIS → BN pipeline."
        )

    else:
    
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

        run_sample_btn = st.button(
            "Run selected sample step-by-step",
            use_container_width=True,
        )
        if run_sample_btn:
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

            true_label_text = "task" if sample["y_true"] == 1 else "rest"
            pred_label_text = "task" if sample["y_pred"] == 1 else "rest"

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("User", str(sample["user_id"]))
            c2.metric("True", true_label_text)
            c3.metric("Pred", pred_label_text)
            c4.metric("Posterior P(task)", f"{sample['posterior_task']:.4f}")
            c5.metric("Threshold", f"{sample['threshold']:.4f}")

            with st.expander("Stage 1 — Raw physiological window", expanded=True):
                st.markdown("**Input:** one test sample with four synchronous physiological signals.")
                st.dataframe(sample["raw_df"], use_container_width=True)
                plot_signal_window(sample["raw_df"])

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

                if sample["soft_membership_df"] is not None:
                    st.markdown("**Soft evidence membership**")
                    st.dataframe(sample["soft_membership_df"], use_container_width=True, hide_index=True)
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

                st.markdown("**Posterior from soft evidence BN inference**")
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

                debug_items = []
                if bn_trace.get("dominant_bins"):
                    debug_items.append("dominant_bins")
                if bn_trace.get("evidence_bins") and bn_trace.get("mode") != "soft":
                    debug_items.append("hard_evidence_bins")
                if bn_trace.get("signal_weights") and signal_weights_are_uniform(bn_trace["signal_weights"]):
                    debug_items.append("uniform_signal_weights")
                if bn_trace.get("mode") != "soft" and bn_trace.get("log_likelihoods"):
                    debug_items.append("hard_contributions")

                if debug_items:
                    with st.expander("Debug: hard-bin approximation", expanded=False):
                        st.caption(
                            "This section is only a debug approximation and is not the inference mechanism "
                            "used in the main soft-evidence experiment."
                        )
                        if "dominant_bins" in debug_items:
                            st.markdown("**Dominant evidence level for visualization only**")
                            st.dataframe(
                                pd.DataFrame(
                                    [
                                        {"signal": k, "dominant_evidence_level": v}
                                        for k, v in bn_trace["dominant_bins"].items()
                                    ]
                                ),
                                use_container_width=True,
                                hide_index=True,
                            )
                        if "hard_evidence_bins" in debug_items:
                            st.markdown("**Hard evidence bins for non-soft variants**")
                            st.dataframe(
                                pd.DataFrame(
                                    [{"signal": k, "hard_bin": v} for k, v in bn_trace["evidence_bins"].items()]
                                ),
                                use_container_width=True,
                                hide_index=True,
                            )
                        if "uniform_signal_weights" in debug_items:
                            st.markdown("**Uniform signal weights**")
                            st.dataframe(signal_weights_df(bn_trace["signal_weights"]), use_container_width=True, hide_index=True)
                        if "hard_contributions" in debug_items:
                            st.markdown("**Hard-bin log-likelihood contributions**")
                            st.dataframe(
                                compute_hard_debug_contribution_table(bn_trace),
                                use_container_width=True,
                                hide_index=True,
                            )
        else:
            st.info("Chọn sample index rồi bấm 'Run selected sample step-by-step'.")

with tab4:
    st.subheader("Model rules and Bayesian CPTs")

    exp_name = results.get("exp_name", "main_run")

    bn_details = load_bn_artifact_json(exp_name, artifacts_root=ROOT / "artifacts")
    anfis_rules_data = load_anfis_rules_json(exp_name, artifacts_root=ROOT / "artifacts")
    bn_graph_path = get_bn_graph_image_path(exp_name, artifacts_root=ROOT / "artifacts")

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

    if bn_details is None and anfis_rules_data is None and bn_graph_path is None:
        st.warning(
            "⚠️ **No artifacts found.** Run the model first to generate graph, rules, and metrics.\n"
            f"Artifacts should be saved in: `artifacts/{exp_name}/`"
        )
