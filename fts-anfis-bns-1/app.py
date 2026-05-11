from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fbn_pipeline.config import ExperimentConfig, apply_preset, setup_seed, EXPERIMENT_PRESETS
from fbn_pipeline.data import load_dataset
from fbn_pipeline.metrics import find_best_threshold, per_subject_metrics
from fbn_pipeline.pipeline import run_cv, retrain_and_test, compute_oof_metrics
from fbn_pipeline.models.bn_fusion import SIGNAL_NAMES


st.set_page_config(page_title="Fuzzy–ANFIS–BN Dashboard", layout="wide")
st.title("Fuzzy–ANFIS–Bayesian Network Dashboard")


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
    rows = []
    total = len(y)
    for v, c in zip(values, counts):
        rows.append(
            {
                "label": mapping.get(int(v), str(v)),
                "count": int(c),
                "ratio": round(float(c / total), 4),
            }
        )
    return pd.DataFrame(rows)


def plot_threshold_sweep(thresholds: np.ndarray, scores: np.ndarray):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(thresholds, scores)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold sweep")
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)


def confusion_df(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=["Actual rest", "Actual task"],
        columns=["Pred rest", "Pred task"],
    )


def cpt_dict_to_df(cpt_signal_dict: dict) -> pd.DataFrame:
    return pd.DataFrame(cpt_signal_dict).T


def build_sample_signal_df(x_sample: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(x_sample.T, columns=SIGNAL_NAMES).rename_axis("t")


def run_full_experiment(data_path: str, preset_name: str) -> dict:
    cfg = ExperimentConfig(data_path=data_path)
    cfg = apply_preset(cfg, preset_name)
    setup_seed(cfg.seed)

    dataset = load_dataset(ROOT / cfg.data_path)

    oof_post, oof_y, fold_indices = run_cv(
        dataset.X_train,
        dataset.y_train,
        dataset.user_train,
        cfg,
    )

    best_thr, thr_score, thresholds, scores = find_best_threshold(
        oof_y,
        oof_post,
        metric=cfg.threshold_metric,
        n_grid=cfg.threshold_grid,
        return_sweep=True,
    )

    oof_metrics = compute_oof_metrics(oof_y, oof_post, best_thr)
    oof_pred = (oof_post >= best_thr).astype(int)

    test_pred, test_post, test_metrics, models = retrain_and_test(
        dataset.X_train,
        dataset.y_train,
        dataset.X_test,
        dataset.y_test,
        best_thr,
        cfg,
    )

    subject_rows = per_subject_metrics(
        dataset.y_test,
        test_pred,
        test_post,
        dataset.user_test,
    )

    return {
        "cfg": cfg,
        "dataset": dataset,
        "oof_post": oof_post,
        "oof_y": oof_y,
        "oof_pred": oof_pred,
        "fold_indices": fold_indices,
        "best_thr": best_thr,
        "thr_score": thr_score,
        "thresholds": thresholds,
        "scores": scores,
        "oof_metrics": oof_metrics,
        "test_pred": test_pred,
        "test_post": test_post,
        "test_metrics": test_metrics,
        "models": models,
        "subject_df": pd.DataFrame(subject_rows),
    }


def get_sample_artifacts(results: dict, sample_idx: int) -> dict:
    dataset = results["dataset"]
    cfg = results["cfg"]
    models = results["models"]

    x_sample = dataset.X_test[sample_idx]
    y_true = int(dataset.y_test[sample_idx])
    user_id = dataset.user_test[sample_idx]

    fts = models["fts"]
    bn = models["bn"]
    anfis_list = models["anfis_list"]
    te_probs = models["te_probs"]
    test_post = results["test_post"]
    test_pred = results["test_pred"]

    f_sample = fts.transform(x_sample[np.newaxis, ...])[0]
    feat_per_sig = len(f_sample) // len(SIGNAL_NAMES)
    feature_names = get_transition_feature_names(fts.n_sets)

    signal_rows = []
    rule_tables = {}
    transition_tables = {}

    for s, signal_name in enumerate(SIGNAL_NAMES):
        start = s * feat_per_sig
        end = start + feat_per_sig
        f_sig = f_sample[start:end]

        anfis = anfis_list[s]
        prob_s = float(te_probs[sample_idx, s])

        firing = anfis._normalised_firing(f_sig.reshape(1, -1))[0]
        rules = anfis.rule_strings(feature_names=feature_names, top_k=min(4, len(feature_names)))
        top_idx = np.argsort(firing)[::-1][:3]

        top_rule_rows = []
        for ridx in top_idx:
            top_rule_rows.append(
                {
                    "rule": f"R{ridx + 1}",
                    "firing_strength": round(float(firing[ridx]), 4),
                    "text": rules[ridx],
                }
            )

        rule_tables[signal_name] = pd.DataFrame(top_rule_rows)

        trans = fts.get_transition_matrix(x_sample, s)
        states = get_state_names(fts.n_sets)
        transition_tables[signal_name] = pd.DataFrame(trans, index=states, columns=states)

        signal_rows.append(
            {
                "signal": signal_name,
                "anfis_p_task": round(prob_s, 4),
            }
        )

    bn_trace = bn.explain_sample(te_probs[sample_idx])

    return {
        "sample_idx": sample_idx,
        "user_id": user_id,
        "y_true": y_true,
        "y_pred": int(test_pred[sample_idx]),
        "posterior_task": float(test_post[sample_idx]),
        "threshold": float(results["best_thr"]),
        "raw_df": build_sample_signal_df(x_sample),
        "signal_prob_df": pd.DataFrame(signal_rows),
        "rule_tables": rule_tables,
        "transition_tables": transition_tables,
        "bn_trace": bn_trace,
    }


with st.sidebar:
    st.header("Experiment")
    preset_name = st.selectbox(
        "Preset",
        list(EXPERIMENT_PRESETS.keys()),
        index=list(EXPERIMENT_PRESETS.keys()).index("A2_soft") if "A2_soft" in EXPERIMENT_PRESETS else 0,
    )
    data_path = st.text_input("Data path", "last_30s_segments")
    run_btn = st.button("Run / refresh experiment", use_container_width=True)

if run_btn:
    with st.spinner("Running full pipeline..."):
        st.session_state["results"] = run_full_experiment(data_path, preset_name)

if "results" not in st.session_state:
    st.info("Nhấn 'Run / refresh experiment' ở sidebar để chạy pipeline.")
    st.stop()

results = st.session_state["results"]
cfg = results["cfg"]
dataset = results["dataset"]
models = results["models"]
bn = models["bn"]

tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Run experiment", "Explain sample", "Rules & CPT"]
)

with tab1:
    st.subheader("Dataset overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train samples", dataset.X_train.shape[0])
    c2.metric("Test samples", dataset.X_test.shape[0])
    c3.metric("Signals", dataset.X_train.shape[1])
    c4.metric("Timesteps", dataset.X_train.shape[2])

    left, right = st.columns([1, 1])

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
        st.markdown("**Preset configuration**")
        cfg_df = pd.DataFrame(
            [{"parameter": k, "value": v} for k, v in cfg.to_dict().items()]
        )
        st.dataframe(cfg_df, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Cross-validation and held-out test")

    st.markdown("**OOF metrics**")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Threshold", f"{results['best_thr']:.4f}")
    c2.metric("Threshold score", f"{results['thr_score']:.4f}")
    c3.metric("OOF ACC", f"{results['oof_metrics']['acc']:.4f}")
    c4.metric("OOF F1", f"{results['oof_metrics']['f1']:.4f}")
    c5.metric("OOF AUC", f"{results['oof_metrics']['auc']:.4f}")

    plot_threshold_sweep(results["thresholds"], results["scores"])

    left, right = st.columns(2)
    with left:
        st.markdown("**OOF metric table**")
        st.dataframe(pd.DataFrame([results["oof_metrics"]]), use_container_width=True, hide_index=True)

        st.markdown("**OOF confusion matrix**")
        st.dataframe(confusion_df(results["oof_y"], results["oof_pred"]), use_container_width=True)

    with right:
        st.markdown("**Test metric table**")
        st.dataframe(pd.DataFrame([results["test_metrics"]]), use_container_width=True, hide_index=True)

        st.markdown("**Test confusion matrix**")
        st.dataframe(confusion_df(dataset.y_test, results["test_pred"]), use_container_width=True)

    st.markdown("**Per-subject test metrics**")
    st.dataframe(results["subject_df"], use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Sample-level explanation")

    sample_idx = st.number_input(
        "Chọn test sample index",
        min_value=0,
        max_value=len(dataset.X_test) - 1,
        value=0,
        step=1,
    )

    sample = get_sample_artifacts(results, int(sample_idx))
    true_label_text = "task" if sample["y_true"] == 1 else "rest"
    pred_label_text = "task" if sample["y_pred"] == 1 else "rest"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("User", str(sample["user_id"]))
    c2.metric("True", true_label_text)
    c3.metric("Pred", pred_label_text)
    c4.metric("Posterior P(task)", f"{sample['posterior_task']:.4f}")
    c5.metric("Threshold", f"{sample['threshold']:.4f}")

    st.markdown("**Raw 4-signal window**")
    st.dataframe(sample["raw_df"], use_container_width=True)

    st.markdown("**Per-signal ANFIS probabilities**")
    st.dataframe(sample["signal_prob_df"], use_container_width=True, hide_index=True)
    st.bar_chart(sample["signal_prob_df"].set_index("signal"))

    st.markdown("**FTS transition matrices**")
    cols = st.columns(4)
    for i, sig in enumerate(SIGNAL_NAMES):
        with cols[i]:
            st.markdown(f"**{sig}**")
            st.dataframe(sample["transition_tables"][sig], use_container_width=True)

    st.markdown("**Dominant ANFIS rules per signal**")
    for sig in SIGNAL_NAMES:
        with st.expander(f"{sig} — top activated rules", expanded=False):
            st.dataframe(sample["rule_tables"][sig], use_container_width=True, hide_index=True)

    st.markdown("**BN posterior trace**")
    bn_trace = sample["bn_trace"]

    left, right = st.columns(2)
    with left:
        st.markdown("**Evidence bins**")
        st.dataframe(
            pd.DataFrame(
                [{"signal": k, "bin": v} for k, v in bn_trace["evidence_bins"].items()]
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Signal weights**")
        st.dataframe(
            pd.DataFrame(
                [{"signal": k, "weight": v} for k, v in bn_trace["signal_weights"].items()]
            ),
            use_container_width=True,
            hide_index=True,
        )

    with right:
        st.markdown("**Posterior**")
        post_df = pd.DataFrame(
            [
                {"class": "rest", "posterior": bn_trace["posterior_rest"]},
                {"class": "task", "posterior": bn_trace["posterior_task"]},
            ]
        )
        st.dataframe(post_df, use_container_width=True, hide_index=True)
        st.bar_chart(post_df.set_index("class"))

    st.markdown("**Log-likelihood contributions**")
    contrib_rows = []
    for class_id, class_name in [(0, "rest"), (1, "task")]:
        for sig, value in bn_trace["log_likelihoods"][class_id].items():
            contrib_rows.append(
                {"class": class_name, "signal": sig, "log_contribution": round(float(value), 4)}
            )
    st.dataframe(pd.DataFrame(contrib_rows), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("Model rules and Bayesian CPTs")

    st.markdown("**BN prior**")
    prior_df = pd.DataFrame(
        [
            {"class": "rest", "prior": round(float(bn.prior_[0]), 4)},
            {"class": "task", "prior": round(float(bn.prior_[1]), 4)},
        ]
    )
    st.dataframe(prior_df, use_container_width=True, hide_index=True)

    st.markdown("**BN signal weights**")
    st.dataframe(
        pd.DataFrame(
            [{"signal": k, "weight": v} for k, v in bn.get_signal_weights().items()]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**BN CPT tables**")
    cpt_tables = bn.get_cpt_table()
    cols = st.columns(4)
    for i, sig in enumerate(SIGNAL_NAMES):
        with cols[i]:
            st.markdown(f"**{sig}**")
            st.dataframe(cpt_dict_to_df(cpt_tables[sig]), use_container_width=True)

    st.markdown("**ANFIS rule bases**")
    feature_names = get_transition_feature_names(cfg.n_sets)
    for sig, anfis in zip(SIGNAL_NAMES, models["anfis_list"]):
        with st.expander(f"{sig} rules", expanded=False):
            rules = anfis.rule_strings(feature_names=feature_names, top_k=min(4, len(feature_names)))
            rule_df = pd.DataFrame(
                [{"rule_id": f"R{i+1}", "text": rule} for i, rule in enumerate(rules)]
            )
            st.dataframe(rule_df, use_container_width=True, hide_index=True)