import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from fbn_pipeline.models.bn_fusion import SIGNAL_NAMES


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


def render_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    pipeline_name: str,
    threshold: float,
) -> np.ndarray | None:
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)

    if y_true.shape != y_pred.shape:
        st.error(
            "Cannot display confusion matrix because "
            "y_true and y_pred have different shapes."
        )
        return None

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["REST", "TASK"],
    )

    display.plot(
        ax=ax,
        colorbar=False,
        cmap="Blues",
        values_format="d",
    )

    ax.set_title(
        f"{pipeline_name}\n"
        f"Test Confusion Matrix — threshold = {threshold:.4f}",
        fontsize=11,
        pad=10,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=False,
    )

    plt.close(fig)
    return cm


def compute_cpt_log_likelihood_ratio(
    bn_model,
    *,
    epsilon: float = 1e-10,
) -> np.ndarray:
    if bn_model is None:
        raise ValueError("BN model is required to compute CPT LLR.")

    if not hasattr(bn_model, "cpts_"):
        raise AttributeError(
            "The fitted BN model does not contain cpts_."
        )

    rows: list[np.ndarray] = []

    for cpt in bn_model.cpts_:
        cpt = np.asarray(cpt, dtype=float)

        if cpt.ndim != 2 or cpt.shape[0] != 2:
            raise ValueError(
                "Each CPT must have shape (2, n_bins), "
                "representing REST and TASK."
            )

        p_rest = np.clip(cpt[0], epsilon, None)
        p_task = np.clip(cpt[1], epsilon, None)

        llr = np.log(p_task / p_rest)
        rows.append(llr)

    return np.vstack(rows)


def render_cpt_llr_heatmap(
    bn_model,
    *,
    pipeline_name: str,
) -> None:
    llr = compute_cpt_log_likelihood_ratio(bn_model)

    bin_names = list(
        getattr(
            bn_model,
            "bin_names_",
            [f"bin_{i}" for i in range(llr.shape[1])],
        )
    )

    signal_names = SIGNAL_NAMES[: llr.shape[0]]

    max_abs = float(np.max(np.abs(llr)))
    if max_abs <= 0:
        max_abs = 1.0

    fig_width = max(7.0, 1.25 * len(bin_names))
    fig_height = max(3.6, 0.75 * len(signal_names) + 1.5)

    fig, ax = plt.subplots(
        figsize=(fig_width, fig_height)
    )

    image = ax.imshow(
        llr,
        cmap="RdBu_r",
        vmin=-max_abs,
        vmax=max_abs,
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(bin_names)))
    ax.set_xticklabels(bin_names)

    ax.set_yticks(np.arange(len(signal_names)))
    ax.set_yticklabels(signal_names)

    ax.set_xlabel("Evidence bin")
    ax.set_ylabel("Physiological signal")

    ax.set_title(
        "CPT Log-Likelihood Ratio\n"
        r"$LLR(s,b)=\log\frac{P(E_s=b\mid TASK)}"
        r"{P(E_s=b\mid REST)}$"
        f"\n{pipeline_name}",
        fontsize=11,
        pad=10,
    )

    threshold = max_abs * 0.45

    for row in range(llr.shape[0]):
        for col in range(llr.shape[1]):
            value = float(llr[row, col])

            text_color = (
                "white"
                if abs(value) >= threshold
                else "black"
            )

            ax.text(
                col,
                row,
                f"{value:+.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight="bold",
            )

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label(
        "Log-likelihood ratio",
        rotation=270,
        labelpad=15,
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        use_container_width=True,
    )

    plt.close(fig)

    st.caption(
        "Positive values support TASK; negative values support REST. "
        "The values are derived from the fitted Bayesian Network CPTs."
    )
