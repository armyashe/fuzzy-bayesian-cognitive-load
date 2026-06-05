import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


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
