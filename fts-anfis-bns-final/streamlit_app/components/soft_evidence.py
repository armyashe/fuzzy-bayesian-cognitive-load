import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from fbn_pipeline.models.bn_fusion import SIGNAL_NAMES


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
