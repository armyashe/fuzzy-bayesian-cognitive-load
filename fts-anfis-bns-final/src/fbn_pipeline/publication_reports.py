from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve


def generate_tables_from_run(run_root: Path | str) -> list[Path]:
    root = Path(run_root)
    metadata = json.loads((root / "metadata" / "runtime_metadata.json").read_text(encoding="utf-8"))
    predictions = _safe_read_csv(root / "predictions" / "outer_fold_predictions.csv")
    overall_metrics = _safe_read_csv(root / "metrics" / "overall_metrics.csv")
    per_subject_metrics = _safe_read_csv(root / "metrics" / "per_subject_metrics.csv")
    ci_path = root / "statistics" / "confidence_intervals.csv"
    ci = _safe_read_csv(ci_path) if ci_path.exists() else pd.DataFrame()
    selection_path = root / "statistics" / "selection_frequencies.csv"
    selection = _safe_read_csv(selection_path) if selection_path.exists() else pd.DataFrame()
    stats_path = root / "statistics" / "statistical_tests.csv"
    statistical_tests = _safe_read_csv(stats_path) if stats_path.exists() else pd.DataFrame()
    ablation_path = root / "metrics" / "ablation_results.csv"
    ablations = _safe_read_csv(ablation_path) if ablation_path.exists() else pd.DataFrame()
    external_path = root / "metadata" / "external_contextual_results.csv"
    external = _safe_read_csv(external_path) if external_path.exists() else pd.DataFrame([{"note": "No external contextual results supplied for this run."}])

    output_paths: list[Path] = []

    table_a = pd.DataFrame(
        [
            {
                "subject_count": int(predictions["subject_id"].nunique()),
                "sample_count": int(predictions["sample_id"].nunique()),
                "rest_count": int((predictions["true_label"] == 0).sum()),
                "task_count": int((predictions["true_label"] == 1).sum()),
                "mode": metadata["mode"],
            }
        ]
    )
    output_paths.append(_write_table(root / "tables" / "table_a_dataset_characteristics.csv", table_a))

    table_b = pd.DataFrame(
        [
            {
                "outer_protocol": "LOSO",
                "inner_protocol": "GroupKFold",
                "selection_metric": "balanced_accuracy",
                "threshold_metric": "balanced_accuracy",
                "run_mode": metadata["mode"],
                "seed": metadata["seed"],
            }
        ]
    )
    output_paths.append(_write_table(root / "tables" / "table_b_evaluation_protocol_and_search_spaces.csv", table_b))

    table_c = overall_metrics.merge(ci, on="method", how="left") if not ci.empty else overall_metrics.copy()
    output_paths.append(_write_table(root / "tables" / "table_c_controlled_nested_comparison.csv", table_c))
    if not ablations.empty:
        output_paths.append(_write_table(root / "tables" / "table_d_ablation.csv", ablations))
    output_paths.append(_write_table(root / "tables" / "table_e_per_subject_results.csv", per_subject_metrics))
    if not selection.empty:
        output_paths.append(_write_table(root / "tables" / "table_f_selection_stability.csv", selection))
    if not statistical_tests.empty:
        output_paths.append(_write_table(root / "tables" / "table_g_statistical_comparison.csv", statistical_tests))
    output_paths.append(_write_table(root / "tables" / "table_h_external_contextual_results.csv", external))
    return output_paths


def generate_figures_from_run(run_root: Path | str) -> list[Path]:
    root = Path(run_root)
    metadata = json.loads((root / "metadata" / "runtime_metadata.json").read_text(encoding="utf-8"))
    predictions = _safe_read_csv(root / "predictions" / "outer_fold_predictions.csv")
    overall_metrics = _safe_read_csv(root / "metrics" / "overall_metrics.csv")
    per_subject_metrics = _safe_read_csv(root / "metrics" / "per_subject_metrics.csv")
    selection_path = root / "statistics" / "selection_frequencies.csv"
    selection = _safe_read_csv(selection_path) if selection_path.exists() else pd.DataFrame()
    anfis_path = root / "interpretability" / "anfis_interpretability.csv"
    anfis = _safe_read_csv(anfis_path) if anfis_path.exists() else pd.DataFrame()
    bn_path = root / "interpretability" / "bn_explanations.csv"
    bn = _safe_read_csv(bn_path) if bn_path.exists() else pd.DataFrame()

    output_paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.02, 0.82, "FTS -> ANFIS -> BN", fontsize=18, weight="bold")
    ax.text(0.02, 0.58, f"Run: {metadata['run_name']}", fontsize=12)
    ax.text(0.02, 0.42, f"Mode: {metadata['mode']}", fontsize=12)
    ax.text(0.02, 0.26, "Saved artifacts drive all downstream reports.", fontsize=12)
    output_paths.append(_save_figure(fig, root / "figures" / "architecture_overview.png"))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    ax.text(0.02, 0.80, "Outer LOSO", fontsize=16, weight="bold")
    ax.text(0.02, 0.56, "Inner grouped CV for candidate/threshold selection", fontsize=13)
    ax.text(0.02, 0.32, "Publication metrics from aggregated outer-test predictions only", fontsize=13)
    output_paths.append(_save_figure(fig, root / "figures" / "nested_protocol.png"))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(overall_metrics["method"], overall_metrics["balanced_accuracy"])
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Controlled Nested Comparison")
    ax.set_ylim(0.0, 1.0)
    output_paths.append(_save_figure(fig, root / "figures" / "overall_balanced_accuracy.png"))

    fig, ax = plt.subplots(figsize=(6, 4))
    for method, group in predictions.groupby("method", sort=True):
        if group["true_label"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(group["true_label"], group["score_or_probability"])
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=None, name=method).plot(ax=ax)
    ax.set_title("Aggregated ROC Curves")
    output_paths.append(_save_figure(fig, root / "figures" / "aggregated_roc_curves.png"))

    fig, axes = plt.subplots(1, max(1, len(overall_metrics)), figsize=(5 * max(1, len(overall_metrics)), 4))
    axes = np.atleast_1d(axes)
    for ax, (method, group) in zip(axes, predictions.groupby("method", sort=True), strict=False):
        cm = confusion_matrix(group["true_label"], group["prediction"], labels=[0, 1])
        ConfusionMatrixDisplay(cm, display_labels=["REST", "TASK"]).plot(ax=ax, colorbar=False)
        ax.set_title(method)
    output_paths.append(_save_figure(fig, root / "figures" / "confusion_matrices.png"))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(per_subject_metrics["subject_id"], per_subject_metrics["balanced_accuracy"])
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Per-Subject Balanced Accuracy")
    ax.set_ylim(0.0, 1.0)
    output_paths.append(_save_figure(fig, root / "figures" / "per_subject_balanced_accuracy.png"))

    if not selection.empty and "selection_count" in selection.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        x_labels = selection.iloc[:, 1].astype(str).tolist()
        ax.bar(x_labels, selection["selection_count"])
        ax.set_ylabel("Selection Count")
        ax.set_title("Selection Frequencies")
        output_paths.append(_save_figure(fig, root / "figures" / "selection_frequencies.png"))

        if "hyperparameter" in selection.columns:
            mf_rows = selection.loc[selection["hyperparameter"] == "mf_type"]
        elif "mf_type" in selection.columns:
            mf_rows = selection.rename(columns={"mf_type": "value"})
        else:
            mf_rows = pd.DataFrame()
        if not mf_rows.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(mf_rows["value"], mf_rows["selection_count"])
            ax.set_ylabel("Selection Count")
            ax.set_title("Membership-Function Selection Frequencies")
            output_paths.append(_save_figure(fig, root / "figures" / "membership_function_selection_frequencies.png"))

        if "hyperparameter" in selection.columns:
            rule_rows = selection.loc[selection["hyperparameter"] == "n_rules"]
        else:
            rule_rows = pd.DataFrame()
        if not rule_rows.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(rule_rows["value"].astype(str), rule_rows["selection_count"])
            ax.set_ylabel("Selection Count")
            ax.set_title("ANFIS Rule Selection Frequencies")
            output_paths.append(_save_figure(fig, root / "figures" / "anfis_rule_selection_frequencies.png"))

    if "threshold" in predictions.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(predictions["threshold"], bins=min(10, predictions["threshold"].nunique()))
        ax.set_title("Threshold Distribution")
        ax.set_xlabel("Selected Threshold")
        output_paths.append(_save_figure(fig, root / "figures" / "threshold_distribution.png"))

    if not anfis.empty:
        stability = anfis.groupby("signal")["reconstruction_error"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(stability["signal"], stability["reconstruction_error"])
        ax.set_ylabel("Mean Reconstruction Error")
        ax.set_title("ANFIS Interpretation Stability")
        output_paths.append(_save_figure(fig, root / "figures" / "anfis_interpretation_stability.png"))

    if not bn.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        sample = bn.iloc[0]
        ax.bar(["Posterior TASK"], [sample["posterior_task_probability"]])
        ax.axhline(float(sample.get("selected_threshold", 0.5)), color="red", linestyle="--", label="Threshold")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("BN Sample-Level Evidence Contribution")
        ax.legend()
        output_paths.append(_save_figure(fig, root / "figures" / "bn_sample_level_evidence_contribution.png"))
    return output_paths


def _write_table(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _save_figure(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()
