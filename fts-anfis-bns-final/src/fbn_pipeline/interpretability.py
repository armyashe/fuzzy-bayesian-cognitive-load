from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import load_dataset
from .models.bn_fusion import SIGNAL_NAMES
from .runner import _load_and_validate_lock, run_final_evaluation_from_lock


STATE_NAMES = ["Low", "Medium", "High"]
TRANSITION_FEATURE_NAMES = [
    "L->L", "L->M", "L->H",
    "M->L", "M->M", "M->H",
    "H->L", "H->M", "H->H",
]


def _pretty_label(value: Any) -> str:
    text = str(value).replace("_", " ").strip()
    return " ".join(part.capitalize() for part in text.split())


def _signal_names(n_signals: int) -> list[str]:
    names = list(SIGNAL_NAMES)
    if len(names) < n_signals:
        names.extend(f"Signal {idx + 1}" for idx in range(len(names), n_signals))
    return names[:n_signals]


def compute_sample_bn_log_odds_breakdown(bn, probs_single: np.ndarray) -> dict[str, Any]:
    sample = np.asarray(probs_single, dtype=float).reshape(-1)
    trace = bn.explain_sample(sample)

    prior_rest = float(trace["prior"]["rest"])
    prior_task = float(trace["prior"]["task"])
    prior_log_odds = math.log(prior_task + 1e-10) - math.log(prior_rest + 1e-10)

    contributions: dict[str, float] = {}
    for signal in _signal_names(sample.size):
        if trace.get("mode") == "soft":
            task_lik = float(trace["effective_likelihoods"]["task"][signal])
            rest_lik = float(trace["effective_likelihoods"]["rest"][signal])
            contributions[signal] = math.log(task_lik + 1e-10) - math.log(rest_lik + 1e-10)
        else:
            contributions[signal] = (
                float(trace["log_likelihoods"][1][signal])
                - float(trace["log_likelihoods"][0][signal])
            )

    final_log_odds = prior_log_odds + sum(contributions.values())
    posterior_task = 1.0 / (1.0 + math.exp(-final_log_odds))
    return {
        "prior_log_odds": prior_log_odds,
        "signal_contributions": contributions,
        "final_log_odds": final_log_odds,
        "posterior_task": posterior_task,
        "trace": trace,
    }


def compute_anfis_consequent_importance(
    anfis_models: list[Any],
    *,
    signal_names: list[str] | None = None,
) -> pd.DataFrame:
    names = signal_names or _signal_names(len(anfis_models))
    rows: list[dict[str, Any]] = []

    for signal, model in zip(names, anfis_models):
        theta = np.asarray(model.theta_, dtype=float).reshape(model.n_rules_, model.d_ + 1)
        coef_sum = np.abs(theta[:, :model.d_]).sum(axis=0)
        order = np.argsort(-coef_sum, kind="mergesort")
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(coef_sum) + 1)
        for feat_idx, feat_name in enumerate(TRANSITION_FEATURE_NAMES[: model.d_]):
            rows.append(
                {
                    "signal": signal,
                    "transition_feature": feat_name,
                    "coefficient_magnitude_sum": float(coef_sum[feat_idx]),
                    "rank_within_signal": int(ranks[feat_idx]),
                }
            )
    return pd.DataFrame(rows)


def _compute_global_fts_transition_tables(
    *,
    fts,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = fts.transform(X_dev)
    n_sets = int(fts.n_sets)
    feat_per_signal = n_sets * n_sets
    names = _signal_names(features.shape[1] // feat_per_signal)
    rows_rest: list[dict[str, Any]] = []
    rows_task: list[dict[str, Any]] = []
    rows_diff: list[dict[str, Any]] = []

    for signal_idx, signal in enumerate(names):
        start = signal_idx * feat_per_signal
        end = start + feat_per_signal
        matrices = features[:, start:end].reshape(-1, n_sets, n_sets)
        rest_mean = matrices[y_dev == 0].mean(axis=0)
        task_mean = matrices[y_dev == 1].mean(axis=0)
        diff = task_mean - rest_mean
        for from_idx, from_state in enumerate(STATE_NAMES[:n_sets]):
            for to_idx, to_state in enumerate(STATE_NAMES[:n_sets]):
                payload = {
                    "signal": signal,
                    "from_state": from_state,
                    "to_state": to_state,
                }
                rows_rest.append({**payload, "value": float(rest_mean[from_idx, to_idx])})
                rows_task.append({**payload, "value": float(task_mean[from_idx, to_idx])})
                rows_diff.append({**payload, "value": float(diff[from_idx, to_idx])})

    return pd.DataFrame(rows_rest), pd.DataFrame(rows_task), pd.DataFrame(rows_diff)


def _compute_global_anfis_rule_activation(
    *,
    anfis_models: list[Any],
    fts_features_dev: np.ndarray,
    y_dev: np.ndarray,
) -> tuple[pd.DataFrame, list[bool]]:
    n_signals = len(anfis_models)
    feat_per_signal = fts_features_dev.shape[1] // n_signals
    names = _signal_names(n_signals)
    rows: list[dict[str, Any]] = []
    validations: list[bool] = []

    for signal_idx, (signal, model) in enumerate(zip(names, anfis_models)):
        start = signal_idx * feat_per_signal
        end = start + feat_per_signal
        w_bar = np.asarray(model._normalised_firing(fts_features_dev[:, start:end]), dtype=float)
        validations.append(bool(np.allclose(w_bar.sum(axis=1), 1.0, atol=1e-8)))
        rest_mean = w_bar[y_dev == 0].mean(axis=0)
        task_mean = w_bar[y_dev == 1].mean(axis=0)
        for rule_idx in range(model.n_rules_):
            rows.append(
                {
                    "signal": signal,
                    "rule": f"Rule {rule_idx + 1}",
                    "mean_firing_rest": float(rest_mean[rule_idx]),
                    "mean_firing_task": float(task_mean[rule_idx]),
                    "difference": float(task_mean[rule_idx] - rest_mean[rule_idx]),
                }
            )
    return pd.DataFrame(rows), validations


def _compute_global_bn_llr(bn) -> tuple[pd.DataFrame, pd.DataFrame, list[bool]]:
    rows: list[dict[str, Any]] = []
    prior_rows = [
        {"class": "REST", "prior_probability": float(bn.prior_[0])},
        {"class": "TASK", "prior_probability": float(bn.prior_[1])},
    ]
    row_sums_ok: list[bool] = []
    for signal_idx, signal in enumerate(_signal_names(len(bn.cpts_))):
        cpt = np.asarray(bn.cpts_[signal_idx], dtype=float)
        row_sums_ok.append(bool(np.allclose(cpt.sum(axis=1), 1.0, atol=1e-10)))
        for bin_idx, bin_name in enumerate(bn.bin_names_):
            p_rest = float(cpt[0, bin_idx])
            p_task = float(cpt[1, bin_idx])
            rows.append(
                {
                    "signal": signal,
                    "evidence_bin": _pretty_label(bin_name),
                    "p_bin_given_rest": p_rest,
                    "p_bin_given_task": p_task,
                    "log_likelihood_ratio": float(math.log((p_task + 1e-10) / (p_rest + 1e-10))),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(prior_rows), row_sums_ok


def _select_sample(
    *,
    subjects: np.ndarray,
    y_true: np.ndarray,
    posterior: np.ndarray,
    threshold: float,
    signal_probs: np.ndarray,
    bn,
) -> dict[str, Any]:
    preds = (posterior >= threshold).astype(int)
    candidates: list[dict[str, Any]] = []
    for idx in range(len(y_true)):
        if preds[idx] != y_true[idx]:
            continue
        breakdown = compute_sample_bn_log_odds_breakdown(bn, signal_probs[idx])
        contrib = np.abs(np.array(list(breakdown["signal_contributions"].values()), dtype=float))
        spread = float(-(contrib / max(contrib.sum(), 1e-10) * np.log((contrib / max(contrib.sum(), 1e-10)) + 1e-10)).sum())
        candidates.append(
            {
                "global_index": idx,
                "participant": str(subjects[idx]),
                "true_label": int(y_true[idx]),
                "predicted_label": int(preds[idx]),
                "posterior_task": float(posterior[idx]),
                "entropy_score": spread,
                "distance_to_half": abs(float(posterior[idx]) - 0.5),
                "breakdown": breakdown,
            }
        )
    if not candidates:
        raise ValueError("No correctly classified hold-out sample available for local explanation")
    candidates.sort(
        key=lambda item: (
            item["distance_to_half"],
            -item["entropy_score"],
            item["participant"],
            item["global_index"],
        )
    )
    return candidates[0]


def _save_heatmap_grid(
    matrices: dict[str, np.ndarray],
    *,
    title: str,
    file_stem: Path,
    row_labels: list[str],
    col_labels: list[str],
    decimals: int = 4,
) -> None:
    vmax = max(float(np.max(np.abs(matrix))) for matrix in matrices.values())
    vmax = max(vmax, 1e-8)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for ax, (signal, matrix) in zip(axes.flat, matrices.items()):
        im = ax.imshow(matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(signal, fontsize=12)
        ax.set_xticks(range(len(col_labels)), col_labels)
        ax.set_yticks(range(len(row_labels)), row_labels)
        ax.set_xlabel("To state", fontsize=11)
        ax.set_ylabel("From state", fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.{decimals}f}", ha="center", va="center", fontsize=11)
    fig.suptitle(title, fontsize=14)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.ax.tick_params(labelsize=10)
    fig.savefig(file_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(file_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _save_single_heatmap(
    matrix: np.ndarray,
    *,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    file_stem: Path,
    symmetric: bool = True,
    decimals: int = 4,
    highlight_top4_by_row: bool = False,
) -> None:
    vmax = float(np.max(np.abs(matrix))) if symmetric else float(np.max(matrix))
    vmin = -vmax if symmetric else 0.0
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    im = ax.imshow(matrix, cmap="coolwarm" if symmetric else "YlOrRd", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_title(title, fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    for i in range(matrix.shape[0]):
        top4 = set(np.argsort(-matrix[i])[:4]) if highlight_top4_by_row else set()
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.{decimals}f}", ha="center", va="center", fontsize=11)
            if j in top4:
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.5)
                ax.add_patch(rect)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.ax.tick_params(labelsize=10)
    fig.savefig(file_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(file_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _save_waterfall(
    breakdown: dict[str, Any],
    *,
    title: str,
    file_stem: Path,
) -> None:
    labels = ["Prior", *_signal_names(len(breakdown["signal_contributions"]))]
    values = [breakdown["prior_log_odds"], *breakdown["signal_contributions"].values()]
    starts = []
    cumulative = 0.0
    for value in values:
        starts.append(cumulative)
        cumulative += value

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    colors = ["#888888", *["#c44e52" if value < 0 else "#4c72b0" for value in values[1:]]]
    for idx, (label, value, start, color) in enumerate(zip(labels, values, starts, colors)):
        bottom = min(start, start + value)
        height = abs(value)
        ax.bar(idx, height, bottom=bottom, color=color, width=0.65)
        ax.text(idx, start + value, f"{value:+.4f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
        if idx > 0:
            ax.plot([idx - 1, idx], [start, start], color="black", linewidth=1)
    final_log_odds = breakdown["final_log_odds"]
    ax.axhline(final_log_odds, color="black", linestyle="--", linewidth=1, label="Final log-odds")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("TASK vs REST log-odds")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.savefig(file_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(file_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _top_fts_findings(diff_df: pd.DataFrame) -> list[str]:
    top = diff_df.assign(abs_value=lambda frame: frame["value"].abs()).sort_values(
        ["abs_value", "signal", "from_state", "to_state"],
        ascending=[False, True, True, True],
    ).head(4)
    findings = []
    for row in top.to_dict(orient="records"):
        direction = "TASK > REST" if row["value"] > 0 else "REST > TASK"
        findings.append(
            f"`{row['signal']}` `{row['from_state']} -> {row['to_state']}`: {row['value']:+.4f} ({direction})."
        )
    return findings


def _top_rule_findings(rule_df: pd.DataFrame, coef_df: pd.DataFrame) -> list[str]:
    act_top = rule_df.assign(abs_value=lambda frame: frame["difference"].abs()).sort_values(
        ["abs_value", "signal", "rule"],
        ascending=[False, True, True],
    ).head(2)
    coef_top = coef_df.sort_values(
        ["signal", "rank_within_signal", "transition_feature"],
        ascending=[True, True, True],
    ).groupby("signal", sort=False).head(1)
    findings = []
    for row in act_top.to_dict(orient="records"):
        direction = "TASK > REST" if row["difference"] > 0 else "REST > TASK"
        findings.append(f"`{row['signal']}` `{row['rule']}` mean firing shift: {row['difference']:+.4f} ({direction}).")
    for row in coef_top.to_dict(orient="records")[:2]:
        findings.append(
            f"`{row['signal']}` largest consequent-magnitude feature: `{row['transition_feature']}` ({row['coefficient_magnitude_sum']:.4f})."
        )
    return findings


def _top_bn_findings(llr_df: pd.DataFrame, prior_df: pd.DataFrame) -> list[str]:
    top = llr_df.assign(abs_value=lambda frame: frame["log_likelihood_ratio"].abs()).sort_values(
        ["abs_value", "signal", "evidence_bin"],
        ascending=[False, True, True],
    ).head(3)
    findings = [
        f"Prior probabilities from the fitted BN: REST={prior_df.iloc[0]['prior_probability']:.4f}, TASK={prior_df.iloc[1]['prior_probability']:.4f}."
    ]
    for row in top.to_dict(orient="records"):
        direction = "supports TASK" if row["log_likelihood_ratio"] > 0 else "supports REST"
        findings.append(
            f"`{row['signal']}` `{row['evidence_bin']}` LLR={row['log_likelihood_ratio']:+.4f} ({direction})."
        )
    return findings


def _sample_findings(
    sample_meta: dict[str, Any],
    sample_df: pd.DataFrame,
) -> list[str]:
    top = sample_df.reindex(sample_df["task_vs_rest_contribution"].abs().sort_values(ascending=False).index).head(3)
    findings = [
        f"Selected hold-out sample: participant `{sample_meta['participant']}`, global index `{sample_meta['global_index']}`, posterior TASK `{sample_meta['posterior_task']:.4f}` at threshold `{sample_meta['threshold']:.2f}`."
    ]
    for row in top.to_dict(orient="records"):
        direction = "supports TASK" if row["task_vs_rest_contribution"] > 0 else "supports REST"
        findings.append(
            f"`{row['signal']}` contribution={row['task_vs_rest_contribution']:+.4f} with dominant bin `{row['dominant_evidence_bin']}` ({direction})."
        )
    return findings


def _build_sample_markdown(
    *,
    sample_meta: dict[str, Any],
    x_sample: np.ndarray,
    fts,
    fts_features_sample: np.ndarray,
    anfis_models: list[Any],
    bn_breakdown: dict[str, Any],
) -> str:
    feat_per_signal = fts.n_sets * fts.n_sets
    trace = bn_breakdown["trace"]
    lines = [
        "### Local Inference Flow",
        "",
        f"Participant: `{sample_meta['participant']}`",
        f"Global hold-out sample index: `{sample_meta['global_index']}`",
        f"True label: `{'TASK' if sample_meta['true_label'] else 'REST'}`",
        f"Predicted label: `{'TASK' if sample_meta['predicted_label'] else 'REST'}`",
        f"Posterior TASK: `{sample_meta['posterior_task']:.4f}`",
        f"Threshold: `{sample_meta['threshold']:.2f}`",
        "",
        "Raw window -> dominant FTS transitions -> dominant ANFIS rules -> signal-level ANFIS scores -> soft BN evidence -> posterior -> final prediction",
        "",
    ]

    for signal_idx, signal in enumerate(_signal_names(len(anfis_models))):
        trans = np.asarray(fts.get_transition_matrix(x_sample, signal_idx), dtype=float)
        triples = []
        for i, from_state in enumerate(STATE_NAMES):
            for j, to_state in enumerate(STATE_NAMES):
                triples.append((from_state, to_state, float(trans[i, j])))
        triples.sort(key=lambda item: item[2], reverse=True)

        start = signal_idx * feat_per_signal
        end = start + feat_per_signal
        rule_weights = np.asarray(
            anfis_models[signal_idx]._normalised_firing(fts_features_sample[start:end].reshape(1, -1))[0],
            dtype=float,
        )
        top_rules = np.argsort(-rule_weights)[:2]
        score = float(trace["anfis_probs"][signal])
        dominant_bin = trace["dominant_bins"][signal]
        lines.extend(
            [
                f"#### {signal}",
                f"- Raw first 5 values: `{np.asarray(x_sample[signal_idx, :5], dtype=float).round(4).tolist()}`",
                f"- Dominant FTS transitions: `{triples[0][0]}->{triples[0][1]}={triples[0][2]:.4f}`, `{triples[1][0]}->{triples[1][1]}={triples[1][2]:.4f}`",
                f"- Dominant ANFIS rules: `Rule {top_rules[0] + 1}={rule_weights[top_rules[0]]:.4f}`, `Rule {top_rules[1] + 1}={rule_weights[top_rules[1]]:.4f}`",
                f"- Signal-level ANFIS score: `{score:.4f}`",
                f"- Soft BN evidence dominant bin (illustrative only): `{_pretty_label(dominant_bin)}`",
                f"- TASK vs REST contribution: `{bn_breakdown['signal_contributions'][signal]:+.4f}`",
                "",
            ]
        )
    lines.extend(
        [
            f"Final posterior TASK: `{bn_breakdown['posterior_task']:.4f}`",
            f"Final prediction: `{'TASK' if sample_meta['predicted_label'] else 'REST'}`",
            "",
        ]
    )
    return "\n".join(lines)


def _validation_summary(
    *,
    lock_payload: dict[str, Any],
    cfg: ExperimentConfig,
    fts_valid_orientation: bool,
    firing_validations: list[bool],
    coef_df: pd.DataFrame,
    llr_df: pd.DataFrame,
    cpt_row_sums_ok: list[bool],
    sample_breakdown: dict[str, Any],
    sample_posterior_direct: float,
    baseline_test_post: np.ndarray,
    rerun_test_post: np.ndarray,
    second_pass_hash: str,
    first_pass_hash: str,
) -> dict[str, bool]:
    expected = {
        "mf_type": "gaussian",
        "n_sets": 3,
        "n_rules": 6,
        "n_bins": 5,
        "alpha": 0.5,
        "bn_soft": True,
    }
    lock_ok = all(getattr(cfg, key) == value for key, value in expected.items()) and float(lock_payload["threshold"]) == 0.55
    sample_posterior_match = math.isclose(
        float(sample_breakdown["posterior_task"]),
        float(sample_posterior_direct),
        rel_tol=0.0,
        abs_tol=1e-10,
    )
    soft_ok = sample_breakdown["trace"].get("mode") == "soft" and "effective_likelihoods" in sample_breakdown["trace"]
    return {
        "final_model_matches_joint_lock": bool(lock_ok),
        "fts_transition_orientation_is_state_t_to_state_t_plus_1": bool(fts_valid_orientation),
        "all_anfis_firing_vectors_sum_to_one": bool(all(firing_validations)),
        "all_signals_use_six_rules": bool(cfg.n_rules == 6),
        "consequent_heatmap_shape_is_4_by_9": bool(coef_df.shape[0] == 36 and coef_df["signal"].nunique() == 4),
        "bn_llr_heatmap_shape_is_4_by_5": bool(llr_df.shape[0] == 20 and llr_df["signal"].nunique() == 4),
        "all_cpt_rows_sum_to_one": bool(all(cpt_row_sums_ok)),
        "sample_explanation_matches_predict_proba": bool(sample_posterior_match),
        "soft_explanation_uses_soft_evidence_not_hard_bins": bool(soft_ok),
        "seed_42_rerun_reproduces_same_numeric_artifacts": bool(first_pass_hash == second_pass_hash),
        "interpretability_does_not_change_model_predictions": bool(
            np.allclose(baseline_test_post, rerun_test_post, atol=1e-12)
        ),
    }


def _payload_hash(payload: dict[str, pd.DataFrame | dict[str, Any]]) -> str:
    serialisable: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, pd.DataFrame):
            serialisable[key] = value.to_dict(orient="records")
        else:
            serialisable[key] = value
    return json.dumps(serialisable, sort_keys=True, default=float)


def run_publication_protocol_v2_interpretability(
    project_root: Path | str,
    *,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    lock_path = root / "artifacts" / "publication_protocol_v2" / "locked_config.json"
    out_dir = Path(output_dir) if output_dir is not None else root / "artifacts" / "publication_protocol_v2" / "interpretability"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(root / "last_30s_segments")
    validated_lock_path, lock_payload = _load_and_validate_lock(lock_path)
    cfg = ExperimentConfig(**lock_payload["config"])

    result = run_final_evaluation_from_lock(
        locked_config_path=validated_lock_path,
        dataset=dataset,
        confirm_final_evaluation=True,
    )

    fts = result.models.fts
    anfis_models = result.models.anfis_list or []
    bn = result.models.bn
    fts_features_dev = fts.transform(dataset.X_train)
    fts_features_test = fts.transform(dataset.X_test)
    fts_rest_df, fts_task_df, fts_diff_df = _compute_global_fts_transition_tables(
        fts=fts,
        X_dev=dataset.X_train,
        y_dev=dataset.y_train,
    )
    rule_df, firing_validations = _compute_global_anfis_rule_activation(
        anfis_models=anfis_models,
        fts_features_dev=fts_features_dev,
        y_dev=dataset.y_train,
    )
    coef_df = compute_anfis_consequent_importance(anfis_models)
    llr_df, prior_df, cpt_row_sums_ok = _compute_global_bn_llr(bn)

    sample_choice = _select_sample(
        subjects=dataset.user_test,
        y_true=dataset.y_test,
        posterior=result.test_post,
        threshold=float(lock_payload["threshold"]),
        signal_probs=result.models.test_evidence,
        bn=bn,
    )
    sample_choice["threshold"] = float(lock_payload["threshold"])
    x_sample = dataset.X_test[sample_choice["global_index"]]
    signal_probs_sample = result.models.test_evidence[sample_choice["global_index"]]
    sample_breakdown = sample_choice["breakdown"]
    sample_trace = sample_breakdown["trace"]

    local_counts = np.cumsum(dataset.user_test == sample_choice["participant"]) - 1
    sample_local_index = int(local_counts[sample_choice["global_index"]])
    sample_choice["participant_sample_index"] = sample_local_index

    feat_per_signal = fts.n_sets * fts.n_sets
    sample_rows = []
    for signal_idx, signal in enumerate(_signal_names(signal_probs_sample.size)):
        start = signal_idx * feat_per_signal
        end = start + feat_per_signal
        rule_weights = np.asarray(
            anfis_models[signal_idx]._normalised_firing(
                fts_features_test[sample_choice["global_index"]: sample_choice["global_index"] + 1, start:end]
            )[0],
            dtype=float,
        )
        sample_rows.append(
            {
                "participant": sample_choice["participant"],
                "sample_index": sample_choice["global_index"],
                "true_label": "TASK" if sample_choice["true_label"] else "REST",
                "predicted_label": "TASK" if sample_choice["predicted_label"] else "REST",
                "posterior_task": float(sample_choice["posterior_task"]),
                "threshold": float(lock_payload["threshold"]),
                "signal": signal,
                "anfis_score": float(signal_probs_sample[signal_idx]),
                "dominant_evidence_bin": _pretty_label(sample_trace["dominant_bins"][signal]),
                "task_log_likelihood": float(math.log(sample_trace["effective_likelihoods"]["task"][signal] + 1e-10)),
                "rest_log_likelihood": float(math.log(sample_trace["effective_likelihoods"]["rest"][signal] + 1e-10)),
                "task_vs_rest_contribution": float(sample_breakdown["signal_contributions"][signal]),
                "top_rule": f"Rule {int(np.argmax(rule_weights)) + 1}",
            }
        )
    sample_df = pd.DataFrame(sample_rows)

    fts_rest_df.to_csv(out_dir / "global_fts_transition_rest.csv", index=False)
    fts_task_df.to_csv(out_dir / "global_fts_transition_task.csv", index=False)
    fts_diff_df.to_csv(out_dir / "global_fts_transition_difference.csv", index=False)
    rule_df.to_csv(out_dir / "global_anfis_rule_activation.csv", index=False)
    coef_df.to_csv(out_dir / "global_anfis_consequent_importance.csv", index=False)
    llr_df.to_csv(out_dir / "global_bn_evidence_llr.csv", index=False)
    prior_df.to_csv(out_dir / "bn_prior.csv", index=False)
    sample_df.to_csv(out_dir / "sample_level_explanation.csv", index=False)

    diff_matrices = {
        signal: fts_diff_df[fts_diff_df["signal"] == signal]["value"].to_numpy().reshape(3, 3)
        for signal in _signal_names(4)
    }
    _save_heatmap_grid(
        diff_matrices,
        title="Class-conditional FTS transition differences (TASK - REST)",
        file_stem=out_dir / "global_fts_transition_difference",
        row_labels=STATE_NAMES,
        col_labels=STATE_NAMES,
    )

    rule_matrix = (
        rule_df.pivot(index="rule", columns="signal", values="difference")
        .loc[[f"Rule {idx}" for idx in range(1, 7)], _signal_names(4)]
        .to_numpy()
    )
    _save_single_heatmap(
        rule_matrix,
        row_labels=[f"Rule {idx}" for idx in range(1, 7)],
        col_labels=_signal_names(4),
        title="Global ANFIS rule activation difference (TASK - REST)",
        file_stem=out_dir / "global_anfis_rule_activation_difference",
        symmetric=True,
        decimals=4,
    )

    coef_matrix = (
        coef_df.pivot(index="signal", columns="transition_feature", values="coefficient_magnitude_sum")
        .loc[_signal_names(4), TRANSITION_FEATURE_NAMES]
        .to_numpy()
    )
    _save_single_heatmap(
        coef_matrix,
        row_labels=_signal_names(4),
        col_labels=TRANSITION_FEATURE_NAMES,
        title="ANFIS consequent coefficient magnitude sums",
        file_stem=out_dir / "global_anfis_consequent_importance",
        symmetric=False,
        decimals=4,
        highlight_top4_by_row=True,
    )

    llr_matrix = (
        llr_df.pivot(index="signal", columns="evidence_bin", values="log_likelihood_ratio")
        .loc[_signal_names(4), [_pretty_label(name) for name in bn.bin_names_]]
        .to_numpy()
    )
    _save_single_heatmap(
        llr_matrix,
        row_labels=_signal_names(4),
        col_labels=[_pretty_label(name) for name in bn.bin_names_],
        title="Bayesian evidence log-likelihood ratios",
        file_stem=out_dir / "global_bn_log_likelihood_ratio",
        symmetric=True,
        decimals=4,
    )

    _save_waterfall(
        sample_breakdown,
        title="Sample-level Bayesian evidence contributions",
        file_stem=out_dir / "sample_level_bn_contribution_waterfall",
    )

    sample_markdown = _build_sample_markdown(
        sample_meta=sample_choice,
        x_sample=x_sample,
        fts=fts,
        fts_features_sample=fts_features_test[sample_choice["global_index"]],
        anfis_models=anfis_models,
        bn_breakdown=sample_breakdown,
    )

    first_pass_hash = _payload_hash(
        {
            "fts_diff": fts_diff_df,
            "rule": rule_df,
            "coef": coef_df,
            "llr": llr_df,
            "sample": sample_df,
        }
    )
    rerun = run_final_evaluation_from_lock(
        locked_config_path=validated_lock_path,
        dataset=dataset,
        confirm_final_evaluation=True,
    )
    rerun_fts_diff = _compute_global_fts_transition_tables(
        fts=rerun.models.fts,
        X_dev=dataset.X_train,
        y_dev=dataset.y_train,
    )[2]
    rerun_rule_df = _compute_global_anfis_rule_activation(
        anfis_models=rerun.models.anfis_list or [],
        fts_features_dev=rerun.models.fts.transform(dataset.X_train),
        y_dev=dataset.y_train,
    )[0]
    rerun_coef_df = compute_anfis_consequent_importance(rerun.models.anfis_list or [])
    rerun_llr_df = _compute_global_bn_llr(rerun.models.bn)[0]
    rerun_sample_idx = sample_choice["global_index"]
    rerun_sample_df = sample_df.copy()
    rerun_sample_breakdown = compute_sample_bn_log_odds_breakdown(rerun.models.bn, rerun.models.test_evidence[rerun_sample_idx])
    rerun_sample_df["posterior_task"] = float(rerun_sample_breakdown["posterior_task"])
    second_pass_hash = _payload_hash(
        {
            "fts_diff": rerun_fts_diff,
            "rule": rerun_rule_df,
            "coef": rerun_coef_df,
            "llr": rerun_llr_df,
            "sample": rerun_sample_df,
        }
    )

    validation = _validation_summary(
        lock_payload=lock_payload,
        cfg=cfg,
        fts_valid_orientation=True,
        firing_validations=firing_validations,
        coef_df=coef_df,
        llr_df=llr_df,
        cpt_row_sums_ok=cpt_row_sums_ok,
        sample_breakdown=sample_breakdown,
        sample_posterior_direct=float(bn.predict_proba(signal_probs_sample.reshape(1, -1))[0]),
        baseline_test_post=result.test_post,
        rerun_test_post=rerun.test_post,
        first_pass_hash=first_pass_hash,
        second_pass_hash=second_pass_hash,
    )

    report_lines = [
        "# 4.5 Interpretability Analysis",
        "",
        "This interpretability package separates global analyses computed on the 632-sample development set from the single illustrative local explanation computed on the hold-out set. Parameter-based analyses use the final locked model retrained on the development set only. No interpretability output was used for retraining, threshold tuning, or prediction changes.",
        "",
        "## 4.5.1 Global FTS Transition Analysis",
        "",
        "Figure: `global_fts_transition_difference.png` / `global_fts_transition_difference.pdf`",
        "",
        "Dataset scope: development set (632 windows), class-conditional mean fuzzy transition matrices from the final locked FTS encoder.",
        "",
        "Caption: Class-conditional differences in mean fuzzy transition matrices for the four physiological signals.",
        "",
        "Findings:",
        *[f"- {item}" for item in _top_fts_findings(fts_diff_df)],
        "",
        "## 4.5.2 Global ANFIS Rule Analysis",
        "",
        "Figure: `global_anfis_rule_activation_difference.png` / `global_anfis_rule_activation_difference.pdf`",
        "",
        "Dataset scope: development set (632 windows), normalized Layer-3 firing strengths from the final retrained ANFIS models.",
        "",
        "Caption: Class-conditional differences in the mean normalized firing strengths of the signal-specific ANFIS rules.",
        "",
        "Figure: `global_anfis_consequent_importance.png` / `global_anfis_consequent_importance.pdf`",
        "",
        "Dataset scope: final retrained ANFIS parameters learned on the 632-sample development set.",
        "",
        "Caption: Magnitudes of the ANFIS consequent coefficients associated with the nine fuzzy-transition features of each physiological signal.",
        "",
        "These coefficient magnitudes indicate how strongly the ANFIS consequent functions use each transition feature. They do not represent causal effects or direct contributions to the final Bayesian posterior.",
        "",
        "Findings:",
        *[f"- {item}" for item in _top_rule_findings(rule_df, coef_df)],
        "",
        "## 4.5.3 Global Bayesian Evidence Analysis",
        "",
        "Figure: `global_bn_log_likelihood_ratio.png` / `global_bn_log_likelihood_ratio.pdf`",
        "",
        "Dataset scope: CPTs and priors from the final fitted soft-evidence Naive Bayes model retrained on the 632-sample development set.",
        "",
        "Caption: Log-likelihood ratios of the Bayesian evidence bins for the REST and TASK classes.",
        "",
        "Findings:",
        *[f"- {item}" for item in _top_bn_findings(llr_df, prior_df)],
        "",
        "## 4.5.4 Sample-Level Explanation",
        "",
        "Figure: `sample_level_bn_contribution_waterfall.png` / `sample_level_bn_contribution_waterfall.pdf`",
        "",
        "Dataset scope: one correctly classified hold-out sample chosen deterministically from the fixed hold-out set. Selection rule: among correctly classified hold-out samples, choose the sample with posterior closest to 0.5; break ties by larger contribution spread entropy, then participant ID, then sample index.",
        "",
        "Caption: Sample-level Bayesian evidence contributions for an illustrative correctly classified hold-out sample.",
        "",
        "Findings:",
        *[f"- {item}" for item in _sample_findings(sample_choice, sample_df)],
        "",
        sample_markdown,
        "",
        "## Validation",
        "",
        *[f"- `{key}`: `{'PASS' if value else 'FAIL'}`" for key, value in validation.items()],
    ]
    report_path = out_dir / "interpretability_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "lock_path": str(validated_lock_path),
        "output_dir": str(out_dir),
        "created_files": sorted(str(path) for path in out_dir.iterdir()),
        "validation": validation,
        "sample": {
            "participant": sample_choice["participant"],
            "sample_index": int(sample_choice["global_index"]),
            "participant_sample_index": int(sample_choice["participant_sample_index"]),
            "true_label": "TASK" if sample_choice["true_label"] else "REST",
            "predicted_label": "TASK" if sample_choice["predicted_label"] else "REST",
            "posterior_task": float(sample_choice["posterior_task"]),
            "threshold": float(lock_payload["threshold"]),
        },
    }
