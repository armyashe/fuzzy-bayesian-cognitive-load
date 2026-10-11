from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

from .config import setup_seed
from .metrics import compute_metrics

STAT_FEATURE_NAMES = [
    "mean",
    "std",
    "median",
    "min",
    "max",
    "iqr",
    "slope",
    "mean_abs_diff",
]

SIGNAL_NAMES = ["temp", "hr", "gsr", "rr"]


@dataclass(slots=True)
class BaselineDevelopmentResult:
    family: str
    representation: str
    configuration_id: str
    hyperparameters: dict[str, Any]
    preprocessing_spec: dict[str, Any]
    feature_spec: dict[str, Any]
    seed: int
    k_folds: int
    grouping_protocol: str
    fold_indices: list[tuple[np.ndarray, np.ndarray]]
    oof_scores: np.ndarray
    oof_pred: np.ndarray
    oof_y: np.ndarray
    threshold: float
    threshold_metric: str
    threshold_score: float
    oof_metrics: dict[str, float]
    config_hash: str
    prediction_hash: str
    complexity: tuple

    def as_row(self) -> dict[str, Any]:
        return {
            "baseline_family": self.family,
            "input_representation": self.representation,
            "configuration_id": self.configuration_id,
            "selected_threshold": self.threshold,
            "threshold_metric": self.threshold_metric,
            "threshold_score": self.threshold_score,
            "oof_accuracy": self.oof_metrics["accuracy"],
            "oof_balanced_accuracy": self.oof_metrics["bal_acc"],
            "oof_precision": self.oof_metrics["precision"],
            "oof_recall": self.oof_metrics["recall"],
            "oof_task_f1": self.oof_metrics["f1"],
            "oof_macro_f1": self.oof_metrics["macro_f1"],
            "oof_auc": self.oof_metrics["auc"],
            "effective_parameters": json.dumps(self.hyperparameters, sort_keys=True),
            "preprocessing_spec": json.dumps(self.preprocessing_spec, sort_keys=True),
            "feature_spec": json.dumps(self.feature_spec, sort_keys=True),
            "seed": self.seed,
            "k_folds": self.k_folds,
            "grouping_protocol": self.grouping_protocol,
            "prediction_hash": self.prediction_hash,
            "config_hash": self.config_hash,
        }


@dataclass(slots=True)
class BaselineSelection:
    selected_result: BaselineDevelopmentResult
    ranked_results: list[BaselineDevelopmentResult]
    ranking_metric: str
    tie_break_rules: list[str]
    timestamp_utc: str


def _json_hash(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=float)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _array_hash(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _model_kind(family: str) -> str:
    if family in {"logreg", "stat_logreg", "flat_logreg"}:
        return "logreg"
    if family in {"svm", "stat_svm"}:
        return "svm"
    if family in {"random_forest", "stat_random_forest"}:
        return "random_forest"
    if family in {"hist_gradient_boosting", "stat_hist_gradient_boosting"}:
        return "hist_gradient_boosting"
    raise ValueError(f"Unknown family: {family}")


def extract_statistical_features(X: np.ndarray) -> np.ndarray:
    if X.ndim != 3 or X.shape[1:] != (4, 30):
        raise ValueError(f"Expected X shape (n, 4, 30), got {X.shape}")

    means = np.mean(X, axis=2)
    stds = np.std(X, axis=2)
    medians = np.median(X, axis=2)
    mins = np.min(X, axis=2)
    maxs = np.max(X, axis=2)
    q75 = np.percentile(X, 75, axis=2)
    q25 = np.percentile(X, 25, axis=2)
    iqrs = q75 - q25

    x = np.arange(X.shape[2], dtype=float)
    x_centered = x - x.mean()
    denom = np.sum(x_centered ** 2)
    slopes = np.sum(X * x_centered.reshape(1, 1, -1), axis=2) / denom

    mean_abs_diff = np.mean(np.abs(np.diff(X, axis=2)), axis=2)

    features = np.stack(
        [means, stds, medians, mins, maxs, iqrs, slopes, mean_abs_diff],
        axis=2,
    )
    return features.reshape(X.shape[0], -1)


def flatten_windows(X: np.ndarray) -> np.ndarray:
    if X.ndim != 3 or X.shape[1:] != (4, 30):
        raise ValueError(f"Expected X shape (n, 4, 30), got {X.shape}")
    return X.reshape(X.shape[0], -1)


def build_feature_matrix(X: np.ndarray, representation: str) -> tuple[np.ndarray, dict[str, Any]]:
    if representation == "statistical_32":
        return extract_statistical_features(X), {
            "representation": representation,
            "shape": 32,
            "per_signal_features": STAT_FEATURE_NAMES,
            "signals": SIGNAL_NAMES,
        }
    if representation == "flattened_120":
        return flatten_windows(X), {
            "representation": representation,
            "shape": 120,
            "signals": SIGNAL_NAMES,
            "time_steps": 30,
        }
    raise ValueError(f"Unknown representation: {representation}")


def _scale_required(family: str) -> bool:
    return _model_kind(family) in {"logreg", "svm"}


def _make_estimator(family: str, hyperparameters: dict[str, Any], seed: int):
    kind = _model_kind(family)
    if kind == "logreg":
        return LogisticRegression(
            C=hyperparameters["C"],
            class_weight=hyperparameters["class_weight"],
            solver="lbfgs",
            max_iter=5000,
            random_state=seed,
        )
    if kind == "svm":
        return SVC(
            kernel="rbf",
            C=hyperparameters["C"],
            gamma=hyperparameters["gamma"],
            class_weight="balanced",
        )
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=hyperparameters["max_depth"],
            min_samples_leaf=hyperparameters["min_samples_leaf"],
            class_weight="balanced",
            n_jobs=1,
            random_state=seed,
        )
    if kind == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=hyperparameters["learning_rate"],
            max_leaf_nodes=hyperparameters["max_leaf_nodes"],
            l2_regularization=hyperparameters["l2_regularization"],
            max_iter=200,
            random_state=seed,
        )
    raise ValueError(f"Unknown family: {family}")


def _decision_scores(estimator, X: np.ndarray, family: str) -> np.ndarray:
    if _model_kind(family) == "svm":
        return estimator.decision_function(X)
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    raise ValueError(f"Estimator for family={family} does not expose supported scores")


def _threshold_metric_fn(metric: str):
    options = {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0),
        "macro_f1": lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
    }
    if metric not in options:
        raise ValueError(f"Unsupported threshold metric: {metric}")
    return options[metric]


def find_best_score_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    metric: str = "balanced_accuracy",
    n_grid: int = 181,
) -> tuple[float, float]:
    fn = _threshold_metric_fn(metric)
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if lo == hi:
        preds = (scores >= lo).astype(int)
        return lo, round(float(fn(y_true, preds)), 4)
    thresholds = np.linspace(lo, hi, n_grid)
    best_threshold = thresholds[0]
    best_score = -1.0
    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        score = float(fn(y_true, preds))
        if score > best_score:
            best_threshold = float(threshold)
            best_score = score
    return round(best_threshold, 6), round(best_score, 4)


def _family_complexity(family: str, hyperparameters: dict[str, Any]) -> tuple:
    kind = _model_kind(family)
    if kind == "logreg":
        class_weight_rank = 0 if hyperparameters["class_weight"] is None else 1
        return (float(hyperparameters["C"]), class_weight_rank)
    if kind == "svm":
        gamma = hyperparameters["gamma"]
        gamma_rank = 0 if gamma == "scale" else 1
        gamma_value = 0.0 if gamma == "scale" else float(gamma)
        return (float(hyperparameters["C"]), gamma_rank, gamma_value)
    if kind == "random_forest":
        depth = 9999 if hyperparameters["max_depth"] is None else int(hyperparameters["max_depth"])
        return (depth, -int(hyperparameters["min_samples_leaf"]))
    if kind == "hist_gradient_boosting":
        return (
            int(hyperparameters["max_leaf_nodes"]),
            -float(hyperparameters["l2_regularization"]),
            float(hyperparameters["learning_rate"]),
        )
    raise ValueError(f"Unknown family: {family}")


def run_baseline_development(
    X_train: np.ndarray,
    y_train: np.ndarray,
    user_train: np.ndarray,
    *,
    family: str,
    representation: str,
    hyperparameters: dict[str, Any],
    configuration_id: str,
    seed: int = 42,
    k_folds: int = 5,
    threshold_metric: str = "balanced_accuracy",
) -> BaselineDevelopmentResult:
    setup_seed(seed)
    features, feature_spec = build_feature_matrix(X_train, representation)
    gkf = GroupKFold(n_splits=k_folds)
    fold_indices: list[tuple[np.ndarray, np.ndarray]] = []
    oof_scores = np.full(len(y_train), np.nan, dtype=float)
    preprocessing_spec = {
        "imputer": "median",
        "scaler": "standard" if _scale_required(family) else "none",
    }

    with threadpool_limits(limits=1):
        for fold_id, (tr_idx, va_idx) in enumerate(gkf.split(features, y_train, user_train)):
            fold_indices.append((tr_idx, va_idx))
            X_tr = features[tr_idx]
            X_va = features[va_idx]
            y_tr = y_train[tr_idx]

            steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
            if _scale_required(family):
                steps.append(("scaler", StandardScaler()))
            steps.append(("model", _make_estimator(family, hyperparameters, seed + fold_id)))
            pipeline = Pipeline(steps)
            pipeline.fit(X_tr, y_tr)
            oof_scores[va_idx] = _decision_scores(pipeline, X_va, family)

    best_threshold, threshold_score = find_best_score_threshold(
        y_train,
        oof_scores,
        metric=threshold_metric,
    )
    oof_pred = (oof_scores >= best_threshold).astype(int)
    oof_metrics = compute_metrics(y_train, oof_scores, best_threshold)
    config_payload = {
        "family": family,
        "representation": representation,
        "hyperparameters": hyperparameters,
        "preprocessing_spec": preprocessing_spec,
        "feature_spec": feature_spec,
        "seed": seed,
        "k_folds": k_folds,
    }
    return BaselineDevelopmentResult(
        family=family,
        representation=representation,
        configuration_id=configuration_id,
        hyperparameters=dict(hyperparameters),
        preprocessing_spec=preprocessing_spec,
        feature_spec=feature_spec,
        seed=seed,
        k_folds=k_folds,
        grouping_protocol="GroupKFold grouped by training subject IDs only",
        fold_indices=fold_indices,
        oof_scores=oof_scores,
        oof_pred=oof_pred,
        oof_y=y_train.copy(),
        threshold=best_threshold,
        threshold_metric=threshold_metric,
        threshold_score=threshold_score,
        oof_metrics=oof_metrics,
        config_hash=_json_hash(config_payload),
        prediction_hash=_array_hash(oof_scores),
        complexity=_family_complexity(family, hyperparameters),
    )


def select_best_baseline(results: list[BaselineDevelopmentResult]) -> BaselineSelection:
    if not results:
        raise ValueError("results must not be empty")
    ranked = sorted(
        results,
        key=lambda result: (
            -float(result.oof_metrics["bal_acc"]),
            -float(result.oof_metrics["macro_f1"]),
            -float(result.oof_metrics["auc"]),
            result.complexity,
            result.configuration_id,
        ),
    )
    return BaselineSelection(
        selected_result=ranked[0],
        ranked_results=ranked,
        ranking_metric="oof_balanced_accuracy",
        tie_break_rules=[
            "oof_macro_f1",
            "oof_auc",
            "lower_model_complexity",
            "configuration_id_lexicographic",
        ],
        timestamp_utc=_utc_timestamp(),
    )


def baseline_lock_payload(
    result: BaselineDevelopmentResult,
    *,
    comparison_manifest_hash: str,
) -> dict[str, Any]:
    payload = {
        "protocol_version": "publication_protocol_v2",
        "schema_version": 1,
        "baseline_family": result.family,
        "configuration_id": result.configuration_id,
        "input_representation": result.representation,
        "feature_definitions": result.feature_spec,
        "selected_hyperparameters": result.hyperparameters,
        "preprocessing_specification": result.preprocessing_spec,
        "oof_metrics": result.oof_metrics,
        "oof_threshold": result.threshold,
        "threshold_metric": result.threshold_metric,
        "seed": result.seed,
        "k_folds": result.k_folds,
        "grouping_protocol": result.grouping_protocol,
        "comparison_manifest_hash": comparison_manifest_hash,
        "selected_using": "training_oof_only",
        "threshold_source": "training_oof_only",
        "final_test_evaluated": False,
        "prediction_hash": result.prediction_hash,
    }
    payload["config_hash"] = result.config_hash
    payload["lock_hash"] = _json_hash({k: v for k, v in payload.items() if k != "lock_hash"})
    return payload


def write_baseline_lock(
    result: BaselineDevelopmentResult,
    *,
    comparison_manifest_hash: str,
    output_path: Path | str,
) -> dict[str, Any]:
    payload = baseline_lock_payload(
        result,
        comparison_manifest_hash=comparison_manifest_hash,
    )
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
