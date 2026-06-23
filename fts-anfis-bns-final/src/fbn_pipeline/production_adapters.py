from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .models.anfis_tsk import ANFIS_TSK
from .models.fts_encoder import FTSEncoder
from .nested_evaluation import CandidateSpec, NestedDataset
from .production_bn import (
    CrossFittedEvidenceResult,
    fit_bn_from_cross_fitted_evidence,
    generate_cross_fitted_signal_evidence,
    save_cross_fitted_evidence_csv,
)


HAS_XGBOOST = importlib.util.find_spec("xgboost") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


def _make_fts_encoder(mf_type: str) -> FTSEncoder:
    return FTSEncoder(
        n_sets=3,
        mf_type=mf_type,
        norm_mode="row",
        robust=False,
    )


def _split_signal_blocks(features: np.ndarray) -> list[np.ndarray]:
    if features.shape[1] != 36:
        raise ValueError("Expected 36 FTS features")
    return [features[:, start:start + 9] for start in range(0, 36, 9)]


def _fit_signal_models(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    mf_type: str,
    n_rules: int,
    seed: int,
) -> tuple[FTSEncoder, list[ANFIS_TSK]]:
    encoder = _make_fts_encoder(mf_type)
    train_features = encoder.fit_transform(X_train)
    signal_blocks = _split_signal_blocks(train_features)
    models: list[ANFIS_TSK] = []
    for signal_index, block in enumerate(signal_blocks):
        model = ANFIS_TSK(
            n_rules=n_rules,
            sigma_scale=1.5,
            ridge_lambda=0.1,
            n_epochs=15,
            lr=0.01,
            random_state=seed + signal_index,
            mf_type="gaussian",
        )
        model.fit(block, y_train)
        models.append(model)
    return encoder, models


def _predict_signal_evidence(
    *,
    encoder: FTSEncoder,
    signal_models: list[ANFIS_TSK],
    X_data: np.ndarray,
) -> np.ndarray:
    features = encoder.transform(X_data)
    signal_blocks = _split_signal_blocks(features)
    evidence = np.zeros((len(X_data), 4), dtype=float)
    for signal_index, (model, block) in enumerate(zip(signal_models, signal_blocks)):
        evidence[:, signal_index] = model.predict_proba(block)
    return evidence


@dataclass(slots=True)
class ProposedFTSANFISBNAdapter:
    membership_types: tuple[str, ...] = ("triangular", "gaussian", "trapezoid", "gbell")
    rule_counts: tuple[int, ...] = (3, 6, 8, 12)
    crossfit_folds: int = 5
    output_directory: str | None = None
    method_name: str = "proposed_fts_anfis_bn"
    _current_outer_fold_index: int | None = None
    _current_inner_fold_index: int | None = None

    def candidate_specs(self) -> list[CandidateSpec]:
        specs: list[CandidateSpec] = []
        for mf_type in self.membership_types:
            for n_rules in self.rule_counts:
                specs.append(
                    CandidateSpec(
                        candidate_id=f"proposed_{mf_type}_{n_rules}",
                        hyperparameters={"mf_type": mf_type, "n_rules": int(n_rules)},
                        complexity=float(n_rules),
                    )
                )
        return specs

    def set_outer_fold_context(self, outer_fold_manifest: Any) -> None:
        self._current_outer_fold_index = int(outer_fold_manifest.outer_fold_index)

    def set_inner_fold_context(self, outer_fold_index: int, inner_fold_index: int | None) -> None:
        self._current_outer_fold_index = int(outer_fold_index)
        self._current_inner_fold_index = None if inner_fold_index is None else int(inner_fold_index)

    def fit_predict_inner_fold(
        self,
        train_data: NestedDataset,
        valid_data: NestedDataset,
        candidate: CandidateSpec,
        *,
        seed: int,
    ) -> np.ndarray:
        return self._fit_predict(train_data=train_data, eval_data=valid_data, candidate=candidate, seed=seed)

    def fit_predict_outer_fold(
        self,
        train_data: NestedDataset,
        test_data: NestedDataset,
        candidate: CandidateSpec,
        *,
        seed: int,
    ) -> np.ndarray:
        return self._fit_predict(train_data=train_data, eval_data=test_data, candidate=candidate, seed=seed)

    def _fit_predict(
        self,
        *,
        train_data: NestedDataset,
        eval_data: NestedDataset,
        candidate: CandidateSpec,
        seed: int,
    ) -> np.ndarray:
        mf_type = str(candidate.hyperparameters["mf_type"])
        n_rules = int(candidate.hyperparameters["n_rules"])

        crossfit = generate_cross_fitted_signal_evidence(
            dataset=train_data,
            n_splits=min(self.crossfit_folds, len(np.unique(train_data.subject_ids))),
            seed=seed,
            predictor=lambda cf_train, cf_valid, cf_seed: self._predict_partition_evidence(
                train_data=cf_train,
                eval_data=cf_valid,
                mf_type=mf_type,
                n_rules=n_rules,
                seed=cf_seed,
            ),
            metadata={
                "outer_fold_index": self._current_outer_fold_index,
                "inner_fold_index": self._current_inner_fold_index,
                "membership_type": mf_type,
                "rule_count": n_rules,
                "seed": seed,
            },
        )
        self._persist_crossfit_evidence(candidate, crossfit)
        bn = fit_bn_from_cross_fitted_evidence(
            evidence_result=crossfit,
            y_true=train_data.y,
            n_bins=5,
            alpha=0.5,
            soft_evidence=True,
        )
        encoder, signal_models = _fit_signal_models(
            X_train=train_data.X,
            y_train=train_data.y,
            mf_type=mf_type,
            n_rules=n_rules,
            seed=seed,
        )
        eval_evidence = _predict_signal_evidence(
            encoder=encoder,
            signal_models=signal_models,
            X_data=eval_data.X,
        )
        return bn.predict_proba(eval_evidence)

    def _predict_partition_evidence(
        self,
        *,
        train_data: NestedDataset,
        eval_data: NestedDataset,
        mf_type: str,
        n_rules: int,
        seed: int,
    ) -> np.ndarray:
        encoder, signal_models = _fit_signal_models(
            X_train=train_data.X,
            y_train=train_data.y,
            mf_type=mf_type,
            n_rules=n_rules,
            seed=seed,
        )
        return _predict_signal_evidence(
            encoder=encoder,
            signal_models=signal_models,
            X_data=eval_data.X,
        )

    def _persist_crossfit_evidence(
        self,
        candidate: CandidateSpec,
        evidence_result: CrossFittedEvidenceResult,
    ) -> None:
        if not self.output_directory:
            return
        output_dir = Path(self.output_directory) / self.method_name / "cross_fitted_evidence"
        output_name = (
            f"outer_{self._current_outer_fold_index}_"
            f"inner_{self._current_inner_fold_index}_"
            f"{candidate.candidate_id}.csv"
        )
        save_cross_fitted_evidence_csv(evidence_result, output_dir / output_name)


@dataclass(slots=True)
class FTSLogisticRegressionAdapter:
    membership_types: tuple[str, ...] = ("triangular", "gaussian", "trapezoid", "gbell")
    c_values: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    method_name: str = "fts_logistic_regression"

    def candidate_specs(self) -> list[CandidateSpec]:
        specs: list[CandidateSpec] = []
        for mf_type in self.membership_types:
            for c_value in self.c_values:
                specs.append(
                    CandidateSpec(
                        candidate_id=f"fts_lr_{mf_type}_{c_value}",
                        hyperparameters={"mf_type": mf_type, "C": float(c_value)},
                        complexity=float(1.0 / c_value),
                    )
                )
        return specs

    def fit_predict_inner_fold(self, train_data: NestedDataset, valid_data: NestedDataset, candidate: CandidateSpec, *, seed: int) -> np.ndarray:
        return self._fit_predict(train_data=train_data, eval_data=valid_data, candidate=candidate, seed=seed)

    def fit_predict_outer_fold(self, train_data: NestedDataset, test_data: NestedDataset, candidate: CandidateSpec, *, seed: int) -> np.ndarray:
        return self._fit_predict(train_data=train_data, eval_data=test_data, candidate=candidate, seed=seed)

    def _fit_predict(self, *, train_data: NestedDataset, eval_data: NestedDataset, candidate: CandidateSpec, seed: int) -> np.ndarray:
        encoder = _make_fts_encoder(str(candidate.hyperparameters["mf_type"]))
        train_features = encoder.fit_transform(train_data.X)
        eval_features = encoder.transform(eval_data.X)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features)
        eval_scaled = scaler.transform(eval_features)
        model = LogisticRegression(
            C=float(candidate.hyperparameters["C"]),
            solver="liblinear",
            random_state=seed,
            max_iter=1000,
        )
        model.fit(train_scaled, train_data.y)
        return model.predict_proba(eval_scaled)[:, 1]


@dataclass(slots=True)
class FTSRBFSVMAdapter:
    membership_types: tuple[str, ...] = ("triangular", "gaussian", "trapezoid", "gbell")
    c_values: tuple[float, ...] = (0.1, 1.0, 10.0)
    gamma_values: tuple[object, ...] = ("scale", 0.01, 0.1)
    method_name: str = "fts_rbf_svm"

    def candidate_specs(self) -> list[CandidateSpec]:
        specs: list[CandidateSpec] = []
        for mf_type in self.membership_types:
            for c_value in self.c_values:
                for gamma in self.gamma_values:
                    specs.append(
                        CandidateSpec(
                            candidate_id=f"fts_svm_{mf_type}_{c_value}_{gamma}",
                            hyperparameters={"mf_type": mf_type, "C": float(c_value), "gamma": gamma},
                            complexity=float(c_value if gamma == "scale" else c_value * float(gamma)),
                        )
                    )
        return specs

    def fit_predict_inner_fold(self, train_data: NestedDataset, valid_data: NestedDataset, candidate: CandidateSpec, *, seed: int) -> np.ndarray:
        return self._fit_predict(train_data=train_data, eval_data=valid_data, candidate=candidate)

    def fit_predict_outer_fold(self, train_data: NestedDataset, test_data: NestedDataset, candidate: CandidateSpec, *, seed: int) -> np.ndarray:
        return self._fit_predict(train_data=train_data, eval_data=test_data, candidate=candidate)

    def _fit_predict(self, *, train_data: NestedDataset, eval_data: NestedDataset, candidate: CandidateSpec) -> np.ndarray:
        encoder = _make_fts_encoder(str(candidate.hyperparameters["mf_type"]))
        train_features = encoder.fit_transform(train_data.X)
        eval_features = encoder.transform(eval_data.X)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features)
        eval_scaled = scaler.transform(eval_features)
        model = SVC(
            C=float(candidate.hyperparameters["C"]),
            gamma=candidate.hyperparameters["gamma"],
            kernel="rbf",
            probability=False,
        )
        model.fit(train_scaled, train_data.y)
        decision_scores = model.decision_function(eval_scaled)
        return 1.0 / (1.0 + np.exp(-np.clip(decision_scores, -500, 500)))


def _get_xgb_classifier():
    if not HAS_XGBOOST:
        raise RuntimeError("xgboost is not available in the current interpreter")
    from xgboost import XGBClassifier

    return XGBClassifier


@dataclass(slots=True)
class RawCNNPreparedArrays:
    train_x: np.ndarray
    train_y: np.ndarray
    valid_x: np.ndarray
    valid_y: np.ndarray
    eval_x: np.ndarray
    train_subjects: np.ndarray
    valid_subjects: np.ndarray
    channel_mean: np.ndarray
    channel_std: np.ndarray


def _prepare_raw_cnn_arrays(
    *,
    train_data: NestedDataset,
    eval_data: NestedDataset,
    seed: int,
) -> RawCNNPreparedArrays:
    if train_data.X.ndim != 3 or train_data.X.shape[1:] != (4, 30):
        raise ValueError(f"Raw CNN expects train input shape (N, 4, 30), got {train_data.X.shape}")
    if eval_data.X.ndim != 3 or eval_data.X.shape[1:] != (4, 30):
        raise ValueError(f"Raw CNN expects eval input shape (N, 4, 30), got {eval_data.X.shape}")

    unique_subjects = np.unique(train_data.subject_ids.astype(str))
    if len(unique_subjects) < 2:
        raise ValueError("Raw CNN requires at least two training subjects for grouped early stopping")

    splits = list(
        GroupKFold(n_splits=2).split(
            train_data.X,
            train_data.y,
            groups=train_data.subject_ids.astype(str),
        )
    )
    split_index = seed % len(splits)
    train_indices, valid_indices = splits[split_index]
    fit_x = train_data.X[train_indices]
    channel_mean = fit_x.mean(axis=(0, 2))
    channel_std = fit_x.std(axis=(0, 2))
    channel_std = np.where(channel_std == 0.0, 1.0, channel_std)

    def normalize(values: np.ndarray) -> np.ndarray:
        return ((values - channel_mean[None, :, None]) / channel_std[None, :, None]).astype(np.float32)

    return RawCNNPreparedArrays(
        train_x=normalize(train_data.X[train_indices]),
        train_y=train_data.y[train_indices].astype(np.float32),
        valid_x=normalize(train_data.X[valid_indices]),
        valid_y=train_data.y[valid_indices].astype(np.float32),
        eval_x=normalize(eval_data.X),
        train_subjects=train_data.subject_ids[train_indices].astype(str),
        valid_subjects=train_data.subject_ids[valid_indices].astype(str),
        channel_mean=channel_mean.astype(np.float32),
        channel_std=channel_std.astype(np.float32),
    )


def _fit_raw_cnn_model(
    *,
    prepared: RawCNNPreparedArrays,
    learning_rate: float,
    dropout: float,
    epochs: int,
    patience: int,
    seed: int,
) -> np.ndarray:
    if not HAS_TORCH:
        raise RuntimeError("torch is not available in the current interpreter")
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(1)

    class TinyRawCNN(nn.Module):
        def __init__(self, dropout_rate: float) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(4, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(8, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.dropout = nn.Dropout(dropout_rate)
            self.classifier = nn.Linear(8, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            features = self.features(x).squeeze(-1)
            return self.classifier(self.dropout(features)).squeeze(-1)

    model = TinyRawCNN(dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    train_x = torch.from_numpy(prepared.train_x)
    train_y = torch.from_numpy(prepared.train_y)
    valid_x = torch.from_numpy(prepared.valid_x)
    valid_y = torch.from_numpy(prepared.valid_y)

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0

    for _ in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad()
        train_loss = loss_fn(model(train_x), train_y)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            valid_loss = float(loss_fn(model(valid_x), valid_y).item())
        if valid_loss < best_loss - 1e-6:
            best_loss = valid_loss
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= max(1, patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(prepared.eval_x))
        return torch.sigmoid(logits).cpu().numpy().astype(float)


@dataclass(slots=True)
class FTSXGBoostAdapter:
    membership_types: tuple[str, ...] = ("triangular", "gaussian", "trapezoid", "gbell")
    n_estimators_values: tuple[int, ...] = (100, 300)
    max_depth_values: tuple[int, ...] = (3, 5)
    learning_rate_values: tuple[float, ...] = (0.03, 0.1)
    method_name: str = "fts_xgboost"

    def candidate_specs(self) -> list[CandidateSpec]:
        specs: list[CandidateSpec] = []
        for mf_type in self.membership_types:
            for n_estimators in self.n_estimators_values:
                for max_depth in self.max_depth_values:
                    for learning_rate in self.learning_rate_values:
                        specs.append(
                            CandidateSpec(
                                candidate_id=f"fts_xgb_{mf_type}_{n_estimators}_{max_depth}_{learning_rate}",
                                hyperparameters={
                                    "mf_type": mf_type,
                                    "n_estimators": int(n_estimators),
                                    "max_depth": int(max_depth),
                                    "learning_rate": float(learning_rate),
                                },
                                complexity=float(n_estimators * max_depth),
                            )
                        )
        return specs

    def fit_predict_inner_fold(self, train_data: NestedDataset, valid_data: NestedDataset, candidate: CandidateSpec, *, seed: int) -> np.ndarray:
        return self._fit_predict(train_data=train_data, eval_data=valid_data, candidate=candidate, seed=seed)

    def fit_predict_outer_fold(self, train_data: NestedDataset, test_data: NestedDataset, candidate: CandidateSpec, *, seed: int) -> np.ndarray:
        return self._fit_predict(train_data=train_data, eval_data=test_data, candidate=candidate, seed=seed)

    def _fit_predict(self, *, train_data: NestedDataset, eval_data: NestedDataset, candidate: CandidateSpec, seed: int) -> np.ndarray:
        encoder = _make_fts_encoder(str(candidate.hyperparameters["mf_type"]))
        train_features = encoder.fit_transform(train_data.X)
        eval_features = encoder.transform(eval_data.X)
        model = _get_xgb_classifier()(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            seed=seed,
            n_estimators=int(candidate.hyperparameters["n_estimators"]),
            max_depth=int(candidate.hyperparameters["max_depth"]),
            learning_rate=float(candidate.hyperparameters["learning_rate"]),
            subsample=1.0,
            colsample_bytree=1.0,
            n_jobs=1,
            verbosity=0,
        )
        model.fit(train_features, train_data.y)
        return model.predict_proba(eval_features)[:, 1]


@dataclass(slots=True)
class Raw1DCNNAdapter:
    learning_rate_values: tuple[float, ...] = (3e-4, 1e-3)
    dropout_values: tuple[float, ...] = (0.0, 0.2)
    epochs: int = 12
    patience: int = 3
    method_name: str = "raw_1d_cnn"

    def candidate_specs(self) -> list[CandidateSpec]:
        specs: list[CandidateSpec] = []
        for learning_rate in self.learning_rate_values:
            for dropout in self.dropout_values:
                specs.append(
                    CandidateSpec(
                        candidate_id=f"raw_cnn_lr_{learning_rate}_dropout_{dropout}",
                        hyperparameters={
                            "learning_rate": float(learning_rate),
                            "dropout": float(dropout),
                        },
                        complexity=float(1000.0 * dropout + learning_rate),
                    )
                )
        return specs

    def fit_predict_inner_fold(self, train_data: NestedDataset, valid_data: NestedDataset, candidate: CandidateSpec, *, seed: int) -> np.ndarray:
        return self._fit_predict(train_data=train_data, eval_data=valid_data, candidate=candidate, seed=seed)

    def fit_predict_outer_fold(self, train_data: NestedDataset, test_data: NestedDataset, candidate: CandidateSpec, *, seed: int) -> np.ndarray:
        return self._fit_predict(train_data=train_data, eval_data=test_data, candidate=candidate, seed=seed)

    def _fit_predict(self, *, train_data: NestedDataset, eval_data: NestedDataset, candidate: CandidateSpec, seed: int) -> np.ndarray:
        prepared = _prepare_raw_cnn_arrays(train_data=train_data, eval_data=eval_data, seed=seed)
        return _fit_raw_cnn_model(
            prepared=prepared,
            learning_rate=float(candidate.hyperparameters["learning_rate"]),
            dropout=float(candidate.hyperparameters["dropout"]),
            epochs=self.epochs,
            patience=self.patience,
            seed=seed,
        )
