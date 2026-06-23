from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from .nested_evaluation import CandidateSpec, NestedDataset


@dataclass(slots=True)
class LogisticRegressionAdapter:
    candidate_cs: tuple[float, ...]
    method_name: str = "logistic_regression"

    def candidate_specs(self) -> list[CandidateSpec]:
        specs: list[CandidateSpec] = []
        for c in self.candidate_cs:
            specs.append(
                CandidateSpec(
                    candidate_id=f"lr_c_{c}",
                    hyperparameters={"C": float(c)},
                    complexity=float(c),
                )
            )
        return specs

    def fit_predict_inner_fold(
        self,
        train_data: NestedDataset,
        valid_data: NestedDataset,
        candidate: CandidateSpec,
        *,
        seed: int,
    ) -> np.ndarray:
        model = self._make_model(candidate, seed)
        model.fit(_flatten(train_data.X), train_data.y)
        return model.predict_proba(_flatten(valid_data.X))[:, 1]

    def fit_predict_outer_fold(
        self,
        train_data: NestedDataset,
        test_data: NestedDataset,
        candidate: CandidateSpec,
        *,
        seed: int,
    ) -> np.ndarray:
        model = self._make_model(candidate, seed)
        model.fit(_flatten(train_data.X), train_data.y)
        return model.predict_proba(_flatten(test_data.X))[:, 1]

    @staticmethod
    def _make_model(candidate: CandidateSpec, seed: int) -> LogisticRegression:
        return LogisticRegression(
            C=float(candidate.hyperparameters["C"]),
            solver="liblinear",
            random_state=seed,
            max_iter=1000,
        )


def _flatten(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)
