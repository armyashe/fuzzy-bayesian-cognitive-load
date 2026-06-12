from __future__ import annotations

from typing import Protocol

import numpy as np


class FusionModel(Protocol):
    """Common API implemented by BN fusion models."""

    n_bins: int
    bin_names_: list[str]
    prior_: np.ndarray

    def fit(
        self,
        probs: np.ndarray,
        y: np.ndarray,
        signal_weights: np.ndarray | None = None,
    ) -> "FusionModel":
        ...

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:
        ...

    def predict(self, probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        ...

    def get_cpt_table(self) -> dict:
        ...

    def get_signal_weights(self) -> dict:
        ...

    def explain_sample(self, probs_single: np.ndarray) -> dict:
        ...
