from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbn_pipeline.metrics import compute_metrics, find_best_threshold


def test_balanced_accuracy_is_calculated_correctly() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.2, 0.7, 0.6, 0.9])

    metrics = compute_metrics(y_true, y_prob, threshold=0.5)

    assert metrics["bal_acc"] == 0.75


def test_threshold_selection_uses_only_the_provided_arrays() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    best_t_1, score_1 = find_best_threshold(y_true, y_prob, n_grid=5)
    best_t_2, score_2 = find_best_threshold(y_true.copy(), y_prob.copy(), n_grid=5)

    assert best_t_1 == best_t_2
    assert score_1 == score_2


def test_threshold_tie_breaking_is_deterministic() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.4, 0.6, 0.4, 0.6])
    best_t, score = find_best_threshold(
        y_true,
        y_prob,
        metric="balanced_accuracy",
        low=0.4,
        high=0.6,
        n_grid=3,
    )

    assert best_t == 0.5
    assert score == 1.0
