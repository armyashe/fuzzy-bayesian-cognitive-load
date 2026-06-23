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

from fbn_pipeline.models.fts_encoder import FTSEncoder


def make_windows(n_samples: int = 6, n_steps: int = 10) -> np.ndarray:
    base = np.linspace(0.0, 1.0, n_steps)
    windows = np.zeros((n_samples, 4, n_steps), dtype=float)
    for i in range(n_samples):
        for s in range(4):
            windows[i, s, :] = (s + 1) * 5.0 + i * 0.1 + base * (s + 2)
    return windows


def test_three_fuzzy_sets_create_nine_transition_features_per_signal() -> None:
    X = make_windows()
    encoder = FTSEncoder(n_sets=3, mf_type="gaussian")
    features = encoder.fit_transform(X)

    assert features.shape == (X.shape[0], 36)
    assert features.shape[1] // 4 == 9


def test_row_normalized_transition_rows_sum_to_one_when_nonzero() -> None:
    X = make_windows()
    encoder = FTSEncoder(n_sets=3, mf_type="gaussian", norm_mode="row")
    features = encoder.fit_transform(X).reshape(X.shape[0], 4, 3, 3)

    row_sums = features.sum(axis=3)
    mask = row_sums > 1e-10
    np.testing.assert_allclose(row_sums[mask], np.ones_like(row_sums[mask]))


def test_transform_output_is_finite_and_fit_transform_are_separated() -> None:
    X_train = make_windows(n_samples=5)
    X_test = make_windows(n_samples=3) + 100.0
    encoder = FTSEncoder(n_sets=3, mf_type="gaussian")
    encoder.fit(X_train)
    transformed = encoder.transform(X_test)

    assert transformed.shape == (3, 36)
    assert np.isfinite(transformed).all()
