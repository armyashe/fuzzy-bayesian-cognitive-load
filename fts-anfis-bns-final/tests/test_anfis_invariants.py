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

from fbn_pipeline.models.anfis_tsk import ANFIS_TSK


def make_signal_features(n_samples: int = 24, n_features: int = 9) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, n_features))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0.0).astype(int)
    return X, y


def test_anfis_same_seed_is_deterministic() -> None:
    X, y = make_signal_features()
    model_a = ANFIS_TSK(n_rules=4, n_epochs=4, lr=0.01, random_state=7)
    model_b = ANFIS_TSK(n_rules=4, n_epochs=4, lr=0.01, random_state=7)

    model_a.fit(X, y)
    model_b.fit(X, y)

    np.testing.assert_allclose(model_a.predict_proba(X), model_b.predict_proba(X))


def test_anfis_probability_shape_range_and_finiteness() -> None:
    X, y = make_signal_features()
    model = ANFIS_TSK(n_rules=5, n_epochs=3, lr=0.01, random_state=11)
    model.fit(X, y)
    probs = model.predict_proba(X)

    assert probs.shape == (X.shape[0],)
    assert np.isfinite(probs).all()
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_anfis_premise_and_consequent_dimensions_match_rule_count() -> None:
    X, y = make_signal_features()
    model = ANFIS_TSK(n_rules=6, n_epochs=3, lr=0.01, random_state=13)
    model.fit(X, y)

    assert model.centers_.shape == model.sigmas_.shape
    assert model.centers_.shape[0] == model.n_rules_
    assert model.theta_.shape[0] == model.n_rules_ * (X.shape[1] + 1)
