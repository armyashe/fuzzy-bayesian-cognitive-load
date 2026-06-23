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

from fbn_pipeline.models.bn_fusion import BNFusion


def make_probs_and_labels(n_samples: int = 32) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    probs = rng.uniform(0.05, 0.95, size=(n_samples, 4))
    y = (probs.mean(axis=1) > 0.5).astype(int)
    return probs, y


def test_bn_priors_cpts_and_bin_edges_are_valid() -> None:
    probs, y = make_probs_and_labels()
    bn = BNFusion(n_bins=3, alpha=0.5)
    bn.fit(probs, y)

    np.testing.assert_allclose(bn.prior_.sum(), 1.0)
    for edges in bn.bin_edges_:
        assert np.all(np.diff(edges) >= 0.0)
    for cpt in bn.cpts_:
        np.testing.assert_allclose(cpt.sum(axis=1), np.ones(cpt.shape[0]))


def test_bn_posterior_probabilities_are_finite_and_normalized() -> None:
    probs, y = make_probs_and_labels()
    bn = BNFusion(n_bins=4, alpha=1.0, soft=True)
    bn.fit(probs, y)
    post = bn.predict_proba(probs)

    assert post.shape == (probs.shape[0],)
    assert np.isfinite(post).all()
    for row in probs[:5]:
        trace = bn.explain_sample(row)
        np.testing.assert_allclose(
            trace["posterior_task"] + trace["posterior_rest"],
            1.0,
        )
