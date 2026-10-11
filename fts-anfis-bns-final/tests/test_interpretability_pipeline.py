from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbn_pipeline.models.bn_fusion import BNFusion
from fbn_pipeline.interpretability import (
    TRANSITION_FEATURE_NAMES,
    compute_anfis_consequent_importance,
    compute_sample_bn_log_odds_breakdown,
)


class _FakeAnfis:
    def __init__(self, theta: np.ndarray, d: int) -> None:
        self.theta_ = theta
        self.d_ = d
        self.n_rules_ = theta.size // (d + 1)


class TestInterpretabilityPipeline(unittest.TestCase):
    def test_soft_bn_breakdown_matches_predict_proba(self) -> None:
        train_probs = np.array(
            [
                [0.10, 0.15, 0.20, 0.25],
                [0.20, 0.25, 0.30, 0.35],
                [0.70, 0.75, 0.80, 0.85],
                [0.80, 0.85, 0.90, 0.95],
            ],
            dtype=float,
        )
        y = np.array([0, 0, 1, 1], dtype=int)
        bn = BNFusion(n_bins=5, alpha=0.5, soft=True).fit(train_probs, y)

        sample = np.array([0.62, 0.55, 0.73, 0.61], dtype=float)
        breakdown = compute_sample_bn_log_odds_breakdown(bn, sample)

        self.assertAlmostEqual(
            breakdown["posterior_task"],
            float(bn.predict_proba(sample.reshape(1, -1))[0]),
            places=10,
        )
        self.assertAlmostEqual(
            breakdown["final_log_odds"],
            breakdown["prior_log_odds"] + sum(breakdown["signal_contributions"].values()),
            places=10,
        )
        self.assertEqual(set(breakdown["signal_contributions"]), {"Temp", "HR", "GSR", "RR"})

    def test_consequent_importance_uses_row_major_transition_order_without_bias(self) -> None:
        d = 9
        rule_1 = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, 100.0])
        rule_2 = np.array([-0.5, 0.0, 1.5, 0.0, -2.5, 0.0, 3.5, 0.0, -4.5, 200.0])
        theta = np.concatenate([rule_1, rule_2])
        models = [_FakeAnfis(theta=theta, d=d) for _ in range(4)]

        df = compute_anfis_consequent_importance(models)

        temp = df[df["signal"] == "Temp"].reset_index(drop=True)
        self.assertEqual(temp["transition_feature"].tolist(), TRANSITION_FEATURE_NAMES)
        self.assertEqual(
            temp["coefficient_magnitude_sum"].round(4).tolist(),
            [1.5, 2.0, 4.5, 4.0, 7.5, 6.0, 10.5, 8.0, 13.5],
        )
        self.assertEqual(temp.iloc[0]["rank_within_signal"], 9)
        self.assertEqual(temp.iloc[-1]["rank_within_signal"], 1)


if __name__ == "__main__":
    unittest.main()
