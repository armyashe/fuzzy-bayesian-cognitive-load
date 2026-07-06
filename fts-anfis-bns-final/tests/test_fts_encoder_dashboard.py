from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbn_pipeline.ablation import Ablation
from fbn_pipeline.config import ExperimentConfig
from fbn_pipeline.models.fts_encoder import FTSEncoder
from fbn_pipeline.stages import extract_features_for_ablation
from streamlit_app.tabs import bn_rules_tab


def make_windows(n_samples: int = 8, n_steps: int = 12) -> np.ndarray:
    base = np.linspace(0.0, 1.0, n_steps)
    windows = np.zeros((n_samples, 4, n_steps), dtype=float)
    for i in range(n_samples):
        for s in range(4):
            windows[i, s, :] = (s + 1) * 10.0 + i * 0.25 + base * (s + 1)
    return windows


class TestFTSEncoderDashboard(unittest.TestCase):
    def test_stage1_fits_train_and_transforms_test_without_refit(self) -> None:
        calls = []

        class SpyFTSEncoder(FTSEncoder):
            def fit(self, X):
                calls.append(("fit", X.shape, float(np.nanmax(X))))
                return super().fit(X)

            def transform(self, X):
                calls.append(("transform", X.shape, float(np.nanmax(X))))
                return super().transform(X)

        X_train = make_windows(n_samples=6)
        X_test = make_windows(n_samples=3) + 1000.0
        cfg = ExperimentConfig(ablation=Ablation.FULL.value, n_sets=3, mf_type="gaussian")

        with patch("fbn_pipeline.stages.FTSEncoder", SpyFTSEncoder):
            F_train, F_test, fts = extract_features_for_ablation(X_train, X_test, cfg)

        fit_calls = [call for call in calls if call[0] == "fit"]
        transform_calls = [call for call in calls if call[0] == "transform"]
        self.assertEqual(len(fit_calls), 1)
        self.assertEqual(fit_calls[0][1], X_train.shape)
        self.assertLess(fit_calls[0][2], 1000.0)
        self.assertEqual(transform_calls[0][1], X_train.shape)
        self.assertEqual(transform_calls[1][1], X_test.shape)
        self.assertIsNotNone(fts)
        self.assertEqual(F_train.shape[1], 36)
        self.assertEqual(F_test.shape[1], 36)

    def test_summary_and_trace_are_from_fitted_encoder(self) -> None:
        X_train = make_windows(n_samples=7)
        fts = FTSEncoder(n_sets=3, mf_type="gaussian")
        features = fts.fit_transform(X_train)

        summary = fts.get_summary()
        self.assertEqual([item["signal"] for item in summary], ["Temp", "HR", "GSR", "RR"])
        self.assertEqual(features.shape[1], 4 * 3 * 3)

        for item in summary:
            sets = item["fuzzy_sets"]
            centers = [sets[name]["center"] for name in ["Low", "Medium", "High"]]
            sigmas = [sets[name]["sigma"] for name in ["Low", "Medium", "High"]]
            self.assertLess(centers[0], centers[1])
            self.assertLess(centers[1], centers[2])
            self.assertTrue(all(sigma > 0 for sigma in sigmas))
            self.assertLessEqual(item["universe_lower"], item["training_min"])
            self.assertGreaterEqual(item["universe_upper"], item["training_max"])

        matrices = fts.get_mean_training_transition_matrices()
        self.assertIsNotNone(matrices)
        self.assertEqual(np.asarray(matrices["Temp"]).shape, (3, 3))

        trace = fts.get_encoding_trace(X_train[0], signal_idx=0, n_timesteps=5)
        self.assertEqual(np.asarray(trace["transition_matrix"]).shape, (3, 3))
        self.assertEqual(len(trace["flattened_vector"]), 9)
        self.assertEqual(len(trace["membership_degrees"]), 5)

    def test_non_gaussian_summary_tracks_configured_membership(self) -> None:
        X_train = make_windows(n_samples=5)
        fts = FTSEncoder(n_sets=4, mf_type="trapezoid")
        fts.fit_transform(X_train)
        summary = fts.get_summary()

        self.assertEqual(summary[0]["membership_type"], "trapezoid")
        self.assertEqual(summary[0]["n_sets"], 4)
        self.assertEqual(len(summary[0]["fuzzy_sets"]), 4)
        first_set = next(iter(summary[0]["fuzzy_sets"].values()))
        self.assertTrue({"a", "b", "c", "d", "center"}.issubset(first_set))

    def test_no_fts_ablation_returns_no_encoder(self) -> None:
        X_train = make_windows(n_samples=4)
        X_test = make_windows(n_samples=2)
        cfg = ExperimentConfig(ablation=Ablation.NO_FTS.value)

        F_train, F_test, fts = extract_features_for_ablation(X_train, X_test, cfg)

        self.assertIsNone(fts)
        self.assertEqual(F_train.shape[0], X_train.shape[0])
        self.assertEqual(F_test.shape[0], X_test.shape[0])

    def test_no_fts_dashboard_section_does_not_require_fake_encoder(self) -> None:
        class FakeStreamlit:
            def __getattr__(self, _name):
                def method(*_args, **_kwargs):
                    return None

                return method

        with patch.object(bn_rules_tab, "st", FakeStreamlit()):
            bn_rules_tab.render_fts_encoder_section({}, None)


if __name__ == "__main__":
    unittest.main()
