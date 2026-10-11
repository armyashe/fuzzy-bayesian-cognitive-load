from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbn_pipeline.final_evaluation import (
    _development_summary_markdown,
    _participant_level_report_markdown,
    _participant_level_summary,
    _participant_level_test_metrics,
    _validate_participant_level_analysis,
    _summarise_trained_models,
)
from fbn_pipeline.results import TrainedModels


class FakeFTS:
    def get_summary(self):
        return [
            {
                "signal": "Temp",
                "membership_type": "gaussian",
                "n_sets": 3,
                "fuzzy_sets": {"Low": {"center": 0.12345, "sigma": 0.22222}},
            }
        ]


class FakeANFIS:
    mf_type = "gaussian"

    def get_rule_params(self):
        return {
            "centers": np.array([[0.12345, 0.67891]]),
            "sigmas": np.array([[0.33333, 0.44444]]),
            "theta": np.array([[1.23456, 2.34567, 3.45678]]),
            "n_rules": 1,
            "d": 2,
        }


class FakeBN:
    soft = True
    n_bins = 5
    alpha = 0.5
    prior_ = np.array([0.4, 0.6])
    bin_edges_ = [np.array([0.2, 0.4, 0.6, 0.8])]
    bin_mids_ = [np.array([0.1, 0.3, 0.5, 0.7, 0.9])]
    bin_sigmas_ = [np.array([0.1, 0.1, 0.1, 0.1, 0.1])]

    def get_signal_weights(self):
        return {"Temp": 1.0}

    def get_cpt_table(self):
        return {
            "Temp": {
                "rest": {"Low": 0.5},
                "task": {"Low": 0.5},
            }
        }


class TestFinalEvaluationSummary(unittest.TestCase):
    def test_summarise_trained_models_collects_all_sections(self) -> None:
        models = TrainedModels(
            fts=FakeFTS(),
            bn=FakeBN(),
            anfis_list=[FakeANFIS()],
            test_evidence=np.zeros((2, 4)),
            train_evidence=np.zeros((3, 4)),
        )

        summary = _summarise_trained_models(models)

        self.assertIn("fts_encoder", summary)
        self.assertIn("anfis_models", summary)
        self.assertIn("bayesian_network", summary)
        self.assertEqual(summary["anfis_models"][0]["signal"], "Temp")
        self.assertEqual(summary["bayesian_network"]["n_bins"], 5)
        self.assertEqual(summary["train_evidence_shape"], [3, 4])
        self.assertEqual(summary["test_evidence_shape"], [2, 4])

    def test_development_summary_markdown_mentions_selected_oof_choices(self) -> None:
        development = {
            "summary": {
                "selected_configuration_id": "PUB_JOINT__MF_GAUSSIAN__R6",
                "selected_mf_type": "gaussian",
                "selected_n_rules": 6,
                "selected_threshold": 0.55,
                "selected_oof_balanced_accuracy": 0.6236,
                "selected_oof_macro_f1": 0.6222,
                "selected_oof_roc_auc": 0.6201,
                "fold_assignment_hash": "fold-hash-123",
            },
            "long_form_table": pd.DataFrame(
                [
                    {
                        "configuration_id": "PUB_JOINT__MF_GAUSSIAN__R6",
                        "mf_type": "gaussian",
                        "n_rules": 6,
                        "oof_balanced_accuracy": 0.6236,
                        "oof_macro_f1": 0.6222,
                        "oof_roc_auc": 0.6201,
                        "oof_accuracy": 0.6234,
                        "selected_threshold": 0.55,
                        "selected": True,
                    }
                ]
            ),
            "matrix_table": pd.DataFrame(
                [
                    {
                        "Membership function": "Gaussian",
                        "3 rules": 0.6000,
                        "6 rules": 0.6236,
                        "8 rules": 0.5795,
                        "12 rules": 0.5557,
                    }
                ]
            ),
        }
        proposed_lock = {"config": {"mf_type": "gaussian", "n_rules": 6, "bn_soft": True}}
        model_summary = {"bayesian_network": {"soft_evidence": True}}
        path_map = {
            "development_summary_json": Path("dev_summary.json"),
            "development_long_csv": Path("joint.csv"),
            "development_matrix_csv": Path("matrix.csv"),
            "development_report_md": Path("report.md"),
            "development_legacy_dir": Path("legacy"),
        }

        markdown = _development_summary_markdown(
            development=development,
            proposed_lock=proposed_lock,
            model_summary=model_summary,
            path_map=path_map,
        )

        self.assertIn("Joint OOF Balanced Accuracy Matrix", markdown)
        self.assertIn("Selected membership function: `gaussian`", markdown)
        self.assertIn("Selected number of rules: `6`", markdown)
        self.assertIn("Final OOF threshold: `0.5500`", markdown)
        self.assertIn("was not tuned on the hold-out test set", markdown)
        self.assertIn("Learned Parameters on Development Training Set", markdown)

    def test_participant_level_metrics_preserve_order_and_allow_undefined_auc(self) -> None:
        entry = {
            "display_name": "Proposed FTS-ANFIS-BN",
            "model_id": "PUB_JOINT__MF_GAUSSIAN__R6",
            "lock_hash": "lock-hash",
            "threshold": 0.55,
            "y_true": np.array([0, 0, 0, 0, 1, 1], dtype=int),
            "scores": np.array([0.10, 0.70, 0.20, 0.20, 0.80, 0.30], dtype=float),
            "preds": np.array([0, 1, 0, 0, 1, 0], dtype=int),
            "subjects": np.array(["b", "b", "b", "a", "a", "a"]),
            "metrics": {"prediction_hash": "pred-hash"},
        }

        df = _participant_level_test_metrics(
            entry,
            participant_order=["a", "b"],
        )

        self.assertEqual(
            list(df.columns),
            ["participant", "n_samples", "accuracy", "balanced_accuracy", "task_f1", "roc_auc"],
        )
        self.assertEqual(df["participant"].tolist(), ["a", "b"])
        self.assertEqual(df["n_samples"].tolist(), [3, 3])
        self.assertAlmostEqual(float(df.loc[0, "accuracy"]), 0.6667, places=4)
        self.assertAlmostEqual(float(df.loc[0, "roc_auc"]), 1.0, places=4)
        self.assertTrue(pd.isna(df.loc[1, "roc_auc"]))

    def test_participant_level_report_and_validation_include_publication_requirements(self) -> None:
        df = pd.DataFrame(
            [
                {"participant": "3caqi", "n_samples": 2, "accuracy": 1.0, "balanced_accuracy": 1.0, "task_f1": 1.0, "roc_auc": 1.0},
                {"participant": "6frz4", "n_samples": 2, "accuracy": 0.5, "balanced_accuracy": 0.5, "task_f1": 0.5, "roc_auc": np.nan},
            ]
        )
        summary = _participant_level_summary(df)
        markdown = _participant_level_report_markdown(df, summary)

        self.assertIn("Table 8", markdown)
        self.assertIn("Participant-Level Analysis", markdown)
        self.assertIn("unweighted statistics across the five unseen participants", markdown)
        self.assertIn("N/A", markdown)
        self.assertIn("must not be used for model selection", markdown)

        entry = {
            "threshold": 0.55,
            "y_true": np.array([0, 1, 0, 1], dtype=int),
            "scores": np.array([0.10, 0.90, 0.30, 0.80], dtype=float),
            "preds": np.array([0, 1, 0, 1], dtype=int),
            "subjects": np.array(["3caqi", "3caqi", "6frz4", "6frz4"]),
        }
        validation = _validate_participant_level_analysis(
            entry,
            df,
            expected_participants=["3caqi", "6frz4"],
            expected_subjects=np.array(["3caqi", "3caqi", "6frz4", "6frz4"]),
            expected_threshold=0.55,
            expected_accuracy=1.0,
            expected_seed=42,
        )

        self.assertTrue(validation["sum_n_samples_matches_expected"])
        self.assertTrue(validation["all_samples_map_to_exactly_one_participant"])
        self.assertTrue(validation["prediction_order_matches_user_test"])
        self.assertTrue(validation["locked_threshold_matches_expected"])
        self.assertTrue(validation["no_participant_tuning_or_retraining"])
        self.assertTrue(validation["aggregate_accuracy_matches_expected"])
        self.assertTrue(validation["seed_is_42"])


if __name__ == "__main__":
    unittest.main()
