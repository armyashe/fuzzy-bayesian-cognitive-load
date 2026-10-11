from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbn_pipeline.config import ExperimentConfig
from fbn_pipeline.results import DevelopmentResult


class TestJointFtsAnfisSelection(unittest.TestCase):
    def _full_coverage_fold_fixture(
        self,
        n_samples: int = 632,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
        users = np.array(
            ["u1"] * (n_samples // 2) + ["u2"] * (n_samples - (n_samples // 2))
        )
        fold_indices = [
            (
                np.arange(n_samples // 2, n_samples, dtype=int),
                np.arange(0, n_samples // 2, dtype=int),
            ),
            (
                np.arange(0, n_samples // 2, dtype=int),
                np.arange(n_samples // 2, n_samples, dtype=int),
            ),
        ]
        return fold_indices, users

    def _make_result(
        self,
        *,
        mf_type: str,
        n_rules: int,
        bal_acc: float,
        macro_f1: float,
        auc: float,
        threshold: float = 0.55,
        fold_indices: list[tuple[np.ndarray, np.ndarray]] | None = None,
        n_samples: int = 632,
    ) -> DevelopmentResult:
        cfg = ExperimentConfig(
            seed=42,
            k_folds=5,
            mf_type=mf_type,
            n_rules=n_rules,
            bn_structure="naive",
            bn_soft=True,
            n_bins=5,
            alpha=0.5,
            threshold_metric="balanced_accuracy",
            rank_metric="balanced_accuracy",
        )
        oof_post = np.linspace(0.1, 0.9, n_samples)
        oof_pred = (oof_post >= threshold).astype(int)
        oof_y = np.array([0, 1] * (n_samples // 2), dtype=int)
        return DevelopmentResult(
            config=cfg,
            configuration_id=f"JOINT__{mf_type.upper()}__R{n_rules}",
            fold_indices=fold_indices or [],
            threshold=threshold,
            threshold_metric="balanced_accuracy",
            threshold_score=bal_acc,
            oof_post=oof_post,
            oof_pred=oof_pred,
            oof_y=oof_y,
            oof_metrics={
                "accuracy": bal_acc - 0.01,
                "bal_acc": bal_acc,
                "macro_f1": macro_f1,
                "f1": macro_f1 - 0.01,
                "auc": auc,
                "precision": 0.6,
                "recall": 0.7,
            },
            selection_metric_value=bal_acc,
        )

    def test_joint_selection_requires_all_16_configurations(self) -> None:
        from fbn_pipeline.publication_selection import validate_joint_development_results

        fold_indices = [(np.array([2, 3]), np.array([0, 1]))]
        results = [
            self._make_result(
                mf_type="gaussian",
                n_rules=rules,
                bal_acc=0.6 + rules * 0.001,
                macro_f1=0.59,
                auc=0.58,
                fold_indices=fold_indices,
                n_samples=632,
            )
            for rules in (3, 6, 8)
        ]

        with self.assertRaisesRegex(ValueError, "16 configurations"):
            validate_joint_development_results(
                results=results,
                membership_functions=["triangular", "gaussian", "trapezoid", "gbell"],
                rule_sizes=[3, 6, 8, 12],
                user_train=np.array(["u1", "u1", "u2", "u2"]),
                expected_samples=632,
            )

    def test_joint_selection_rejects_incomplete_oof_or_leakage(self) -> None:
        from fbn_pipeline.publication_selection import validate_joint_development_results

        fold_indices = [
            (np.array([2, 3]), np.array([0, 1])),
            (np.array([0, 1]), np.array([2, 3])),
        ]
        results = []
        for mf_type in ("triangular", "gaussian", "trapezoid", "gbell"):
            for n_rules in (3, 6, 8, 12):
                n_samples = 631 if (mf_type, n_rules) == ("gaussian", 6) else 632
                results.append(
                    self._make_result(
                        mf_type=mf_type,
                        n_rules=n_rules,
                        bal_acc=0.6,
                        macro_f1=0.59,
                        auc=0.58,
                        fold_indices=fold_indices,
                        n_samples=n_samples,
                    )
                )

        with self.assertRaisesRegex(ValueError, "632 OOF predictions"):
            validate_joint_development_results(
                results=results,
                membership_functions=["triangular", "gaussian", "trapezoid", "gbell"],
                rule_sizes=[3, 6, 8, 12],
                user_train=np.array(["u1", "u1", "u2", "u2"]),
                expected_samples=632,
            )

    def test_joint_selection_report_mentions_holdout_exclusion_and_selected_config(self) -> None:
        from fbn_pipeline.publication_selection import (
            build_joint_selection_artifacts,
            validate_joint_development_results,
        )

        fold_indices, user_train = self._full_coverage_fold_fixture()
        results = []
        for mf_type in ("triangular", "gaussian", "trapezoid", "gbell"):
            for n_rules in (3, 6, 8, 12):
                bal_acc = 0.65 if (mf_type, n_rules) == ("gaussian", 6) else 0.60
                macro_f1 = 0.64 if (mf_type, n_rules) == ("gaussian", 6) else 0.59
                auc = 0.63 if (mf_type, n_rules) == ("gaussian", 6) else 0.58
                results.append(
                    self._make_result(
                        mf_type=mf_type,
                        n_rules=n_rules,
                        bal_acc=bal_acc,
                        macro_f1=macro_f1,
                        auc=auc,
                        fold_indices=fold_indices,
                    )
                )

        selection = validate_joint_development_results(
            results=results,
            membership_functions=["triangular", "gaussian", "trapezoid", "gbell"],
            rule_sizes=[3, 6, 8, 12],
            user_train=user_train,
            expected_samples=632,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts = build_joint_selection_artifacts(
                results=results,
                selection=selection,
                artifact_dir=Path(tmpdir),
                fold_assignment_hash="fold-hash-123",
                legacy_note="legacy only",
            )
            report = artifacts["report_path"].read_text(encoding="utf-8")

        self.assertIn("Gaussian", report)
        self.assertIn("6", report)
        self.assertIn("hold-out set was not used", report)
        self.assertIn("balanced accuracy", report)
        self.assertIn("0.55", report)


if __name__ == "__main__":
    unittest.main()
