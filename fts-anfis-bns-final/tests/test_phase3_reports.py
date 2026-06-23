from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbn_pipeline.publication_reports import generate_figures_from_run, generate_tables_from_run


def make_run_fixture(run_root: Path) -> None:
    for name in ("metadata", "predictions", "metrics", "statistics", "interpretability", "tables", "figures"):
        (run_root / name).mkdir(parents=True, exist_ok=True)

    (run_root / "metadata" / "runtime_metadata.json").write_text(
        json.dumps(
            {
                "run_name": "phase3_smoke",
                "mode": "smoke",
                "allow_partial_outer_folds": True,
                "seed": 7,
                "dataset_order_hash": "d0",
                "configuration_hash": "c0",
                "python_version": "3.12",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "method": "proposed",
                "outer_fold_index": 0,
                "subject_id": "s0",
                "sample_id": "a",
                "true_label": 0,
                "score_or_probability": 0.2,
                "prediction": 0,
                "threshold": 0.5,
                "seed": 7,
                "configuration_hash": "cfg0",
                "dataset_order_hash": "d0",
            },
            {
                "method": "proposed",
                "outer_fold_index": 1,
                "subject_id": "s1",
                "sample_id": "b",
                "true_label": 1,
                "score_or_probability": 0.8,
                "prediction": 1,
                "threshold": 0.5,
                "seed": 7,
                "configuration_hash": "cfg1",
                "dataset_order_hash": "d0",
            },
        ]
    ).to_csv(run_root / "predictions" / "outer_fold_predictions.csv", index=False)
    pd.DataFrame(
        [{"method": "proposed", "balanced_accuracy": 1.0, "accuracy": 1.0, "macro_f1": 1.0, "roc_auc": 1.0}]
    ).to_csv(run_root / "metrics" / "overall_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"method": "proposed", "subject_id": "s0", "balanced_accuracy": 1.0, "accuracy": 1.0},
            {"method": "proposed", "subject_id": "s1", "balanced_accuracy": 1.0, "accuracy": 1.0},
        ]
    ).to_csv(run_root / "metrics" / "per_subject_metrics.csv", index=False)
    pd.DataFrame(
        [{"method": "proposed", "metric": "balanced_accuracy", "lower": 1.0, "upper": 1.0}]
    ).to_csv(run_root / "statistics" / "confidence_intervals.csv", index=False)
    pd.DataFrame(
        [{"method": "proposed", "mf_type": "gaussian", "selection_count": 2, "selection_rate": 1.0}]
    ).to_csv(run_root / "statistics" / "selection_frequencies.csv", index=False)


def test_table_and_figure_generation_use_saved_artifacts_only(tmp_path: Path) -> None:
    run_root = tmp_path / "artifacts" / "phase3_publication" / "phase3_smoke"
    make_run_fixture(run_root)

    table_paths = generate_tables_from_run(run_root)
    figure_paths = generate_figures_from_run(run_root)

    assert any(path.name.startswith("table_c_") for path in table_paths)
    assert any(path.suffix == ".png" for path in figure_paths)
