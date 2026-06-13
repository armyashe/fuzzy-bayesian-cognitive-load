from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fbn_pipeline import ExperimentConfig, apply_preset  # noqa: E402
from fbn_pipeline.data import load_dataset  # noqa: E402
from fbn_pipeline.runner import run_one_experiment  # noqa: E402


PLATEAU_SCALES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
WIDTH_SCALES = [0.8, 1.0, 1.2, 1.5]
MARGIN_SCALES = [0.00, 0.05, 0.10]

TARGET = {
    "accuracy": 0.6736,
    "precision": 0.6636,
    "recall": 0.7245,
    "f1": 0.6927,
}


def _target_distance(metrics: dict[str, float]) -> float:
    return float(
        sum((metrics[name] - target) ** 2 for name, target in TARGET.items())
    )


def _run_cfg(dataset, plateau: float, width: float, margin: float):
    cfg = ExperimentConfig(data_path=str(PROJECT_ROOT / "last_30s_segments"))
    cfg = apply_preset(cfg, "THESIS_FINAL").merged(
        mf_type="trapezoid",
        trap_plateau_scale=plateau,
        trap_width_scale=width,
        grid_margin_scale=margin,
        bn_structure="naive",
        ablation="exp1",
        anfis_mf_type="gaussian",
        experiment_group="EXP1",
        variant_name="EXP1_MF_TRAP",
    )
    with contextlib.redirect_stdout(io.StringIO()):
        result = run_one_experiment(
            cfg,
            dataset,
            exp_name="EXP1_MF_TRAP",
            preset_name="EXP1_MF_TRAP_DEBUG",
            csv_path=None,
            print_progress=False,
        )
    cm = confusion_matrix(dataset.y_test, result.test_pred, labels=[0, 1])
    return result, cm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grid-search legacy trapezoid FTS parameters for EXP1_MF_TRAP."
    )
    parser.add_argument(
        "--data-folder",
        default=str(PROJECT_ROOT / "last_30s_segments"),
        help="Path to last_30s_segments data folder.",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.data_folder)
    best = None

    print(
        "plateau_scale | width_scale | margin_scale | threshold | "
        "test_acc | precision | recall | f1 | auc | TN | FP | FN | TP"
    )
    print("-" * 111)

    for plateau in PLATEAU_SCALES:
        for width in WIDTH_SCALES:
            for margin in MARGIN_SCALES:
                result, cm = _run_cfg(dataset, plateau, width, margin)
                test = result.test_metrics
                tn, fp, fn, tp = [int(x) for x in cm.ravel()]
                print(
                    f"{plateau:13.2f} | {width:11.2f} | {margin:12.2f} | "
                    f"{result.best_threshold:9.4f} | "
                    f"{test['accuracy']:8.4f} | {test['precision']:9.4f} | "
                    f"{test['recall']:6.4f} | {test['f1']:4.4f} | "
                    f"{test['auc']:4.4f} | {tn:2d} | {fp:2d} | {fn:2d} | {tp:2d}"
                )
                candidate = (_target_distance(test), plateau, width, margin, result, cm)
                if best is None or candidate[0] < best[0]:
                    best = candidate

    assert best is not None
    _, plateau, width, margin, result, cm = best
    test = result.test_metrics
    tn, fp, fn, tp = [int(x) for x in cm.ravel()]
    print("\nBest match by squared distance to target test metrics:")
    print(
        f"plateau_scale={plateau:.2f}, width_scale={width:.2f}, "
        f"margin_scale={margin:.2f}, threshold={result.best_threshold:.4f}"
    )
    print(
        f"test_acc={test['accuracy']:.4f}, precision={test['precision']:.4f}, "
        f"recall={test['recall']:.4f}, f1={test['f1']:.4f}, auc={test['auc']:.4f}"
    )
    print(f"confusion_matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")


if __name__ == "__main__":
    main()
