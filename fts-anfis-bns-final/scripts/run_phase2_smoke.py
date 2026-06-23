from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbn_pipeline.production_adapters import (
    HAS_XGBOOST,
    HAS_TORCH,
    FTSLogisticRegressionAdapter,
    FTSRBFSVMAdapter,
    ProposedFTSANFISBNAdapter,
    Raw1DCNNAdapter,
)
from fbn_pipeline.production_data import load_full_subject_dataset
from fbn_pipeline.production_runner import ensure_shared_manifest, run_and_persist_method
from fbn_pipeline.structured_config import EvaluationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reduced Phase 2 nested smoke evaluations.")
    parser.add_argument("--method", choices=("proposed", "lr", "svm", "xgboost", "cnn"), required=True)
    parser.add_argument("--subject-limit", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=2)
    parser.add_argument("--output-dir", default="artifacts/phase2_smoke")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--allow-partial-outer-folds", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.method == "proposed" and args.subject_limit < 5:
        raise SystemExit("The proposed smoke run requires at least 5 subjects for inner BN cross-fitting.")
    dataset = load_full_subject_dataset(ROOT / "last_30s_segments")
    dataset = subset_subjects(dataset, args.subject_limit)
    evaluation = EvaluationConfig(
        outer_protocol="loso",
        inner_folds=args.inner_folds,
        selection_metric="balanced_accuracy",
        threshold_metric="balanced_accuracy",
        threshold_grid=(0.3, 0.4, 0.5, 0.6, 0.7),
        seed=42,
        deterministic_threads=1,
        output_directory=args.output_dir,
    )
    output_dir = ROOT / args.output_dir
    manifest = ensure_shared_manifest(dataset=dataset, evaluation=evaluation, output_directory=output_dir)
    adapter = make_adapter(args.method, output_dir, epochs=args.epochs)
    result = run_and_persist_method(
        dataset=dataset,
        adapter=adapter,
        evaluation=evaluation,
        output_directory=output_dir,
        manifest=manifest,
        allow_partial_outer_folds=args.allow_partial_outer_folds,
    )
    print(f"method={result.method_name}")
    print(f"subjects={len(np.unique(dataset.subject_ids))}")
    print(f"overall_metrics={result.overall_metrics}")
    return 0


def subset_subjects(dataset, subject_limit: int):
    unique_subjects = np.unique(dataset.subject_ids.astype(str))
    selected = set(unique_subjects[:subject_limit].tolist())
    indices = np.array(
        [idx for idx, subject_id in enumerate(dataset.subject_ids.astype(str).tolist()) if subject_id in selected],
        dtype=int,
    )
    return dataset.subset(indices)


def make_adapter(method: str, output_dir: Path, *, epochs: int):
    if method == "proposed":
        return ProposedFTSANFISBNAdapter(
            membership_types=("triangular", "gaussian"),
            rule_counts=(3,),
            crossfit_folds=2,
            output_directory=str(output_dir),
        )
    if method == "lr":
        return FTSLogisticRegressionAdapter(
            membership_types=("triangular", "gaussian"),
            c_values=(0.1, 1.0),
        )
    if method == "svm":
        return FTSRBFSVMAdapter(
            membership_types=("triangular", "gaussian"),
            c_values=(0.1,),
            gamma_values=("scale",),
        )
    if method == "xgboost":
        if not HAS_XGBOOST:
            raise SystemExit("xgboost is not installed in this environment")
        from fbn_pipeline.production_adapters import FTSXGBoostAdapter

        return FTSXGBoostAdapter(
            membership_types=("triangular", "gaussian"),
            n_estimators_values=(100,),
            max_depth_values=(3,),
            learning_rate_values=(0.1,),
        )
    if method == "cnn":
        if not HAS_TORCH:
            raise SystemExit("torch is not installed in this environment")
        return Raw1DCNNAdapter(
            learning_rate_values=(3e-4, 1e-3),
            dropout_values=(0.0, 0.2),
            epochs=max(1, epochs),
            patience=2,
        )
    raise ValueError(f"Unsupported method: {method}")


if __name__ == "__main__":
    raise SystemExit(main())
