from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbn_pipeline.production_adapters import ProposedFTSANFISBNAdapter
from fbn_pipeline.publication_artifacts import create_run_layout
from fbn_pipeline.publication_runner import build_publication_evaluation, run_publication_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the proposed publication pipeline.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", choices=("publication", "smoke"), default="publication")
    parser.add_argument("--protocol", choices=("official_holdout", "nested_loso_secondary"), default="official_holdout")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--train-subject-limit", type=int)
    parser.add_argument("--test-subject-limit", type=int)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="artifacts/phase3_publication")
    parser.add_argument("--data-dir", default="last_30s_segments")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    layout = create_run_layout(
        output_root=ROOT / args.output_root,
        run_name=args.run_name,
        mode=args.mode,
        overwrite=args.overwrite,
    )
    evaluation = build_publication_evaluation(
        output_directory=str(layout.run_root),
        inner_folds=args.inner_folds,
        seed=args.seed,
    )
    run_publication_workflow(
        data_dir=ROOT / args.data_dir,
        layout=layout,
        evaluation=evaluation,
        adapters=[ProposedFTSANFISBNAdapter(output_directory=str(layout.metadata_dir / "raw_results"))],
        command=" ".join(sys.argv),
        protocol=args.protocol,
        allow_partial_outer_folds=args.mode == "smoke" and args.protocol == "nested_loso_secondary",
        train_subject_limit=args.train_subject_limit,
        test_subject_limit=args.test_subject_limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
