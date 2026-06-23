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
from fbn_pipeline.publication_runner import build_publication_evaluation, load_subject_dataset, run_publication_methods


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the proposed model plus frozen Phase 3 ablations.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", choices=("publication", "smoke"), default="publication")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--subject-limit", type=int)
    parser.add_argument("--inner-folds", type=int, default=10)
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
    dataset = load_subject_dataset(ROOT / args.data_dir, subject_limit=args.subject_limit)
    evaluation = build_publication_evaluation(
        output_directory=str(layout.run_root),
        inner_folds=args.inner_folds,
        seed=args.seed,
    )
    run_publication_methods(
        dataset=dataset,
        layout=layout,
        evaluation=evaluation,
        adapters=[ProposedFTSANFISBNAdapter(output_directory=str(layout.metadata_dir / "raw_results"))],
        command=" ".join(sys.argv),
        allow_partial_outer_folds=args.mode == "smoke",
        ablation_names=("fts_anfis_mean", "fts_summary_bn"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
