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

from fbn_pipeline.publication_artifacts import load_run_layout
from fbn_pipeline.publication_reports import generate_figures_from_run, generate_tables_from_run
from fbn_pipeline.publication_runner import derive_publication_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate Phase 3 statistics, tables, and figures from saved artifacts.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", default="artifacts/phase3_publication")
    parser.add_argument("--bootstrap-repetitions", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_root = ROOT / args.output_root / args.run_name
    layout = load_run_layout(run_root)
    bootstrap_repetitions = args.bootstrap_repetitions or (200 if layout.mode == "smoke" else 2000)
    derive_publication_outputs(
        layout=layout,
        bootstrap_repetitions=bootstrap_repetitions,
    )
    generate_tables_from_run(run_root)
    generate_figures_from_run(run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
