from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbn_pipeline.legacy_snapshot import run_legacy_snapshot_twice


FIXTURE_PATH = ROOT / "tests" / "fixtures" / "legacy_reference.json"


def main() -> int:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot = run_legacy_snapshot_twice()
    FIXTURE_PATH.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote fixture to {FIXTURE_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
