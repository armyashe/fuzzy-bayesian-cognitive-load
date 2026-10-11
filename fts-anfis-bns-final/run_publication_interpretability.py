from __future__ import annotations

import json
import sys
from pathlib import Path


def _resolve_project_root() -> Path:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


PROJECT_ROOT = _resolve_project_root()

from fbn_pipeline.interpretability import run_publication_protocol_v2_interpretability  # noqa: E402


def main() -> None:
    result = run_publication_protocol_v2_interpretability(PROJECT_ROOT)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
