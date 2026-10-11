from __future__ import annotations

import sys
from pathlib import Path


def _resolve_project_root() -> Path:
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    return Path.cwd().resolve()


PROJECT_ROOT = _resolve_project_root()
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from fbn_pipeline.publication_selection import run_publication_joint_selection


def main() -> None:
    result = run_publication_joint_selection(PROJECT_ROOT, print_progress=True)
    selected = result["selected_result"]
    print("\nJOINT PUBLICATION SELECTION COMPLETE")
    print(f"Selected configuration : {selected.configuration_id}")
    print(f"Membership function    : {selected.config.mf_type}")
    print(f"Rules                  : {int(selected.config.n_rules)}")
    print(f"OOF threshold          : {selected.threshold:.4f}")
    print(f"OOF balanced accuracy  : {selected.oof_metrics['bal_acc']:.4f}")
    print(f"Joint summary          : {result['artifacts']['summary_path']}")
    print(f"Updated lock           : {result['paths']['lock_path']}")


if __name__ == "__main__":
    main()
