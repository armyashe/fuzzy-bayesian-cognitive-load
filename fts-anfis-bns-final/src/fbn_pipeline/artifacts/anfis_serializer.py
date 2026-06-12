from __future__ import annotations

from typing import Any

from ..console_reporting import feature_names_for_fts
from ..models.anfis_tsk import ANFIS_TSK
from ..models.bn_fusion import SIGNAL_NAMES


def build_anfis_rules_artifact(
    anfis_list: list[ANFIS_TSK] | None,
    n_sets: int,
) -> dict[str, Any]:
    """Build a JSON-serializable ANFIS rule artifact."""
    if anfis_list is None:
        return {}

    feature_names = feature_names_for_fts(n_sets)
    signal_names = (
        SIGNAL_NAMES
        if len(anfis_list) == len(SIGNAL_NAMES)
        else [f"S{s}" for s in range(len(anfis_list))]
    )

    rules_dict: dict[str, list[str]] = {}
    for s, sig_name in enumerate(signal_names):
        if s < len(anfis_list):
            rules_dict[sig_name] = anfis_list[s].rule_strings_full(
                feature_names=feature_names,
            )
    return rules_dict
