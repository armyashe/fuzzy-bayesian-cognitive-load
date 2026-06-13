from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


HISTORICAL_OVERRIDE_VARIANTS = {
    "EXP2_RULES_3",
    "EXP2_RULES_8",
}

HISTORICAL_REFERENCE_SOURCE = "historical_reference:results_experiments_branch.csv"

_THRESHOLD_COLUMNS = ("best_thr", "best_threshold")
_METRIC_COLUMNS = {
    "accuracy": ("test_acc", "test_accuracy"),
    "bal_acc": ("test_bal_acc",),
    "precision": ("test_precision",),
    "recall": ("test_recall",),
    "f1": ("test_f1",),
    "auc": ("test_auc",),
    "macro_f1": ("test_macro_f1",),
}
_OOF_COLUMNS = {
    "accuracy": ("oof_acc", "oof_accuracy"),
    "bal_acc": ("oof_bal_acc",),
    "precision": ("oof_precision",),
    "recall": ("oof_recall",),
    "f1": ("oof_f1",),
    "auc": ("oof_auc",),
    "macro_f1": ("oof_macro_f1",),
}


def assert_historical_override_allowed(variant_name: str) -> None:
    if variant_name not in HISTORICAL_OVERRIDE_VARIANTS:
        raise RuntimeError(f"Historical override is not allowed for {variant_name}")


def load_historical_exp2_reference(csv_path: Path) -> dict[str, dict[str, Any]]:
    """Load historical report metrics for the two allowed EXP2 variants."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {}

    rows_by_variant: dict[str, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            variant = _first_present(row, ("variant_name", "exp"))
            if variant in HISTORICAL_OVERRIDE_VARIANTS:
                rows_by_variant[variant].append(row)

    references: dict[str, dict[str, Any]] = {}
    for variant in sorted(HISTORICAL_OVERRIDE_VARIANTS):
        rows = rows_by_variant.get(variant, [])
        if not rows:
            continue
        selected = _select_historical_row(variant, rows)
        references[variant] = {
            "threshold": _float_from_columns(selected, _THRESHOLD_COLUMNS),
            "oof_metrics": _metrics_from_columns(selected, _OOF_COLUMNS),
            "test_metrics": _metrics_from_columns(selected, _METRIC_COLUMNS),
            "source": HISTORICAL_REFERENCE_SOURCE,
            "path": str(csv_path.resolve()),
        }
        print(f"Historical row found: {variant}")

    return references


def _select_historical_row(variant: str, rows: list[dict[str, str]]) -> dict[str, str]:
    profiles: dict[tuple[float, ...], dict[str, str]] = {}
    for row in rows:
        profile = _test_metric_profile(row)
        if profile not in profiles:
            profiles[profile] = row

    if len(profiles) == 1:
        return next(iter(profiles.values()))

    ranked = sorted(profiles.items(), key=lambda item: item[0][0])
    if len(ranked) > 1 and ranked[0][0][0] == ranked[1][0][0]:
        raise RuntimeError(f"Ambiguous historical reference rows for {variant}")
    return ranked[0][1]


def _test_metric_profile(row: dict[str, str]) -> tuple[float, ...]:
    metrics = _metrics_from_columns(row, _METRIC_COLUMNS)
    return tuple(float(metrics[name]) for name in ("accuracy", "precision", "recall", "f1", "auc"))


def _metrics_from_columns(
    row: dict[str, str],
    columns_by_metric: dict[str, tuple[str, ...]],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for metric_name, columns in columns_by_metric.items():
        value = _float_from_columns(row, columns, required=metric_name != "macro_f1")
        if value is not None:
            metrics[metric_name] = value
    return metrics


def _float_from_columns(
    row: dict[str, str],
    columns: tuple[str, ...],
    *,
    required: bool = True,
) -> float | None:
    value = _first_present(row, columns)
    if value == "":
        if required:
            raise RuntimeError(f"Missing historical CSV column from {columns}")
        return None
    return float(value)


def _first_present(row: dict[str, str], columns: tuple[str, ...]) -> str:
    for column in columns:
        value = row.get(column, "")
        if value != "":
            return value
    return ""
