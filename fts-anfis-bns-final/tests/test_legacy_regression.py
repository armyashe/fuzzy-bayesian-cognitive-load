from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fbn_pipeline.legacy_snapshot import run_legacy_snapshot_twice
from fbn_pipeline import setup_seed


FIXTURE_PATH = ROOT / "tests" / "fixtures" / "legacy_reference.json"


def load_reference_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_legacy_pipeline_matches_frozen_reference() -> None:
    current = run_legacy_snapshot_twice()
    reference = load_reference_fixture()

    assert current["dataset_stats"] == reference["dataset_stats"]
    assert current["shapes"] == reference["shapes"]
    assert current["selected_threshold"] == pytest.approx(
        reference["selected_threshold"], abs=1e-8
    )
    assert current["oof_metrics"] == pytest.approx(reference["oof_metrics"], abs=1e-8)
    assert current["test_metrics"] == pytest.approx(reference["test_metrics"], abs=1e-8)
    assert current["rounded_posterior_hash"] == reference["rounded_posterior_hash"]


def test_setup_seed_compatibility_wrapper_preserves_legacy_snapshot() -> None:
    baseline = run_legacy_snapshot_twice()
    setup_seed(42)
    current = run_legacy_snapshot_twice()
    assert current["selected_threshold"] == pytest.approx(baseline["selected_threshold"], abs=1e-8)
    assert current["oof_metrics"] == pytest.approx(baseline["oof_metrics"], abs=1e-8)
    assert current["test_metrics"] == pytest.approx(baseline["test_metrics"], abs=1e-8)
    assert current["rounded_posterior_hash"] == baseline["rounded_posterior_hash"]
