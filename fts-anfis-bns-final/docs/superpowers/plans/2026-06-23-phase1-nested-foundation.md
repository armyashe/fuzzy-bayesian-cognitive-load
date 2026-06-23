# Phase 1 Nested Evaluation Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Phase 1 audit, regression protection, structured config/result models, deterministic execution utilities, and a generic nested LOSO evaluation engine in parallel with the legacy runner without changing the legacy publication path.

**Architecture:** Keep the legacy `main.py -> fbn_pipeline.runner.run_one_experiment()` execution path intact and numerically stable. Add new modules for deterministic setup, typed config/result models, audit/reporting, and a separate nested engine that is exercised only with a dummy or logistic adapter on synthetic grouped data.

**Tech Stack:** Python 3.11, NumPy, pandas, scikit-learn, pytest/unittest, dataclasses, JSON/CSV/NPZ serialization.

---

### Task 1: Legacy regression harness and fixture

**Files:**
- Create: `fts-anfis-bns-final/tests/test_legacy_regression.py`
- Create: `fts-anfis-bns-final/tests/fixtures/legacy_reference.json`
- Create: `fts-anfis-bns-final/scripts/generate_legacy_fixture.py`
- Modify: `fts-anfis-bns-final/src/fbn_pipeline/runner.py`
- Modify: `fts-anfis-bns-final/src/fbn_pipeline/results.py`
- Test: `fts-anfis-bns-final/tests/test_legacy_regression.py`

- [ ] **Step 1: Write the failing regression test**

```python
def test_legacy_pipeline_matches_frozen_reference():
    current = run_legacy_snapshot_twice()
    reference = load_reference_fixture()
    assert current["rounded_posterior_hash"] == reference["rounded_posterior_hash"]
    assert current["selected_threshold"] == pytest.approx(reference["selected_threshold"], abs=1e-8)
    assert current["oof_metrics"] == pytest.approx(reference["oof_metrics"], abs=1e-8)
    assert current["test_metrics"] == pytest.approx(reference["test_metrics"], abs=1e-8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest fts-anfis-bns-final/tests/test_legacy_regression.py -q`
Expected: FAIL because the helper/fixture does not exist yet.

- [ ] **Step 3: Add the minimal legacy snapshot helpers**

```python
@dataclass(slots=True)
class LegacySnapshot:
    dataset_stats: dict[str, object]
    shapes: dict[str, list[int]]
    selected_threshold: float
    oof_metrics: dict[str, float]
    test_metrics: dict[str, float]
    rounded_posterior_hash: str
```

- [ ] **Step 4: Add the fixture generation script**

```python
first = run_legacy_snapshot(...)
second = run_legacy_snapshot(...)
if first != second:
    raise SystemExit("Legacy snapshot is not deterministic")
write_fixture(first)
```

- [ ] **Step 5: Run the targeted test and fixture generation**

Run: `python fts-anfis-bns-final/scripts/generate_legacy_fixture.py`
Expected: Writes `tests/fixtures/legacy_reference.json` only when two identical runs agree.

- [ ] **Step 6: Re-run regression test**

Run: `python -m pytest fts-anfis-bns-final/tests/test_legacy_regression.py -q`
Expected: PASS

### Task 2: Deterministic utility and compatibility wrapper

**Files:**
- Create: `fts-anfis-bns-final/src/fbn_pipeline/determinism.py`
- Modify: `fts-anfis-bns-final/src/fbn_pipeline/config.py`
- Modify: `fts-anfis-bns-final/src/fbn_pipeline/__init__.py`
- Test: `fts-anfis-bns-final/tests/test_legacy_regression.py`

- [ ] **Step 1: Write the failing deterministic test**

```python
def test_setup_seed_compatibility_wrapper_preserves_legacy_snapshot():
    baseline = run_legacy_snapshot()
    setup_seed(42)
    current = run_legacy_snapshot()
    assert current["rounded_posterior_hash"] == baseline["rounded_posterior_hash"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest fts-anfis-bns-final/tests/test_legacy_regression.py -q`
Expected: FAIL until the shared deterministic utility exists.

- [ ] **Step 3: Add a dedicated deterministic utility**

```python
def configure_determinism(seed: int, *, deterministic_threads: int = 1) -> DeterminismMetadata:
    if seed is None:
        raise ValueError("seed is required")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return DeterminismMetadata(...)
```

- [ ] **Step 4: Keep `setup_seed()` as a wrapper**

```python
def setup_seed(seed: int) -> None:
    configure_determinism(seed)
```

- [ ] **Step 5: Re-run regression coverage**

Run: `python -m pytest fts-anfis-bns-final/tests/test_legacy_regression.py -q`
Expected: PASS

### Task 3: Structured config and result models

**Files:**
- Create: `fts-anfis-bns-final/src/fbn_pipeline/structured_config.py`
- Create: `fts-anfis-bns-final/src/fbn_pipeline/nested_results.py`
- Modify: `fts-anfis-bns-final/src/fbn_pipeline/results.py`
- Modify: `fts-anfis-bns-final/src/fbn_pipeline/__init__.py`
- Test: `fts-anfis-bns-final/tests/test_structured_models.py`

- [ ] **Step 1: Write failing tests for validation and serialization**

```python
def test_evaluation_config_rejects_inner_folds_above_available_groups():
    with pytest.raises(ValueError):
        EvaluationConfig(outer_protocol="loso", inner_folds=10, ...)

def test_nested_result_roundtrip_preserves_sample_order(tmp_path):
    result = NestedEvaluationResult(...)
    save_nested_result(result, tmp_path)
    loaded = load_nested_result(tmp_path)
    assert loaded.aggregated_sample_ids == result.aggregated_sample_ids
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest fts-anfis-bns-final/tests/test_structured_models.py -q`
Expected: FAIL because the new modules do not exist yet.

- [ ] **Step 3: Add minimal typed config models with validation**

```python
@dataclass(slots=True)
class EvaluationConfig:
    outer_protocol: str
    inner_folds: int
    threshold_grid: tuple[float, ...]
    seed: int
    def __post_init__(self) -> None:
        if self.outer_protocol != "loso":
            raise ValueError(...)
```

- [ ] **Step 4: Add serializable nested result dataclasses**

```python
@dataclass(slots=True)
class OuterFoldResult:
    outer_fold_index: int
    outer_test_subject: str
    outer_test_probabilities: np.ndarray
```

- [ ] **Step 5: Re-run structured model tests**

Run: `python -m pytest fts-anfis-bns-final/tests/test_structured_models.py -q`
Expected: PASS

### Task 4: Audit document and leakage notes

**Files:**
- Create: `fts-anfis-bns-final/docs/phase1_leakage_audit.md`
- Test: none

- [ ] **Step 1: Write the audit document**

```md
## Current execution path
1. `main.py` builds `ExperimentConfig`, loads `DatasetSplit`, and calls `run_one_experiment()`.
2. `run_one_experiment()` calls `run_cv_result()` on the fixed legacy train split.
...
```

- [ ] **Step 2: Include explicit leakage statements**

```md
- BN currently receives in-sample ANFIS training probabilities for fitting and validation/test probabilities for inference.
- Threshold selection uses only OOF probabilities from the legacy train split.
- Outer LOSO does not exist in the legacy path.
```

### Task 5: FTS, ANFIS, BN, and metrics invariants

**Files:**
- Create: `fts-anfis-bns-final/tests/test_fts_invariants.py`
- Create: `fts-anfis-bns-final/tests/test_anfis_invariants.py`
- Create: `fts-anfis-bns-final/tests/test_bn_invariants.py`
- Create: `fts-anfis-bns-final/tests/test_metrics_invariants.py`
- Test: each file above

- [ ] **Step 1: Write the failing invariant tests**

```python
def test_three_fuzzy_sets_produce_nine_features_per_signal(): ...
def test_anfis_same_seed_is_deterministic(): ...
def test_bn_rows_sum_to_one(): ...
def test_threshold_tie_breaking_is_deterministic(): ...
```

- [ ] **Step 2: Run tests to verify what already passes and what is missing**

Run: `python -m pytest fts-anfis-bns-final/tests/test_fts_invariants.py fts-anfis-bns-final/tests/test_anfis_invariants.py fts-anfis-bns-final/tests/test_bn_invariants.py fts-anfis-bns-final/tests/test_metrics_invariants.py -q`
Expected: Mixed FAIL/PASS depending on missing coverage and any small helper changes needed.

- [ ] **Step 3: Add only the minimal support code needed**

```python
def stable_threshold_choice(...):
    # choose first threshold on the fixed grid that reaches the max score
```

- [ ] **Step 4: Re-run the invariant suite**

Run: `python -m pytest fts-anfis-bns-final/tests/test_fts_invariants.py fts-anfis-bns-final/tests/test_anfis_invariants.py fts-anfis-bns-final/tests/test_bn_invariants.py fts-anfis-bns-final/tests/test_metrics_invariants.py -q`
Expected: PASS

### Task 6: Nested engine and leakage guards

**Files:**
- Create: `fts-anfis-bns-final/src/fbn_pipeline/nested_evaluation.py`
- Create: `fts-anfis-bns-final/src/fbn_pipeline/nested_adapters.py`
- Create: `fts-anfis-bns-final/tests/test_nested_engine.py`
- Modify: `fts-anfis-bns-final/src/fbn_pipeline/__init__.py`
- Test: `fts-anfis-bns-final/tests/test_nested_engine.py`

- [ ] **Step 1: Write the failing nested-engine tests**

```python
def test_each_subject_is_outer_test_once(): ...
def test_outer_test_labels_do_not_change_selected_candidate(): ...
def test_deliberate_leakage_raises(): ...
def test_repeated_runs_with_same_seed_are_deterministic(): ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest fts-anfis-bns-final/tests/test_nested_engine.py -q`
Expected: FAIL because the engine and adapter do not exist yet.

- [ ] **Step 3: Add the minimal nested engine**

```python
class NestedMethodAdapter(Protocol):
    def candidate_configs(self) -> list[dict]: ...
    def fit_predict_inner_fold(...): ...
    def fit_predict_outer_fold(...): ...
```

- [ ] **Step 4: Add a dummy/logistic adapter and explicit guards**

```python
if set(inner_valid_subjects) & set(inner_train_subjects):
    raise LeakageError(...)
if len(prediction_counts) != len(outer_indices):
    raise LeakageError(...)
```

- [ ] **Step 5: Re-run nested engine tests**

Run: `python -m pytest fts-anfis-bns-final/tests/test_nested_engine.py -q`
Expected: PASS

### Task 7: Full Phase 1 verification

**Files:**
- Modify: any touched files as needed from previous tasks
- Test: all targeted tests

- [ ] **Step 1: Run the legacy regression suite**

Run: `python -m pytest fts-anfis-bns-final/tests/test_legacy_regression.py -q`
Expected: PASS

- [ ] **Step 2: Run the new targeted Phase 1 suite**

Run: `python -m pytest fts-anfis-bns-final/tests/test_structured_models.py fts-anfis-bns-final/tests/test_fts_invariants.py fts-anfis-bns-final/tests/test_anfis_invariants.py fts-anfis-bns-final/tests/test_bn_invariants.py fts-anfis-bns-final/tests/test_metrics_invariants.py fts-anfis-bns-final/tests/test_nested_engine.py -q`
Expected: PASS

- [ ] **Step 3: Run the existing test file to catch regressions**

Run: `python -m pytest fts-anfis-bns-final/tests/test_fts_encoder_dashboard.py -q`
Expected: PASS

- [ ] **Step 4: Run a legacy smoke check**

Run: `python fts-anfis-bns-final/main.py --preset THESIS_FINAL --ablation exp1 --no-save-csv`
Expected: completes successfully with unchanged threshold, metrics, and rounded posterior hash versus the frozen fixture.

- [ ] **Step 5: Record any failed or skipped checks honestly**

```text
If a command fails due to environment or dependency issues, capture the exact command and error in the final report instead of claiming success.
```
