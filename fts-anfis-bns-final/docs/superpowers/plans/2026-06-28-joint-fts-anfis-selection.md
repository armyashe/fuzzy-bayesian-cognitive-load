# Joint FTS-ANFIS Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the sequential publication-time FTS membership and ANFIS rule selection flow with a single joint 16-configuration OOF selection protocol, while preserving legacy artifacts for traceability and keeping final hold-out evaluation lock-driven.

**Architecture:** Add a publication-joint-selection orchestration layer that reuses the existing `run_development`, `select_best_development_config`, and lock/retrain flow. Extend CV/development helpers to support shared fold assignments, strict OOF coverage validation, atomic lock writes, and publication-specific artifact generation for the 16 `(mf_type, n_rules)` combinations.

**Tech Stack:** Python, pandas, numpy, sklearn `GroupKFold`, existing `fbn_pipeline` runner/pipeline modules, unittest/pytest.

---

### Task 1: Add tests for joint selection validation and reporting

**Files:**
- Create: `tests/test_joint_fts_anfis_selection.py`
- Modify: `tests/test_final_evaluation_summary.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_joint_selection_requires_all_16_configurations():
    ...

def test_joint_selection_rejects_incomplete_oof_or_leakage():
    ...

def test_joint_selection_report_mentions_holdout_exclusion_and_selected_config():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\doqua\AppData\Local\Programs\Python\Python311\python.exe -m pytest tests\test_joint_fts_anfis_selection.py tests\test_final_evaluation_summary.py -q`
Expected: FAIL because joint-selection helpers and report fields do not exist yet.

- [ ] **Step 3: Keep test fixtures lightweight**

Use fake `DevelopmentResult` objects and synthetic fold indices instead of full ANFIS training so validation logic is tested directly and deterministically.

- [ ] **Step 4: Re-run focused tests after edits**

Run the same pytest command and expect all joint-selection unit tests to pass.


### Task 2: Extend CV/development helpers for shared folds and atomic locks

**Files:**
- Modify: `src/fbn_pipeline/pipeline.py`
- Modify: `src/fbn_pipeline/runner.py`

- [ ] **Step 1: Add optional shared fold support**

Extend `run_cv_result(...)` and `run_development(...)` to accept precomputed `fold_indices` so all 16 configurations use exactly the same fold assignments.

- [ ] **Step 2: Add fold provenance and OOF validation helpers**

Implement helpers for:
- computing deterministic fold assignments from `(X_train, y_train, user_train, k_folds)`
- validating no participant overlap between train/validation in each fold
- validating every development sample appears exactly once in validation
- validating exactly 632 OOF predictions per configuration
- hashing fold assignments for provenance

- [ ] **Step 3: Harden lock writing**

Change `lock_development_config(...)` to write to a sibling temp file and atomically replace the target only after payload validation completes.

- [ ] **Step 4: Keep ranking policy explicit**

Ensure runner-level selection remains:
- balanced accuracy
- macro-F1
- ROC-AUC
- lower model complexity
- configuration ID lexicographic


### Task 3: Implement publication joint-selection orchestration

**Files:**
- Create: `src/fbn_pipeline/publication_selection.py`
- Modify: `src/fbn_pipeline/__init__.py`

- [ ] **Step 1: Add the public orchestration function**

Implement:

```python
def run_joint_fts_anfis_selection(
    *,
    training_split,
    artifact_root: Path,
    membership_functions: list[str],
    rule_sizes: list[int],
    seed: int = 42,
) -> dict[str, Any]:
    ...
```

Return:
- `long_form_df`
- `matrix_df`
- `selection`
- `selected_result`
- `selected_threshold`
- `fold_indices`
- `fold_assignment_hash`
- `created_files`

- [ ] **Step 2: Fix the search space**

Only evaluate the 16 publication configurations:
- `mf_type in {"triangular", "gaussian", "trapezoid", "gbell"}`
- `n_rules in {3, 6, 8, 12}`
- `bn_structure="naive"`
- `bn_soft=True`
- `n_bins=5`
- `alpha=0.5`

- [ ] **Step 3: Reuse existing development flow**

For each configuration:
- derive config from publication defaults
- run `run_development(...)` with shared `fold_indices`
- validate OOF coverage and no leakage
- collect metrics and provenance

- [ ] **Step 4: Fail closed**

Raise clear `ValueError` if:
- fewer than 16 configurations complete
- any configuration has NaN or missing OOF predictions
- any sample appears zero or multiple times in validation
- any participant leakage is detected


### Task 4: Generate joint-selection artifacts and preserve legacy sequential artifacts

**Files:**
- Modify: `src/fbn_pipeline/publication_selection.py`

- [ ] **Step 1: Create new publication artifacts**

Write:
- `artifacts/publication_protocol_v2/development/joint_fts_anfis_selection_oof.csv`
- `artifacts/publication_protocol_v2/development/joint_fts_anfis_balanced_accuracy_matrix.csv`
- `artifacts/publication_protocol_v2/development/joint_fts_anfis_selection_report.md`
- `artifacts/publication_protocol_v2/development/joint_fts_anfis_selection_summary.json`

- [ ] **Step 2: Mark legacy sequential artifacts**

Create:
- `artifacts/publication_protocol_v2/development/legacy_sequential_selection/README.md`

Preserve existing sequential artifacts by moving or copying them into the legacy directory without overwriting existing content. Make the README state they are exploratory only and not used for publication locking.

- [ ] **Step 3: Include provenance**

Record in summary/lock inputs:
- configuration IDs
- selected `mf_type`
- selected `n_rules`
- selected threshold
- OOF balanced accuracy
- tie-break rules
- seed
- fold-assignment hash
- source artifact path
- prediction/config hashes when available


### Task 5: Switch publication lock and final-evaluation flow to joint artifacts

**Files:**
- Modify: `src/fbn_pipeline/final_evaluation.py`
- Modify: `src/fbn_pipeline/runner.py`

- [ ] **Step 1: Remove hardcoded proposed model ID dependency**

Use `locked_config.json` as the sole source of proposed model configuration ID during final evaluation.

- [ ] **Step 2: Update development-summary loading**

Make `final_evaluation.py` read the new joint-selection summary/report artifacts. If joint-selection artifacts are missing, fail with a clear message and do not read legacy sequential files.

- [ ] **Step 3: Extend lock metadata**

Ensure the new proposed lock records:
- selected `mf_type`
- selected `n_rules`
- selected OOF threshold
- OOF balanced accuracy
- tie-break rules
- seed
- fold-assignment provenance
- source joint-selection artifact

- [ ] **Step 4: Keep final retraining lock-driven**

Continue to use `run_final_evaluation_from_lock(...)`, but only with the new joint-selection-derived lock.


### Task 6: Add a clear command surface for the new publication flow

**Files:**
- Create: `run_publication_joint_selection.py`
- Modify: `README.md`

- [ ] **Step 1: Add a simple CLI entrypoint**

Expose a command that:
- loads the training split
- runs joint selection
- validates artifacts
- atomically updates `artifacts/publication_protocol_v2/locked_config.json`

- [ ] **Step 2: Document usage**

Add a README section with:
- the new command
- output artifact paths
- the statement that legacy sequential artifacts are reference-only


### Task 7: Verification

**Files:**
- Test: `tests/test_joint_fts_anfis_selection.py`
- Test: `tests/test_final_evaluation_summary.py`

- [ ] **Step 1: Import smoke test**

Run:

```bash
C:\Users\doqua\AppData\Local\Programs\Python\Python311\python.exe -c "import sys; sys.path.insert(0, 'src'); import fbn_pipeline; import fbn_pipeline.publication_selection"
```

Expected: no import errors

- [ ] **Step 2: Run focused unit tests**

Run:

```bash
C:\Users\doqua\AppData\Local\Programs\Python\Python311\python.exe -m pytest tests\test_joint_fts_anfis_selection.py tests\test_final_evaluation_summary.py -q
```

Expected: PASS

- [ ] **Step 3: Run protocol validation before full 16-config execution**

Run the lightweight validation path that checks:
- 16 expected configuration IDs
- no hold-out data usage
- shared folds
- atomic-lock preconditions

- [ ] **Step 4: Only then run full joint selection**

Run the CLI command and confirm:
- all 16 configurations complete
- each has exactly 632 OOF predictions
- no participant leakage
- lock updates only after artifacts are written and validated

