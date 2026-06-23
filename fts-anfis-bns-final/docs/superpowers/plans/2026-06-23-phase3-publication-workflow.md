# Phase 3 Publication Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Phase 1/2 nested evaluation stack into a publication-ready Phase 3 workflow with leakage-safe ablations, strict artifact provenance, reproducible statistics, and report generation from saved artifacts only.

**Architecture:** Reuse the existing nested LOSO engine, fold manifest, adapters, and persistence path. Add one publication-layer orchestration module that owns stable run directories, metadata, artifact schemas, ablation execution, statistics, and report generation. Keep smoke/publication separation explicit in metadata and directory policy, and keep report generation artifact-driven so tables and figures never retrain models.

**Tech Stack:** Python 3.12, NumPy, Pandas, SciPy, scikit-learn, matplotlib, pytest, existing `fbn_pipeline` Phase 1/2 modules.

---

### Task 1: Add Phase 3 run layout and provenance helpers

**Files:**
- Create: `src/fbn_pipeline/publication_artifacts.py`
- Modify: `src/fbn_pipeline/production_runner.py`
- Test: `tests/test_phase3_reporting.py`

- [ ] **Step 1: Write the failing tests**
- [ ] Verify a run root `artifacts/phase3_publication/<run_name>/` is required, stable, and rejects overwrite by default.
- [ ] Verify metadata records `run_name`, `mode`, `allow_partial_outer_folds`, `seed`, `configuration_hash`, `dataset_order_hash`, timestamp, Python version, and package versions.
- [ ] Verify smoke/publication metadata differ and partial runs cannot advertise complete publication coverage.

### Task 2: Add artifact-driven metric utilities with explicit undefined handling

**Files:**
- Create: `src/fbn_pipeline/publication_metrics.py`
- Modify: `src/fbn_pipeline/metrics.py`
- Test: `tests/test_phase3_reporting.py`

- [ ] **Step 1: Write the failing tests**
- [ ] Verify pooled metrics operate on aggregated outer predictions only.
- [ ] Verify per-subject rows stay aligned to subject IDs and sample membership.
- [ ] Verify one-class AUC is surfaced as `NaN` with an explicit undefined-reason field.

### Task 3: Add subject-cluster bootstrap and paired statistics

**Files:**
- Create: `src/fbn_pipeline/publication_statistics.py`
- Test: `tests/test_phase3_reporting.py`

- [ ] **Step 1: Write the failing tests**
- [ ] Verify bootstrap resamples subject clusters rather than individual rows.
- [ ] Verify paired comparisons align by subject ID and preserve method pairing.
- [ ] Verify Wilcoxon outputs sample size, statistic, p-value, and effect size.

### Task 4: Add leakage-safe ablation execution

**Files:**
- Modify: `src/fbn_pipeline/ablation.py`
- Create: `src/fbn_pipeline/publication_ablations.py`
- Modify: `src/fbn_pipeline/production_adapters.py`
- Test: `tests/test_phase3_execution.py`

- [ ] **Step 1: Write the failing tests**
- [ ] Verify ablations reuse the selected proposed-model outer-fold configuration.
- [ ] Verify only the intended component changes for `FTS–ANFIS–Mean` and `FTS Summary–BN`.
- [ ] Verify ablation thresholds come from inner OOF predictions under the frozen structural configuration.

### Task 5: Extend persistence into publication artifact groups

**Files:**
- Modify: `src/fbn_pipeline/production_runner.py`
- Create: `src/fbn_pipeline/publication_runner.py`
- Test: `tests/test_phase3_execution.py`

- [ ] **Step 1: Write the failing tests**
- [ ] Verify publication outputs are written into `metadata/`, `predictions/`, `metrics/`, `statistics/`, `interpretability/`, `tables/`, and `figures/`.
- [ ] Verify duplicate or missing publication predictions raise errors.
- [ ] Verify all methods can be joined by `sample_id` and share one manifest hash.

### Task 6: Add interpretability and stability exports

**Files:**
- Create: `src/fbn_pipeline/publication_interpretability.py`
- Modify: `src/fbn_pipeline/production_bn.py`
- Modify: `src/fbn_pipeline/production_adapters.py`
- Test: `tests/test_phase3_reporting.py`

- [ ] **Step 1: Write the failing tests**
- [ ] Verify ANFIS reconstruction reproduces the raw output within tolerance.
- [ ] Verify BN explanation reconstruction reproduces the posterior probability within tolerance.

### Task 7: Add artifact-only table and figure generation

**Files:**
- Create: `src/fbn_pipeline/publication_reports.py`
- Create: `scripts/generate_publication_outputs.py`
- Test: `tests/test_phase3_reports.py`

- [ ] **Step 1: Write the failing tests**
- [ ] Verify table generation reads saved artifacts and does not import model-training code paths.
- [ ] Verify figure generation reads saved artifacts and does not retrain.

### Task 8: Add CLI entry points

**Files:**
- Create: `scripts/run_nested_proposed.py`
- Create: `scripts/run_nested_baselines.py`
- Create: `scripts/run_nested_ablations.py`
- Modify: `scripts/run_phase2_smoke.py`
- Test: `tests/test_phase3_reports.py`

- [ ] **Step 1: Write the failing tests**
- [ ] Verify import smoke for all publication scripts.
- [ ] Verify publication mode defaults to strict complete coverage and smoke mode must be explicit.

### Task 9: Update reproducibility documentation

**Files:**
- Modify: `README.md`
- Modify: `requirements.txt`

- [ ] **Step 1: Document the stable run root**
- [ ] Describe dataset placement, Python 3.12 usage, dependency versions, candidate grids, ablations, confidence intervals, statistical tests, artifact layout, table/figure regeneration, smoke/publication distinction, and known limitations.

### Task 10: Verification

**Files:**
- Modify: `tests/test_phase3_execution.py`
- Modify: `tests/test_phase3_reporting.py`
- Modify: `tests/test_phase3_reports.py`

- [ ] **Step 1: Run the existing Phase 1/2 target suite**
- [ ] **Step 2: Run focused Phase 3 tests**
- [ ] **Step 3: Run reduced ablation smoke**
- [ ] **Step 4: Run reduced statistics/report-generation smoke**
- [ ] **Step 5: Regenerate tables and figures from reduced artifacts**
- [ ] **Step 6: Re-run legacy regression**
- [ ] **Step 7: Run import smoke for all publication scripts**

