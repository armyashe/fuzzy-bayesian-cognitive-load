# Phase 2 Production Nested Methods Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the production nested publication pipeline in parallel with the legacy runner using a shared persisted fold manifest, threshold-optimized candidate ranking, cross-fitted BN evidence, and controlled baseline adapters.

**Architecture:** Keep the legacy publication path untouched. Extend the Phase 1 nested stack with one shared fold-manifest layer, one engine policy fix, reusable production helpers for grouped FTS/model fitting and cross-fitted BN evidence, then add production adapters for the proposed FTS-ANFIS-BN model and the requested baselines.

**Tech Stack:** Python 3.11, NumPy, pandas, scikit-learn, existing ANFIS/BN/FTS modules, JSON/CSV/NPZ serialization, pytest.

---

### Task 1: Shared fold manifest and engine policy update

**Files:**
- Create: `fts-anfis-bns-final/src/fbn_pipeline/fold_manifest.py`
- Modify: `fts-anfis-bns-final/src/fbn_pipeline/nested_evaluation.py`
- Modify: `fts-anfis-bns-final/src/fbn_pipeline/nested_results.py`
- Modify: `fts-anfis-bns-final/src/fbn_pipeline/__init__.py`
- Create: `fts-anfis-bns-final/tests/test_phase2_manifest_and_policy.py`
- Test: `fts-anfis-bns-final/tests/test_phase2_manifest_and_policy.py`

- [ ] **Step 1: Write the failing manifest test**

```python
def test_shared_manifest_persists_dataset_hash_and_fold_assignments(tmp_path):
    manifest = build_subject_loso_manifest(dataset, evaluation, output_path=tmp_path)
    loaded = load_fold_manifest(tmp_path / "fold_manifest.json")
    assert loaded.dataset_order_hash == manifest.dataset_order_hash
    assert loaded.outer_folds[0].outer_test_subject == manifest.outer_folds[0].outer_test_subject
```

- [ ] **Step 2: Write the failing candidate-ranking policy test**

```python
def test_candidate_ranking_uses_threshold_optimized_inner_oof_balanced_accuracy():
    result = run_nested_subject_loso(dataset=dataset, adapter=adapter, evaluation=evaluation, manifest=manifest)
    assert result.outer_fold_results[0].selected_candidate_id == "expected_candidate"
    assert result.outer_fold_results[0].selected_threshold == expected_threshold
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest fts-anfis-bns-final/tests/test_phase2_manifest_and_policy.py -q`
Expected: FAIL because the manifest layer and ranking policy support do not exist yet.

- [ ] **Step 4: Implement the shared manifest**

```python
@dataclass(slots=True)
class FoldManifest:
    dataset_order_hash: str
    outer_folds: list[OuterFoldManifest]

def build_subject_loso_manifest(dataset, evaluation, output_path):
    # compute stable sample indices, subject IDs, labels, seeds, inner fold assignments
```

- [ ] **Step 5: Update the nested engine to consume the persisted manifest and rank by threshold-optimized inner OOF balanced accuracy**

```python
threshold = _select_threshold(y_train, oof_scores, evaluation.threshold_grid)
ranking_bal_acc = balanced_accuracy_score(y_train, (oof_scores >= threshold).astype(int))
```

- [ ] **Step 6: Re-run the targeted tests**

Run: `python -m pytest fts-anfis-bns-final/tests/test_phase2_manifest_and_policy.py -q`
Expected: PASS

### Task 2: Full-subject nested dataset loader and artifact rows

**Files:**
- Modify: `fts-anfis-bns-final/src/fbn_pipeline/data.py`
- Create: `fts-anfis-bns-final/src/fbn_pipeline/production_data.py`
- Create: `fts-anfis-bns-final/src/fbn_pipeline/production_artifacts.py`
- Test: `fts-anfis-bns-final/tests/test_phase2_manifest_and_policy.py`

- [ ] **Step 1: Write the failing full-dataset loader test**

```python
def test_production_dataset_loader_returns_all_23_subjects():
    dataset = load_full_subject_dataset(data_dir)
    assert len(np.unique(dataset.subject_ids)) == 23
    assert len(dataset.sample_ids) == len(dataset.y)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest fts-anfis-bns-final/tests/test_phase2_manifest_and_policy.py -q`
Expected: FAIL because no full nested dataset loader exists.

- [ ] **Step 3: Implement the full-subject dataset loader and artifact row helpers**

```python
def load_full_subject_dataset(data_folder_path: str | Path) -> NestedDataset:
    # reuse Excel loading, preserve dataset order, assign stable sample IDs/indexes
```

- [ ] **Step 4: Re-run the targeted tests**

Run: `python -m pytest fts-anfis-bns-final/tests/test_phase2_manifest_and_policy.py -q`
Expected: PASS

### Task 3: Cross-fitted BN evidence and faithful soft-evidence explanation

**Files:**
- Create: `fts-anfis-bns-final/src/fbn_pipeline/production_bn.py`
- Modify: `fts-anfis-bns-final/src/fbn_pipeline/models/bn_fusion.py`
- Create: `fts-anfis-bns-final/tests/test_phase2_bn_crossfit.py`
- Test: `fts-anfis-bns-final/tests/test_phase2_bn_crossfit.py`

- [ ] **Step 1: Write the failing cross-fit coverage test**

```python
def test_cross_fitted_evidence_gives_each_bn_training_sample_exactly_one_vector():
    evidence = generate_cross_fitted_signal_evidence(train_data, manifest_fold, candidate, evaluation)
    assert len(evidence.rows) == len(train_data.y)
    assert np.all(evidence.coverage_counts == 1)
```

- [ ] **Step 2: Write the failing BN-source test**

```python
def test_bn_is_fitted_from_cross_fitted_evidence_not_final_in_sample_predictions():
    result = fit_bn_with_cross_fitted_evidence(...)
    assert result.bn_training_source == "cross_fitted_signal_evidence"
```

- [ ] **Step 3: Write the failing explanation-equivalence test**

```python
def test_soft_evidence_explanation_reconstructs_predict_proba():
    posterior = bn.predict_proba(scores)[0]
    trace = bn.explain_sample(scores[0], threshold=0.5)
    assert trace["posterior_task"] == pytest.approx(posterior, abs=1e-8)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `python -m pytest fts-anfis-bns-final/tests/test_phase2_bn_crossfit.py -q`
Expected: FAIL because cross-fitted BN helpers and faithful explanation fields are missing.

- [ ] **Step 5: Implement grouped cross-fitted BN evidence generation**

```python
def generate_cross_fitted_signal_evidence(train_data, train_subjects, model_builder, n_splits, seed):
    # fit fresh FTS + 4 ANFIS models per cross-fit split, reassemble one evidence row per sample
```

- [ ] **Step 6: Implement BN fit-from-crossfit and update explanation output**

```python
trace = {
    "prior_log_odds": ...,
    "cumulative_contribution": ...,
    "selected_threshold": threshold,
    "predicted_class": ...
}
```

- [ ] **Step 7: Re-run the targeted tests**

Run: `python -m pytest fts-anfis-bns-final/tests/test_phase2_bn_crossfit.py -q`
Expected: PASS

### Task 4: Production proposed-model adapter

**Files:**
- Create: `fts-anfis-bns-final/src/fbn_pipeline/production_adapters.py`
- Modify: `fts-anfis-bns-final/src/fbn_pipeline/nested_adapters.py`
- Create: `fts-anfis-bns-final/tests/test_phase2_proposed_adapter.py`
- Test: `fts-anfis-bns-final/tests/test_phase2_proposed_adapter.py`

- [ ] **Step 1: Write the failing proposed-adapter leakage test**

```python
def test_proposed_adapter_fits_fts_and_anfis_on_training_subjects_only():
    probs, artifacts = adapter.fit_predict_inner_fold(...)
    assert artifacts["fts_feature_dim"] == 36
    assert artifacts["per_signal_dims"] == [9, 9, 9, 9]
```

- [ ] **Step 2: Write the failing independent-MF test**

```python
def test_fts_based_pipelines_select_membership_functions_independently():
    assert proposed_selected_mf != lr_selected_mf or both_valid_choices
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest fts-anfis-bns-final/tests/test_phase2_proposed_adapter.py -q`
Expected: FAIL because the production proposed adapter does not exist.

- [ ] **Step 4: Implement the 16-candidate proposed adapter using cross-fitted BN evidence**

```python
class ProposedFTSANFISBNAdapter:
    # 4 MF types x 4 rule counts, fixed BN settings, cross-fitted BN training in inner and outer fits
```

- [ ] **Step 5: Re-run the targeted tests**

Run: `python -m pytest fts-anfis-bns-final/tests/test_phase2_proposed_adapter.py -q`
Expected: PASS

### Task 5: FTS-LR and FTS-RBF-SVM baselines

**Files:**
- Modify: `fts-anfis-bns-final/src/fbn_pipeline/production_adapters.py`
- Create: `fts-anfis-bns-final/tests/test_phase2_baselines.py`
- Test: `fts-anfis-bns-final/tests/test_phase2_baselines.py`

- [ ] **Step 1: Write the failing LR/SVM isolation test**

```python
def test_lr_and_svm_use_shared_manifest_and_training_only_preprocessing():
    result = run_nested_subject_loso(..., manifest=manifest)
    assert result.outer_fold_results[0].selected_candidate_id.startswith(("fts_lr", "fts_svm"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest fts-anfis-bns-final/tests/test_phase2_baselines.py -q`
Expected: FAIL because the baseline production adapters do not exist.

- [ ] **Step 3: Implement FTS-LR and FTS-RBF-SVM adapters**

```python
class FTSLogisticRegressionAdapter: ...
class FTSRBFSVMAdapter: ...
```

- [ ] **Step 4: Re-run the targeted tests**

Run: `python -m pytest fts-anfis-bns-final/tests/test_phase2_baselines.py -q`
Expected: PASS

### Task 6: Optional XGBoost baseline

**Files:**
- Modify: `fts-anfis-bns-final/src/fbn_pipeline/production_adapters.py`
- Modify: `fts-anfis-bns-final/tests/test_phase2_baselines.py`
- Test: `fts-anfis-bns-final/tests/test_phase2_baselines.py`

- [ ] **Step 1: Detect availability and write a conditional failing test**

```python
def test_xgboost_adapter_runs_when_dependency_is_available():
    if not HAS_XGBOOST:
        pytest.skip("xgboost not installed")
```

- [ ] **Step 2: Run tests to see whether this is a skip or a real failure**

Run: `python -m pytest fts-anfis-bns-final/tests/test_phase2_baselines.py -q`
Expected: either SKIP if unavailable or FAIL until the adapter is implemented.

- [ ] **Step 3: Implement the adapter only if the dependency is available or can be added cleanly**

```python
class FTSXGBoostAdapter: ...
```

- [ ] **Step 4: Re-run the targeted tests**

Run: `python -m pytest fts-anfis-bns-final/tests/test_phase2_baselines.py -q`
Expected: PASS or SKIP

### Task 7: Optional raw 1D-CNN baseline

**Files:**
- Modify: `fts-anfis-bns-final/src/fbn_pipeline/production_adapters.py`
- Modify: `fts-anfis-bns-final/tests/test_phase2_baselines.py`
- Test: `fts-anfis-bns-final/tests/test_phase2_baselines.py`

- [ ] **Step 1: Detect whether an existing DL framework is already present**

```python
HAS_DL = importlib.util.find_spec("torch") is not None
```

- [ ] **Step 2: If absent, add an explicit unfinished-path test**

```python
def test_cnn_adapter_is_explicitly_unfinished_without_existing_dl_framework():
    pytest.skip("No existing deep-learning framework available")
```

- [ ] **Step 3: If present and safe, implement the grouped early-stopping adapter**

```python
class Raw1DCNNAdapter: ...
```

- [ ] **Step 4: Re-run the targeted tests**

Run: `python -m pytest fts-anfis-bns-final/tests/test_phase2_baselines.py -q`
Expected: PASS or SKIP

### Task 8: Reduced smoke runs and final verification

**Files:**
- Modify: any touched files from previous tasks
- Test: focused suite plus reduced smoke commands

- [ ] **Step 1: Run Phase 1 legacy regression again**

Run: `python -m pytest fts-anfis-bns-final/tests/test_legacy_regression.py -q`
Expected: PASS with the frozen fixture unchanged.

- [ ] **Step 2: Run the Phase 1 nested-engine tests**

Run: `python -m pytest fts-anfis-bns-final/tests/test_nested_engine.py -q`
Expected: PASS

- [ ] **Step 3: Run the Phase 2 focused tests**

Run: `python -m pytest fts-anfis-bns-final/tests/test_phase2_manifest_and_policy.py fts-anfis-bns-final/tests/test_phase2_bn_crossfit.py fts-anfis-bns-final/tests/test_phase2_proposed_adapter.py fts-anfis-bns-final/tests/test_phase2_baselines.py -q`
Expected: PASS with optional skips for unavailable XGBoost/CNN.

- [ ] **Step 4: Run one reduced proposed-model nested smoke run**

Run: `python -m fbn_pipeline.phase2_smoke --method proposed --inner-folds 2 --mf gaussian --rules 3 --max-outer-folds 2`
Expected: completes, persists fold predictions and selected configs, and uses cross-fitted BN evidence.

- [ ] **Step 5: Run reduced smoke runs for implemented baselines**

Run: `python -m fbn_pipeline.phase2_smoke --method fts_lr --inner-folds 2 --max-outer-folds 2`
Expected: completes successfully.

Run: `python -m fbn_pipeline.phase2_smoke --method fts_svm --inner-folds 2 --max-outer-folds 2`
Expected: completes successfully.

Run: `python -m fbn_pipeline.phase2_smoke --method fts_xgb --inner-folds 2 --max-outer-folds 2`
Expected: completes only if XGBoost is implemented; otherwise skip and report honestly.

- [ ] **Step 6: If CNN is implemented safely, run one reduced smoke run**

Run: `python -m fbn_pipeline.phase2_smoke --method raw_cnn --inner-folds 2 --max-outer-folds 2 --epochs 5`
Expected: completes only if the existing DL framework path is available and safe.
