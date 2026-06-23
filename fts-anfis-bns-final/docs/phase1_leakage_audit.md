# Phase 1 Leakage Audit

## Current execution path

1. `main.py`
   Builds `ExperimentConfig`, applies a preset, loads the dataset from `last_30s_segments/`, and calls `fbn_pipeline.runner.run_one_experiment()`.

2. `src/fbn_pipeline/data.py`
   Loads `labels.xlsx`, `temp.xlsx`, `hr.xlsx`, `gsr.xlsx`, and `rr.xlsx`.
   Applies the fixed legacy subject split:
   - training subjects: `TRAIN_IDS`
   - test subjects: `TEST_IDS`
   Binarizes labels into `0=REST`, `1=TASK`.

3. `src/fbn_pipeline/runner.py`
   Calls `setup_seed(cfg.seed)`, then `run_cv_result()` on the fixed legacy training partition only.

4. `src/fbn_pipeline/pipeline.py::run_cv_result()`
   Runs `GroupKFold(n_splits=cfg.k_folds)` over `dataset.X_train`, `dataset.y_train`, and `dataset.user_train`.
   For each fold:
   - splits inner train and validation subjects;
   - extracts features with `extract_features_for_ablation()`;
   - fits ANFIS or BN ablation components on the fold-training subset;
   - predicts the fold-validation subset;
   - writes validation posteriors into the OOF array.

5. `src/fbn_pipeline/stages.py::extract_features_for_ablation()`
   For the full pipeline and `exp2`/`exp3`, creates `FTSEncoder`, fits it on the fold-training raw windows, transforms training and validation/test windows, and returns the fitted encoder.
   For `exp4`, bypasses FTS and emits statistical features instead.

6. `src/fbn_pipeline/training.py::fit_signal_models()`
   Trains one `ANFIS_TSK` per signal on the fold-training features and labels.
   Produces per-signal train probabilities and validation/test probabilities.

7. `src/fbn_pipeline/training.py::calibrate_probs()`
   Optionally calibrates per-signal train probabilities and validation/test probabilities using Platt or isotonic calibration fitted on the fold-training probabilities and labels only.

8. `src/fbn_pipeline/fusion.py`
   - `compute_signal_weights()` derives optional reliability weights from fold-training labels plus fold-training evidence.
   - `make_bn()` constructs the configured BN fusion model.
   - `fuse_without_bn()` averages or weighted-averages per-signal probabilities when BN is disabled.

9. `src/fbn_pipeline/models/bn_fusion.py::BNFusion.fit()`
   Fits quantile bin edges, class prior, and CPTs from fold-training evidence and fold-training labels only.

10. `src/fbn_pipeline/metrics.py::find_best_threshold()`
    After CV finishes, sweeps thresholds on the full legacy training OOF probabilities only.

11. `src/fbn_pipeline/pipeline.py::retrain_and_test_result()`
    Re-fits the selected legacy pipeline on the full fixed training partition and predicts the fixed held-out test partition once.

12. `src/fbn_pipeline/runner.py`
    Computes final test metrics, optional per-subject metrics, optional artifact export, and optional CSV logging.

## Data-dependent operations and where they are fitted

### FTS / preprocessing

- Missing-value imputation medians:
  Fitted inside `FTSEncoder.fit()` on the current training subset only.

- Training min/max and universe-of-discourse bounds:
  Fitted inside `FTSEncoder.fit()` on the current training subset only.

- Fuzzy centers and widths:
  Fitted inside `FTSEncoder.fit()` on the current training subset only.
  - triangular MFs: quantile-derived triangle parameters;
  - gaussian/gbell/trapezoid MFs: grid or robust quantile centers plus spacing-derived widths.

- FTS normalization:
  Applied in `FTSEncoder.transform()` only; no fitted state beyond the learned MF parameters.

### ANFIS

- KMeans initialization:
  Fitted inside `ANFIS_TSK.fit()` on the current training feature subset only.

- Premise parameters (`centers_`, `sigmas_`):
  Fitted inside `ANFIS_TSK.fit()` on the current training feature subset only.

- Consequent parameters (`theta_`):
  Fitted inside `ANFIS_TSK._hybrid_train()` on the current training feature subset only.

### Calibration

- Platt or isotonic calibration:
  Fitted in `calibrate_probs()` on the current training probabilities and training labels only.

### Bayesian fusion

- BN quantile boundaries (`bin_edges_`):
  Fitted inside `BNFusion.fit()` on the current training evidence only.

- BN prior (`prior_`):
  Fitted inside `BNFusion.fit()` on the current training labels only.

- BN CPTs (`cpts_`):
  Fitted inside `BNFusion.fit()` on the current training evidence plus training labels only.

- Reliability weights:
  Fitted by `compute_signal_weights()` from current training labels plus current training evidence only.

### Thresholds and metrics

- Decision-threshold selection:
  Fitted in `find_best_threshold()` from the completed OOF probabilities over the fixed legacy training partition only.

## Leakage findings

- The legacy pipeline does **not** implement outer LOSO evaluation.
  It uses a fixed subject-independent train/test split from `src/fbn_pipeline/data.py`.

- Validation data does **not** affect fitted FTS, ANFIS, calibration, BN bin edges, BN prior, BN CPTs, or reliability weights inside each `GroupKFold` fold.

- The fixed held-out test subjects do **not** affect CV fitting, threshold selection, or final retraining.

- Validation labels **do** affect threshold selection indirectly because threshold search runs on the completed OOF predictions over the fixed legacy training partition.
  This is valid for the current legacy protocol but is not nested LOSO.

- BN currently receives **in-sample ANFIS training evidence** for fitting:
  - inside CV folds: `tr_probs` from ANFIS models fitted on the fold-training subset;
  - in final retraining: `tr_probs` from ANFIS models fitted on the full fixed training partition.

- BN currently receives **out-of-sample ANFIS validation/test evidence** only for inference:
  - `va_probs` during CV;
  - `te_probs` during final held-out test evaluation.

- No fitted FTS, ANFIS, calibration object, or BN object is intentionally reused across unrelated CV folds.
  Each fold creates fresh model instances.

- The selected threshold is reused exactly once after CV:
  it is frozen from legacy training OOF predictions, then applied to final held-out test posteriors.

## Legacy protocol summary

- Fitted inside current CV training fold:
  FTS parameters, ANFIS parameters, calibration parameters, BN bin edges, BN prior, BN CPTs, reliability weights.

- Fitted on the full fixed training partition after CV:
  FTS parameters, ANFIS parameters, calibration parameters, BN bin edges, BN prior, BN CPTs, reliability weights.

- Influenced by validation data:
  OOF metrics and threshold selection only.

- Influenced by held-out test data:
  final test metrics and reporting only.

- Influenced by held-out test labels during model or threshold selection:
  no.
