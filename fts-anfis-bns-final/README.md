# FTS-ANFIS-BN Cognitive Load Pipeline

This repository contains two evaluation paths:

- `Legacy 18/5 pipeline`: preserved for regression protection and contextual comparison only.
- `Phase 3 publication pipeline`: primary official holdout workflow plus optional secondary nested LOSO robustness analysis.

The legacy path and its frozen fixture are intentionally unchanged. New publication results should come from the official holdout workflow by default.

## Dataset Layout

Place the CogLoad window files under:

```text
last_30s_segments/
  gsr.xlsx
  hr.xlsx
  labels.xlsx
  rr.xlsx
  temp.xlsx
```

The Phase 3 loaders expect:

- `825` total windows
- `23` subjects
- `4` signals in channel order: `Temp`, `HR`, `GSR`, `RR`
- window shape `(N, 4, 30)`

## Environment

Use Python `3.12` and the repository venv:

```powershell
python -m venv .venv312_new
.\.venv312_new\Scripts\python.exe -m pip install --upgrade pip
.\.venv312_new\Scripts\python.exe -m pip install -r requirements.txt
```

Key pinned versions used by the publication workflow:

- `Python 3.12.10`
- `numpy 2.4.4`
- `pandas 3.0.2`
- `scikit-learn 1.8.0`
- `scipy 1.17.1`
- `openpyxl 3.1.5`
- `xgboost 3.2.0`
- `torch 2.12.1`
- `pytest 9.1.1`

## Determinism

The nested runner enforces deterministic seeds and single-threaded execution where required. Publication metadata records:

- run name
- mode (`publication` or `smoke`)
- seed
- dataset-order hash
- configuration hash
- timestamp
- Python version
- package versions

## Evaluation Protocol

Primary publication evaluation:

- fixed official subject-independent split
- `18` official training subjects / `632` training windows
- `5` official test subjects / `193` test windows
- `5`-fold `GroupKFold` only inside the `18` official training subjects
- candidate selection by training OOF `balanced_accuracy`
- threshold selection by training OOF `balanced_accuracy`
- final retraining on all `18` official training subjects
- final pooled metrics from the `193` official test predictions only

Secondary robustness evaluation:

- optional `nested_loso_secondary`
- all `23` subjects
- reported separately and never mixed with official holdout results

Strict publication coverage is the default for the primary protocol:

- every official test sample must receive exactly one final prediction per method
- final publication outputs must contain exactly `193` official test prediction rows per method
- no official training sample may appear in final test predictions
- missing predictions or split inconsistencies raise errors

Partial coverage is allowed only in explicit smoke mode.

## Phase 3 Publication Root

Publication and smoke runs use a stable run root:

```text
artifacts/phase3_publication/<run_name>/
  metadata/
  predictions/
  metrics/
  statistics/
  interpretability/
  tables/
  figures/
```

Rules:

- `run_name` must be explicit, for example `nested_full_v1`, `paper_revision_2026`, or `phase3_smoke`
- existing results are not overwritten unless `--overwrite` is passed
- timestamp-only directories are not used as the primary path
- tables and figures are regenerated from the same run root without retraining

## Methods

Controlled methods in the nested workflow:

- `proposed_fts_anfis_bn`
- `fts_logistic_regression`
- `fts_rbf_svm`
- `fts_xgboost`
- `raw_1d_cnn`

Ablations:

- `fts_anfis_mean`
- `fts_summary_bn`

Interpretation:

- `FTS–ANFIS–Mean` isolates the contribution of Bayesian fusion
- `FTS Summary–BN` is a heuristic no-ANFIS ablation, not an external state-of-the-art baseline

The current `FTS Summary–BN` summary score uses the fixed FTS transition terms:

- `medium_to_high`
- `high_to_high`

per signal, summed from the `3x3` FTS transition block.

## Candidate Grids

The publication path keeps the predefined candidate grids from Phase 2:

- Proposed:
  - membership functions: `triangular`, `gaussian`, `trapezoid`, `gbell`
  - ANFIS rule counts: `3`, `6`, `8`, `12`
- Logistic Regression:
  - membership functions: `triangular`, `gaussian`, `trapezoid`, `gbell`
  - `C`: `0.01`, `0.1`, `1.0`, `10.0`
- RBF-SVM:
  - membership functions: `triangular`, `gaussian`, `trapezoid`, `gbell`
  - `C`: `0.1`, `1.0`, `10.0`
  - `gamma`: `scale`, `0.01`, `0.1`
- XGBoost:
  - membership functions: `triangular`, `gaussian`, `trapezoid`, `gbell`
  - estimators: `100`, `300`
  - depth: `3`, `5`
  - learning rate: `0.03`, `0.1`
- Raw CNN:
  - learning rate: `3e-4`, `1e-3`
  - dropout: `0.0`, `0.2`

## Cross-Fitted BN Evidence

The proposed method fits the BN only from cross-fitted ANFIS evidence generated inside the official training subjects. The BN never uses official test evidence or labels for binning, priors, CPT estimation, smoothing, calibration, or threshold selection.

## Output Artifacts

The Phase 3 runner writes machine-readable artifacts such as:

- `metadata/fold_manifest.json`
- `metadata/outer_fold_configs.csv`
- `metadata/runtime_metadata.json`
- `predictions/outer_fold_predictions.csv`
- `metrics/candidate_scores.csv`
- `metrics/candidate_failures.csv`
- `metrics/overall_metrics.csv`
- `metrics/per_subject_metrics.csv`
- `metrics/ablation_results.csv`
- `statistics/confidence_intervals.csv`
- `statistics/statistical_tests.csv`
- `statistics/selection_frequencies.csv`
- `interpretability/cross_fitted_anfis_evidence.csv`
- `interpretability/anfis_interpretability.csv`
- `interpretability/bn_explanations.csv`
- `tables/*.csv`
- `figures/*.png`

## Entry Points

Run the proposed method with the default official holdout protocol:

```powershell
.\.venv312_new\Scripts\python.exe scripts\run_nested_proposed.py `
  --run-name official_proposed_v1 `
  --mode publication
```

Run the full publication workflow into one shared run root:

```powershell
.\.venv312_new\Scripts\python.exe scripts\run_nested_publication.py `
  --run-name paper_revision_2026 `
  --mode publication `
  --include-xgboost `
  --include-cnn `
  --include-ablations
```

Run the controlled baselines on the same official split:

```powershell
.\.venv312_new\Scripts\python.exe scripts\run_nested_baselines.py `
  --run-name official_baselines_v1 `
  --mode publication `
  --include-xgboost `
  --include-cnn
```

Run the proposed model plus the Phase 3 ablations on the official split:

```powershell
.\.venv312_new\Scripts\python.exe scripts\run_nested_ablations.py `
  --run-name official_ablations_v1 `
  --mode publication
```

Run the optional secondary nested LOSO workflow explicitly:

```powershell
.\.venv312_new\Scripts\python.exe scripts\run_nested_publication.py `
  --run-name nested_loso_secondary_v1 `
  --mode publication `
  --protocol nested_loso_secondary `
  --include-xgboost `
  --include-cnn `
  --include-ablations
```

Regenerate statistics, tables, and figures from saved artifacts only:

```powershell
.\.venv312_new\Scripts\python.exe scripts\generate_publication_outputs.py `
  --run-name nested_full_v1
```

## Smoke Mode

Smoke mode is explicit and should use a clearly different run name, for example `phase3_smoke`.

Example:

```powershell
.\.venv312_new\Scripts\python.exe scripts\run_nested_ablations.py `
  --run-name phase3_smoke `
  --mode smoke `
  --train-subject-limit 4 `
  --test-subject-limit 2 `
  --inner-folds 2 `
  --overwrite
```

Smoke mode:

- allows partial outer coverage explicitly
- records `mode=smoke` and partial coverage metadata
- is not publication output

## Legacy Path

Legacy commands such as `main.py` and `run_experiments.py` remain available for contextual use and regression protection, but they are not the primary publication evaluation path. The optional `nested_loso_secondary` workflow is also not the primary publication path.

The frozen fixture at `tests/fixtures/legacy_reference.json` must not be regenerated automatically.

## Testing

Baseline Phase 1/2 regression suite:

```powershell
.\.venv312_new\Scripts\python.exe -m pytest `
tests\test_legacy_regression.py `
tests\test_structured_models.py `
tests\test_fts_invariants.py `
tests\test_anfis_invariants.py `
tests\test_bn_invariants.py `
tests\test_metrics_invariants.py `
tests\test_nested_engine.py `
tests\test_phase2_manifest_and_policy.py `
tests\test_phase2_bn_and_adapters.py `
tests\test_phase2_completion.py -q
```

Focused Phase 3 tests:

```powershell
.\.venv312_new\Scripts\python.exe -m pytest `
tests\test_official_protocol.py `
tests\test_phase3_reporting.py `
tests\test_phase3_execution.py `
tests\test_phase3_reports.py -q
```

## Expected Runtime

Approximate runtime guidance:

- focused tests: seconds
- reduced smoke runs: minutes
- full publication run across all methods and ablations: materially longer and should be started manually

The repository is prepared for the full official holdout run, but the expensive full experiment is intentionally user-controlled.

## Known Limitations

- Statistical confidence intervals and paired tests are artifact-driven; they depend on complete saved prediction artifacts.
- External contextual comparison tables require user-supplied external result artifacts if they are to contain literature values.
- Some publication figures are schematic summaries driven by saved metadata rather than by a dedicated drawing dependency.
