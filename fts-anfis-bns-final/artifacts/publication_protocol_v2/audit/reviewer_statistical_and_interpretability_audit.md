# Reviewer Statistical And Interpretability Audit

## Scope

This audit reviews the current `publication_protocol_v2` flow against two reviewer-facing requirements:

1. statistical uncertainty for the final locked hold-out metrics;
2. stability of interpretability results across folds or resampling.

This document is an audit only. It does not change model selection, thresholds, final predictions, or published artifacts.

## A. Current Status

### A1. Statistical uncertainty

#### What already exists

- Final hold-out point estimates already exist in `artifacts/publication_protocol_v2/final_evaluation/final_main_model_metrics.csv`.
- The proposed model currently reports these point estimates on the locked 5-participant hold-out set:
  - Accuracy: `0.7513`
  - Balanced Accuracy: `0.7519`
  - TASK F1: `0.7447`
  - ROC-AUC: `0.7734`
- Additional point estimates also exist for Precision, Recall, and Macro-F1.
- Participant-level metrics already exist in:
  - `artifacts/publication_protocol_v2/final_evaluation/participant_level_test_results.csv`
  - `artifacts/publication_protocol_v2/final_evaluation/participant_level_test_report.md`
- The current participant-wise unweighted mean +/- SD across the 5 hold-out participants is:
  - Accuracy: `0.7511 +/- 0.0647`
  - Balanced Accuracy: `0.7521 +/- 0.0641`
  - TASK F1: `0.7423 +/- 0.0724`
  - ROC-AUC: `0.7785 +/- 0.0408`
- 95% confidence intervals already exist in `artifacts/publication_protocol_v2/final_evaluation/final_confidence_intervals.csv`.
- A paired McNemar result already exists in `artifacts/publication_protocol_v2/final_evaluation/final_mcnemar_result.json`.

#### What the current implementation does

- Bootstrap implementation exists in `src/fbn_pipeline/final_evaluation.py` via `_clustered_bootstrap_rows(...)`.
- The current bootstrap resamples at the participant level, not the sample level:
  - `subject_ids = np.unique(entry["subjects"])`
  - each replicate samples subjects with replacement;
  - all windows from each sampled subject are included;
  - predictions and scores are reused as locked arrays;
  - there is no retraining and no threshold re-tuning inside the bootstrap loop.
- This is already a subject-clustered bootstrap in substance.
- The current statistical plan in `artifacts/publication_protocol_v2/comparison/statistical_analysis_plan.json` specifies:
  - `resampling_unit = "subject"`
  - `replicates = 10000`
  - `seed = 4242`
  - `level = 0.95`
  - undefined AUC handling: skip undefined replicates and report effective replicate count.
- ROC-AUC undefined replicates are skipped, not imputed as `0`, which is the correct direction.
- Provenance already exists through:
  - `statistical_analysis_plan.json`
  - `final_evaluation_manifest.json`
  - lock hashes
  - prediction hashes
  - recorded bootstrap seed and replicate count in `final_confidence_intervals.csv`.
- Guardrails that prevent hold-out leakage are already explicit:
  - `threshold_source = "training_oof_only"` in `locked_config.json`
  - `selected_using = "training_oof_only"` in `locked_config.json`
  - final evaluation manifest says `no_tuning_occurred = true`
  - bootstrap operates on locked predictions only.

#### What is already sufficient

- Confidence intervals for Accuracy: present.
- Confidence intervals for Balanced Accuracy: present.
- Confidence intervals for TASK F1: present.
- Confidence intervals for ROC-AUC: present.
- Bootstrap implementation: present.
- Participant-level uncertainty summary: present as per-participant rows and mean +/- SD.
- McNemar test: present.
- Bootstrap seed and provenance: present.
- Explicit separation between model/threshold selection and final hold-out evaluation: present.

#### What is still incomplete or fragile

- The current audit trail is good technically, but there is no dedicated reviewer-facing narrative that ties all uncertainty artifacts together in one place.
- The current bootstrap protocol does not match the requested minimal spec exactly:
  - current code uses `10000` replicates, not `2000`;
  - current code uses `seed = 4242`, not `42`.
- The current participant-level report gives mean +/- SD, but not participant-clustered CIs for those participant-level summaries.
- With only 5 hold-out participants, any clustered bootstrap CI is statistically fragile at the cluster level:
  - the resampling unit is correct;
  - the number of independent clusters is still only 5;
  - interval endpoints will be sensitive to any one participant.

#### Assessment of the 5-participant hold-out

- With only 5 hold-out participants, sample-wise bootstrap would be methodologically weaker because it would ignore within-participant dependence.
- Subject-clustered bootstrap is the correct choice despite the small number of clusters.
- The main limitation is not the bootstrap code; it is the extremely small number of resampling units.
- Publication text should say that the CIs are participant-clustered and respect window dependence, but remain unstable because only 5 unseen participants were available.

### A2. Interpretability stability

#### What currently exists

- Interpretability outputs are generated by `src/fbn_pipeline/interpretability.py` and `run_publication_interpretability.py`.
- Current artifacts include:
  - `global_fts_transition_difference.csv/.png/.pdf`
  - `global_anfis_rule_activation.csv`
  - `global_anfis_rule_activation_difference.png/.pdf`
  - `global_anfis_consequent_importance.csv/.png/.pdf`
  - `global_bn_evidence_llr.csv`
  - `global_bn_log_likelihood_ratio.png/.pdf`
  - `sample_level_explanation.csv`
  - `sample_level_bn_contribution_waterfall.png/.pdf`
  - `interpretability_report.md`

#### What these figures are actually based on

- FTS transition difference heatmaps:
  - computed from one final locked FTS encoder retrained on the full 632-window development set;
  - not aggregated across folds.
- ANFIS rule activation differences:
  - computed from one final retrained ANFIS model per signal on the full development set;
  - not aggregated across folds.
- ANFIS consequent coefficient magnitudes:
  - computed from one final retrained ANFIS parameter set;
  - not aggregated across folds.
- BN log-likelihood ratios:
  - computed from one final fitted BN CPT set and priors;
  - not aggregated across folds.
- Sample-level explanation:
  - one deterministic illustrative hold-out sample;
  - explicitly a snapshot, not a stability analysis.

#### Current interpretation status

- All current global interpretability outputs are final-model snapshots.
- None of the current interpretability artifacts quantify fold-wise stability, resampling stability, sign consistency, or selection frequency.
- There is no current artifact that justifies a claim such as "interpretability patterns were stable across folds."

#### Rule permutation problem

- The ANFIS rule permutation problem is real in the current implementation.
- `src/fbn_pipeline/models/anfis_tsk.py` initializes ANFIS premises using KMeans cluster centers and fold-dependent random states.
- There is no canonical rule ordering or alignment layer after fitting.
- Therefore direct comparisons like "Rule 1 in fold A versus Rule 1 in fold B" are not reliable.
- Any rule-level cross-fold stability claim without explicit alignment would be methodologically weak.

#### What is already sufficient

- The current report is careful about using the final locked model and not using interpretability for tuning.
- The current report already labels coefficient magnitudes as usage indicators rather than causal effects.

#### What is missing

- Fold-wise or resampled stability for FTS transition differences.
- Fold-wise or resampled stability for ANFIS rule activation differences.
- Fold-wise rank stability for ANFIS consequent-feature usage.
- Fold-wise stability for BN evidence LLRs.
- Any explicit limitation around rule alignment, beyond what is implied by the code.

## B. Recommended Implementation

### B1. Statistical uncertainty

#### Recommendation

- Keep subject-clustered bootstrap as the main uncertainty method for final hold-out metrics.
- Report CIs for:
  - Accuracy
  - Balanced Accuracy
  - TASK F1
  - ROC-AUC
- McNemar can be kept as a paired comparison against the best locked classical baseline.
- Do not use bootstrap for model selection or threshold selection.

#### Assessment of current implementation versus recommendation

- The current implementation already satisfies the core statistical requirement.
- The main open choice is whether to change the bootstrap protocol from:
  - `10000` replicates, `seed = 4242`
  - to
  - `>=2000` replicates, `seed = 42`.

#### Minimal recommendation

- Do not downgrade from `10000` to `2000`; `10000` is already stronger than the minimum.
- If exact consistency with a reviewer response or manuscript wording matters, change only the bootstrap seed to `42` and document the change.
- If exact consistency does not matter, the current clustered bootstrap can remain unchanged and should simply be surfaced better in the report/manuscript.

#### Effort and impact

- Documentation-only surfacing of current CI pipeline:
  - effort: low
  - impact: high
- Changing seed/protocol wording only:
  - effort: low
  - impact: low to moderate
- Re-running bootstrap with revised seed while preserving the same locked predictions:
  - effort: low
  - impact: moderate on reproducibility wording, low on substance

### B2. Interpretability stability

#### Recommendation

- The most defensible minimal stability analyses are:
  1. fold-wise ANFIS consequent-feature ranking stability;
  2. fold-wise BN evidence LLR stability.
- FTS transition stability is feasible and useful, but not strictly necessary for the minimal reviewer response if scope must stay narrow.
- Rule-level ANFIS activation stability should not be presented as stable unless rule alignment is implemented and validated.

#### Why ANFIS consequent-feature ranking is a good minimal option

- The 9 FTS transition features are fixed and semantically aligned across folds.
- No rule matching is needed if the statistic is:
  - per-signal sum of absolute consequent coefficients across rules;
  - optional within-signal normalization;
  - fold-wise Spearman rank correlation;
  - top-feature selection frequency.
- This directly addresses stability of model usage patterns without overclaiming causal importance.

#### Why BN LLR stability is a good minimal option

- BN evidence tables already have a fixed 4-signal x 5-bin structure.
- Fold-wise CPT refits can produce:
  - mean LLR
  - SD
  - sign consistency
  - rank consistency
- This is publication-ready and cheap to explain.

#### Why rule-level ANFIS stability is not a minimal option

- It requires explicit rule alignment.
- Alignment by premise-center distance plus Hungarian assignment is plausible, but not free.
- If alignment quality is weak, the analysis becomes misleading rather than helpful.

#### Effort and impact

- ANFIS consequent-feature rank stability:
  - effort: low to moderate
  - impact: high
- BN LLR stability:
  - effort: low to moderate
  - impact: high
- FTS transition stability:
  - effort: moderate
  - impact: moderate
- Rule-aligned ANFIS activation stability:
  - effort: moderate to high
  - impact: moderate if alignment is good, negative if alignment is weak

## C. Minimal Publication-Ready Option

### Recommended minimal package

1. Keep the current subject-clustered bootstrap CI analysis for final locked hold-out metrics, and make it explicit in the manuscript/report.
2. Add fold-wise ANFIS consequent-feature ranking stability for the selected locked configuration on the 18-subject development set.
3. Add fold-wise BN LLR stability for the selected locked configuration on the same 5 GroupKFold development splits.
4. Add a clear limitation stating that rule-level ANFIS stability is not claimed because rule identities are not reliably aligned across folds.

### Why this is the best minimal option

- It addresses reviewer concern about uncertainty with almost no methodological change because the bootstrap CI pipeline already exists.
- It addresses reviewer concern about interpretability stability using statistics that are actually comparable across folds.
- It avoids the weakest part of the space: naive rule-by-rule comparisons.

### Proposed outputs for the minimal option

- Statistical uncertainty:
  - keep `final_confidence_intervals.csv`
  - optionally add a short reviewer-facing markdown summary
- Interpretability stability:
  - `interpretability_stability_anfis_consequent.csv`
  - `interpretability_stability_bn_llr.csv`
  - `interpretability_stability_report.md`

### Suggested reporting language

- "Final hold-out uncertainty was quantified using participant-clustered bootstrap confidence intervals computed from fixed locked predictions without retraining or threshold re-selection."
- "Interpretability stability was assessed across the 5 development GroupKFold splits for ANFIS consequent-feature ranking and Bayesian evidence log-likelihood ratios."
- "Rule-level ANFIS stability was not claimed because rule labels are not intrinsically aligned across folds."

## D. Full Option

### Full stability package

- Subject-clustered bootstrap CI for final hold-out metrics.
- Fold-wise FTS transition stability:
  - mean
  - SD
  - sign consistency
  - top-transition selection frequency
- Fold-wise ANFIS consequent-feature stability:
  - coefficient magnitude sums
  - optional within-signal normalization
  - pairwise or final-vs-fold Spearman rank correlation
  - top-feature selection frequency
- Fold-wise BN LLR stability:
  - mean
  - SD
  - sign consistency
  - rank consistency
- Rule-aligned ANFIS activation stability:
  - premise-center distance matrix
  - Hungarian rule matching
  - matched rule activation mean/SD/sign consistency
  - explicit diagnostics for alignment confidence

### Supplementary material suitable for the full option

- supplementary stability tables for all 4 signals
- supplementary heatmaps of fold-wise mean +/- SD
- supplementary selection-frequency tables
- one short limitations subsection on:
  - only 5 hold-out participants for CI estimation
  - fold-specific BN bins are relative quantile bins
  - rule alignment uncertainty

## Explicit Answers To The Audit Questions

### 1. Files read

- `src/fbn_pipeline/final_evaluation.py`
- `src/fbn_pipeline/interpretability.py`
- `src/fbn_pipeline/metrics.py`
- `src/fbn_pipeline/publication_selection.py`
- `src/fbn_pipeline/runner.py`
- `src/fbn_pipeline/training.py`
- `src/fbn_pipeline/pipeline.py`
- `src/fbn_pipeline/results.py`
- `src/fbn_pipeline/models/anfis_tsk.py`
- `src/fbn_pipeline/data.py`
- `run_publication_interpretability.py`
- `tests/test_final_evaluation_summary.py`
- `tests/test_interpretability_pipeline.py`

### 2. Existing artifacts

- `artifacts/publication_protocol_v2/locked_config.json`
- `artifacts/publication_protocol_v2/development/joint_fts_anfis_selection_summary.json`
- `artifacts/publication_protocol_v2/development/joint_fts_anfis_selection_oof.csv`
- `artifacts/publication_protocol_v2/development/final_selected_oof_predictions.npz`
- `artifacts/publication_protocol_v2/comparison/statistical_analysis_plan.json`
- `artifacts/publication_protocol_v2/final_evaluation/final_main_model_metrics.csv`
- `artifacts/publication_protocol_v2/final_evaluation/final_confidence_intervals.csv`
- `artifacts/publication_protocol_v2/final_evaluation/final_mcnemar_result.json`
- `artifacts/publication_protocol_v2/final_evaluation/participant_level_test_results.csv`
- `artifacts/publication_protocol_v2/final_evaluation/participant_level_test_report.md`
- `artifacts/publication_protocol_v2/final_evaluation/final_evaluation_manifest.json`
- `artifacts/publication_protocol_v2/interpretability/interpretability_report.md`
- `artifacts/publication_protocol_v2/interpretability/global_fts_transition_difference.csv`
- `artifacts/publication_protocol_v2/interpretability/global_anfis_rule_activation.csv`
- `artifacts/publication_protocol_v2/interpretability/global_anfis_consequent_importance.csv`
- `artifacts/publication_protocol_v2/interpretability/global_bn_evidence_llr.csv`
- `artifacts/publication_protocol_v2/interpretability/sample_level_explanation.csv`

### 3. Confidence intervals already present?

- Yes.
- They are already present for Accuracy, Balanced Accuracy, TASK F1, and ROC-AUC.
- They are subject-clustered bootstrap percentile CIs.

### 4. Bootstrap already present?

- Yes.
- It resamples participants with replacement and reuses locked predictions/probabilities.
- It does not retrain and does not re-tune thresholds.

### 5. Are current interpretability plots snapshots or stability analyses?

- Snapshots.
- They are based on the final retrained locked model, not aggregated across folds or resampling.

### 6. Is rule permutation a real problem?

- Yes.
- Direct cross-fold comparison of `Rule k` is not reliable in the current implementation.

### 7. Recommended minimal option

- Keep the current subject-clustered bootstrap CI framework.
- Add fold-wise ANFIS consequent-feature ranking stability.
- Add fold-wise BN LLR stability.
- Explicitly state that rule-level ANFIS stability is not claimed without alignment.

### 8. Files that would likely need edits for implementation

- `src/fbn_pipeline/interpretability.py`
- `run_publication_interpretability.py`
- likely one new test file or an extension of:
  - `tests/test_interpretability_pipeline.py`
- possibly `src/fbn_pipeline/publication_selection.py` only if fold metadata export needs to be surfaced more directly
- possibly `src/fbn_pipeline/final_evaluation.py` only if the bootstrap seed/protocol is standardized to `seed = 42`

### 9. Expected commands to run

- Current final evaluation bundle:
  - `python -c "from pathlib import Path; import sys; sys.path.insert(0, r'fts-anfis-bns-final\\src'); from fbn_pipeline.final_evaluation import run_publication_protocol_v2_final_evaluation; print(run_publication_protocol_v2_final_evaluation(Path(r'fts-anfis-bns-final'))['created_files'])"`
- Current interpretability bundle:
  - `python .\\fts-anfis-bns-final\\run_publication_interpretability.py`
- If minimal stability is implemented, the most likely run path is:
  - extend `run_publication_interpretability.py` to emit both snapshot and fold-stability artifacts in one pass

### 10. Remaining methodological risks

- Only 5 hold-out participants means clustered bootstrap CIs remain unstable at the participant level.
- Bootstrap CIs are valid in design but limited by the number of clusters, not by the number of windows.
- BN LLR bins are fold-local quantile bins; cross-fold comparisons are interpretable as relative evidence-bin patterns, not fixed absolute score thresholds.
- ANFIS coefficient magnitude is a model-usage indicator, not causal importance.
- Rule-level ANFIS stability should not be claimed unless alignment is implemented and shown to be reliable.

## Bottom Line

- Statistical uncertainty is mostly already implemented and publication-usable now.
- Interpretability stability is not implemented yet; the current package is final-model snapshot only.
- The best minimal reviewer-facing upgrade is:
  - keep subject-clustered bootstrap CI for final metrics;
  - add fold-wise ANFIS consequent-feature ranking stability;
  - add fold-wise BN LLR stability;
  - state a limitation for rule-level ANFIS stability instead of overclaiming it.
