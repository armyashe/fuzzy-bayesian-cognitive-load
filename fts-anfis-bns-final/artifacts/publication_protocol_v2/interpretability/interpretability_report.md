# 4.5 Interpretability Analysis

This interpretability package separates global analyses computed on the 632-sample development set from the single illustrative local explanation computed on the hold-out set. Parameter-based analyses use the final locked model retrained on the development set only. No interpretability output was used for retraining, threshold tuning, or prediction changes.

## 4.5.1 Global FTS Transition Analysis

Figure: `global_fts_transition_difference.png` / `global_fts_transition_difference.pdf`

Dataset scope: development set (632 windows), class-conditional mean fuzzy transition matrices from the final locked FTS encoder.

Caption: Class-conditional differences in mean fuzzy transition matrices for the four physiological signals.

Findings:
- `RR` `Low -> Low`: -0.0274 (REST > TASK).
- `RR` `Low -> Medium`: +0.0233 (TASK > REST).
- `HR` `High -> Medium`: -0.0179 (REST > TASK).
- `HR` `Medium -> Medium`: -0.0148 (REST > TASK).

## 4.5.2 Global ANFIS Rule Analysis

Figure: `global_anfis_rule_activation_difference.png` / `global_anfis_rule_activation_difference.pdf`

Dataset scope: development set (632 windows), normalized Layer-3 firing strengths from the final retrained ANFIS models.

Caption: Class-conditional differences in the mean normalized firing strengths of the signal-specific ANFIS rules.

Figure: `global_anfis_consequent_importance.png` / `global_anfis_consequent_importance.pdf`

Dataset scope: final retrained ANFIS parameters learned on the 632-sample development set.

Caption: Magnitudes of the ANFIS consequent coefficients associated with the nine fuzzy-transition features of each physiological signal.

These coefficient magnitudes indicate how strongly the ANFIS consequent functions use each transition feature. They do not represent causal effects or direct contributions to the final Bayesian posterior.

Findings:
- `RR` `Rule 3` mean firing shift: -0.0822 (REST > TASK).
- `RR` `Rule 2` mean firing shift: +0.0779 (TASK > REST).
- `GSR` largest consequent-magnitude feature: `L->M` (10.1943).
- `HR` largest consequent-magnitude feature: `M->H` (5.7859).

## 4.5.3 Global Bayesian Evidence Analysis

Figure: `global_bn_log_likelihood_ratio.png` / `global_bn_log_likelihood_ratio.pdf`

Dataset scope: CPTs and priors from the final fitted soft-evidence Naive Bayes model retrained on the 632-sample development set.

Caption: Log-likelihood ratios of the Bayesian evidence bins for the REST and TASK classes.

Findings:
- Prior probabilities from the fitted BN: REST=0.5016, TASK=0.4984.
- `RR` `Very Low` LLR=-1.3366 (supports REST).
- `RR` `Very High` LLR=+1.0433 (supports TASK).
- `HR` `Very High` LLR=+0.7053 (supports TASK).

## 4.5.4 Sample-Level Explanation

Figure: `sample_level_bn_contribution_waterfall.png` / `sample_level_bn_contribution_waterfall.pdf`

Dataset scope: one correctly classified hold-out sample chosen deterministically from the fixed hold-out set. Selection rule: among correctly classified hold-out samples, choose the sample with posterior closest to 0.5; break ties by larger contribution spread entropy, then participant ID, then sample index.

Caption: Sample-level Bayesian evidence contributions for an illustrative correctly classified hold-out sample.

Findings:
- Selected hold-out sample: participant `3caqi`, global index `20`, posterior TASK `0.5007` at threshold `0.55`.
- `GSR` contribution=+0.0882 with dominant bin `High` (supports TASK).
- `RR` contribution=-0.0464 with dominant bin `Low` (supports REST).
- `Temp` contribution=-0.0190 with dominant bin `Low` (supports REST).

### Local Inference Flow

Participant: `3caqi`
Global hold-out sample index: `20`
True label: `REST`
Predicted label: `REST`
Posterior TASK: `0.5007`
Threshold: `0.55`

Raw window -> dominant FTS transitions -> dominant ANFIS rules -> signal-level ANFIS scores -> soft BN evidence -> posterior -> final prediction

#### Temp
- Raw first 5 values: `[34.89, 34.89, 34.89, 34.8967, 34.9033]`
- Dominant FTS transitions: `High->High=0.8952`, `Medium->High=0.8952`
- Dominant ANFIS rules: `Rule 1=1.0000`, `Rule 6=0.0000`
- Signal-level ANFIS score: `0.6184`
- Soft BN evidence dominant bin (illustrative only): `Low`
- TASK vs REST contribution: `-0.0190`

#### HR
- Raw first 5 values: `[60.0, 59.6667, 59.3333, 58.6667, 58.3333]`
- Dominant FTS transitions: `Low->Low=0.6889`, `Medium->Low=0.6734`
- Dominant ANFIS rules: `Rule 3=1.0000`, `Rule 2=0.0000`
- Signal-level ANFIS score: `0.6225`
- Soft BN evidence dominant bin (illustrative only): `Med`
- TASK vs REST contribution: `-0.0135`

#### GSR
- Raw first 5 values: `[2.1978, 2.1978, 2.1962, 2.1946, 2.193]`
- Dominant FTS transitions: `High->Medium=0.6765`, `Medium->Medium=0.6757`
- Dominant ANFIS rules: `Rule 4=1.0000`, `Rule 6=0.0000`
- Signal-level ANFIS score: `0.6332`
- Soft BN evidence dominant bin (illustrative only): `High`
- TASK vs REST contribution: `+0.0882`

#### RR
- Raw first 5 values: `[1.0564, 1.1061, 1.0729, 1.0674, 0.9955]`
- Dominant FTS transitions: `Low->Medium=0.8732`, `Medium->Medium=0.8621`
- Dominant ANFIS rules: `Rule 4=0.9594`, `Rule 6=0.0403`
- Signal-level ANFIS score: `0.6086`
- Soft BN evidence dominant bin (illustrative only): `Low`
- TASK vs REST contribution: `-0.0464`

Final posterior TASK: `0.5007`
Final prediction: `REST`


## Validation

- `final_model_matches_joint_lock`: `PASS`
- `fts_transition_orientation_is_state_t_to_state_t_plus_1`: `PASS`
- `all_anfis_firing_vectors_sum_to_one`: `PASS`
- `all_signals_use_six_rules`: `PASS`
- `consequent_heatmap_shape_is_4_by_9`: `PASS`
- `bn_llr_heatmap_shape_is_4_by_5`: `PASS`
- `all_cpt_rows_sum_to_one`: `PASS`
- `sample_explanation_matches_predict_proba`: `PASS`
- `soft_explanation_uses_soft_evidence_not_hard_bins`: `PASS`
- `seed_42_rerun_reproduces_same_numeric_artifacts`: `PASS`
- `interpretability_does_not_change_model_predictions`: `PASS`