# Final Evaluation Summary

## Validation
{
  "proposed_lock_valid": true,
  "baseline_locks_valid": true,
  "ablation_locks_valid": true,
  "comparison_manifest_valid": true,
  "statistical_plan_valid": true,
  "all_thresholds_from_training_oof": true,
  "all_selected_using_training_oof_only": true,
  "no_final_test_evaluation_recorded_in_locks": true,
  "proposed_reconstructed_from_explicit_locked_config": true,
  "hyperparameter_override_interface_present": false,
  "comparison_manifest_hash": "7538e7e8521030e35d02e56684d11b40afd456795ab9d62c15936269eab68a9f",
  "statistical_plan_hash": "5edc14b998cd87f9c5fbcd2f4dda5c5837754adf46350bca7eda376f240ed164",
  "participant_level_sum_n_samples_matches_expected": true,
  "participant_level_all_samples_map_to_exactly_one_participant": true,
  "participant_level_prediction_order_matches_user_test": true,
  "participant_level_locked_threshold_matches_expected": true,
  "participant_level_no_participant_tuning_or_retraining": true,
  "participant_level_aggregate_accuracy_matches_expected": true,
  "participant_level_seed_is_42": true,
  "participant_level_participant_counts_match_user_test": true,
  "participant_level_array_lengths_consistent": true
}

## Main Final Metrics
| Model | Accuracy | Balanced Accuracy | Precision | Recall | TASK F1 | Macro-F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Proposed FTS-ANFIS-BN | 0.7513 | 0.7519 | 0.7778 | 0.7143 | 0.7447 | 0.7511 | 0.7734 |
| Statistical Logistic Regression | 0.6373 | 0.6369 | 0.6373 | 0.6633 | 0.6500 | 0.6368 | 0.7126 |
| Statistical RBF SVM | 0.6269 | 0.6252 | 0.6102 | 0.7347 | 0.6667 | 0.6216 | 0.7126 |
| Statistical Random Forest | 0.6891 | 0.6890 | 0.6939 | 0.6939 | 0.6939 | 0.6890 | 0.7126 |
| Statistical HistGradientBoosting | 0.6062 | 0.6061 | 0.6122 | 0.6122 | 0.6122 | 0.6061 | 0.6455 |
| Flattened-window Logistic Regression | 0.5492 | 0.5442 | 0.5346 | 0.8673 | 0.6615 | 0.4935 | 0.5454 |

## Ablation Metrics
| Model | Accuracy | Balanced Accuracy | Precision | Recall | TASK F1 | Macro-F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full FTS-ANFIS-BN | 0.7513 | 0.7519 | 0.7778 | 0.7143 | 0.7447 | 0.7511 | 0.7734 |
| No BN | 0.6891 | 0.6873 | 0.6583 | 0.8061 | 0.7248 | 0.6838 | 0.7158 |
| No ANFIS | 0.5389 | 0.5385 | 0.5446 | 0.5612 | 0.5528 | 0.5384 | 0.5709 |
| No FTS | 0.6477 | 0.6490 | 0.6875 | 0.5612 | 0.6180 | 0.6455 | 0.7241 |

## Participant-Level Analysis
| Participant | N | Acc. | Bal. Acc. | TASK F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3caqi | 39.0000 | 0.6923 | 0.6961 | 0.6471 | 0.7842 |
| 6frz4 | 39.0000 | 0.7179 | 0.7184 | 0.7179 | 0.7237 |
| bd47a | 38.0000 | 0.7105 | 0.7105 | 0.7317 | 0.7452 |
| f1gjp | 39.0000 | 0.8718 | 0.8724 | 0.8718 | 0.8026 |
| iz3x1 | 38.0000 | 0.7632 | 0.7632 | 0.7429 | 0.8366 |

Table 8 reports participant-level performance on the five unseen hold-out participants. Accuracy ranged from 0.6923 to 0.8718 (unweighted mean 0.7511, SD 0.0647), indicating substantial between-participant variation in the final evaluation. These participant-level results are reported only as a post hoc stability analysis on unseen participants and were not used to select the model, tune the threshold, or adjust any hyperparameter.

## McNemar Result
{
  "comparison": "Proposed FTS-ANFIS-BN vs Statistical Logistic Regression",
  "proposed_model_id": "PUB_JOINT__MF_GAUSSIAN__R6",
  "baseline_model_id": "STAT_LOGREG__C0.1__CWnone",
  "proposed_correct_baseline_wrong": 35,
  "baseline_correct_proposed_wrong": 13,
  "discordant_total": 48,
  "mcnemar_statistic": 9.1875,
  "p_value": 0.002437,
  "method": "continuity-corrected",
  "proposed_lock_hash": "f790ddc101b1397b5b5cde35a258b69be265830e5e88b67339d3817e1cacf3f8",
  "baseline_lock_hash": "2797cf9d4683cec2b05e8328154a7a360ffe22526bff1c2772b5d53a945f949e",
  "proposed_prediction_hash": "614cdab65fa8a9a2b91a2f987adbdc826933d982d171eced57af07d1104cd8d9",
  "baseline_prediction_hash": "bc04000cff478debea093512b37f234a2b154ebe717482d55cd31ab977be966b"
}

## Integrity Checks
{
  "ablation_full_scientific_config_matches_proposed": false,
  "ablation_full_predictions_identical_to_proposed": false,
  "thresholds_applied_unchanged": true,
  "test_labels_used_only_for_metrics": true,
  "no_configuration_mutation_after_metrics": true,
  "no_candidate_removed_for_poor_test_performance": true,
  "all_result_rows_record_hashes": true,
  "no_hyperparameter_override_interface": true,
  "all_expected_models_evaluated": [
    "PUB_JOINT__MF_GAUSSIAN__R6",
    "STAT_LOGREG__C0.1__CWnone",
    "STAT_SVM__C0.1__G0.01",
    "STAT_RF__Dnone__L3",
    "STAT_HGB__LR0.05__LN7__L20.0",
    "FLAT_LOGREG__C0.1__CWnone",
    "ABLATION_FULL",
    "ABLATION_NO_BN",
    "ABLATION_NO_ANFIS",
    "ABLATION_NO_FTS"
  ]
}

## Closing Statement

The held-out evaluation was executed as one locked batch using configurations and thresholds selected exclusively from subject-wise OOF predictions on the 18-subject development set. No model selection or hyperparameter tuning was performed using the five-subject held-out results.