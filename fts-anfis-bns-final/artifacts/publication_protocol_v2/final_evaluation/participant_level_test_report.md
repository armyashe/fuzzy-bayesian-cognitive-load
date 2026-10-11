# Participant-Level Analysis

Table 8. Participant-level test results on the five unseen hold-out participants.

| Participant | N | Acc. | Bal. Acc. | TASK F1 | ROC-AUC |
| ----------- | -: | ---: | --------: | ------: | ------: |
| 3caqi | 39 | 0.6923 | 0.6961 | 0.6471 | 0.7842 |
| 6frz4 | 39 | 0.7179 | 0.7184 | 0.7179 | 0.7237 |
| bd47a | 38 | 0.7105 | 0.7105 | 0.7317 | 0.7452 |
| f1gjp | 39 | 0.8718 | 0.8724 | 0.8718 | 0.8026 |
| iz3x1 | 38 | 0.7632 | 0.7632 | 0.7429 | 0.8366 |

unweighted statistics across the five unseen participants (not pooled over 193 samples):

- Mean: Acc. `0.7511`, Bal. Acc. `0.7521`, TASK F1 `0.7423`, ROC-AUC `0.7785`
- Standard deviation: Acc. `0.0647`, Bal. Acc. `0.0642`, TASK F1 `0.0729`, ROC-AUC `0.0403`
- Minimum Accuracy: `0.6923`
- Maximum Accuracy: `0.8718`
- Highest Accuracy participant: `f1gjp`
- Lowest Accuracy participant: `3caqi`

## Section 4.4 Summary

Table 8 reports participant-level performance on the five unseen hold-out participants. Accuracy ranged from 0.6923 to 0.8718 (unweighted mean 0.7511, SD 0.0647), indicating substantial between-participant variation in the final evaluation. These participant-level results are reported only as a post hoc stability analysis on unseen participants and were not used to select the model, tune the threshold, or adjust any hyperparameter.

These participant-level results must not be used for model selection, threshold tuning, or participant-specific retraining.