# Joint FTS-ANFIS Selection Report

## OOF Balanced Accuracy Matrix

| Membership function | 3 rules | 6 rules | 8 rules | 12 rules |
| --- | ---: | ---: | ---: | ---: |
| Triangular | 0.6032 | 0.6055 | 0.5858 | 0.5885 |
| Gaussian | 0.6047 | 0.6236 | 0.5795 | 0.5557 |
| Trapezoidal | 0.5978 | 0.6111 | 0.6080 | 0.5790 |
| Generalized bell | 0.5658 | 0.5763 | 0.5844 | 0.5558 |

## Selected Configuration

- Configuration ID: `PUB_JOINT__MF_GAUSSIAN__R6`
- Membership function: `gaussian`
- Number of rules: `6`
- OOF balanced accuracy: `0.6236`
- OOF Macro-F1: `0.6222`
- OOF ROC-AUC: `0.6201`
- Selected threshold: `0.5500`
- Fold assignment hash: `90bc408cd6a1a6e21b1d711df2ea62abd752a32246dbbfe44a322dc6a12f7ae0`

The hold-out set was not used for membership-function selection, rule-base selection, threshold selection, configuration ranking, or any publication-time hyperparameter tuning.

## Legacy Sequential Artifacts

Legacy sequential artifacts were preserved under `development/legacy_sequential_selection/` and are reference-only.