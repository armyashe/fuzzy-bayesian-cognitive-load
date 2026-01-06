# Fuzzy Bayesian Network for Cognitive Load Classification

## 1. Introduction
Cognitive load estimation plays a critical role in human–computer interaction,
education, and adaptive systems. This project proposes an interpretable
Fuzzy Bayesian Network (FBN) to classify cognitive load levels (Low, Medium, High)
using physiological signals.

## 2. Physiological Signals
The following signals are used:
- Heart Rate (HR)
- Respiration Rate (RR)
- Galvanic Skin Response (GSR)
- Skin Temperature (Temp)
### Dataset
This work uses the cognitive load dataset released for the UbiComp 2020 challenge,
provided by Gjoreski et al.

Dataset source:
https://github.com/MartinGjoreski/martingjoreski.github.io/blob/master/files/CogDatasets.rar

Signals are segmented into fixed-length windows (e.g., 30 seconds).

## Relation to Previous Work

This work is an extension of our previous study on cognitive load classification
using a classical Bayesian Network (BN), where physiological features were modeled
using crisp states.

Previous implementation:
https://github.com/TyTy7023/cogLoad_experiment.git

In contrast to the previous BN-based approach, this study introduces:
- Fuzzy linguistic modeling of physiological signals
- Fuzzy Bayesian Network (FBN) inference
- Improved interpretability under noisy real-world measurements
- Direct and fair comparison with the baseline Bayesian Network model

## 3. Methodology
The proposed framework consists of:
1. Feature extraction from raw physiological signals
2. Fuzzification using linguistic variables (Low, Medium, High)
3. Bayesian Network construction with fuzzy states
4. Probabilistic inference for cognitive load classification

## 4. Model Structure
The Bayesian Network structure assumes conditional independence between
physiological signals given the cognitive load state.

## 5. Experiments
- Subject-independent evaluation
- Multi-class cognitive load classification
- Metrics: Accuracy, F1-score

## 6. Interpretability
The proposed model provides:
- Explicit fuzzy membership functions
- Interpretable CPTs
- Transparent inference process

## 7. Future Work
- Dynamic Bayesian Network (DBN)
- Temporal fuzzy state transitions
- Online cognitive load monitoring

## 8. References
[1] M. Gjoreski, H. Gjoreski, M. Luštrek, and M. Gams,
"Monitoring cognitive load with wearable sensors,"
Proc. ACM IMWUT, vol. 4, no. 1, 2020.

(To be updated)
