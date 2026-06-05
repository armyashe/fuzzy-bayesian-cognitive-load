import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def label_distribution(y: np.ndarray) -> pd.DataFrame:
    values, counts = np.unique(y, return_counts=True)
    mapping = {0: "rest", 1: "task"}
    total = len(y)
    rows = []
    for v, c in zip(values, counts):
        rows.append(
            {
                "label": mapping.get(int(v), str(v)),
                "count": int(c),
                "ratio": round(float(c / total), 4),
            }
        )
    return pd.DataFrame(rows)


def confusion_df(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=["Actual rest", "Actual task"],
        columns=["Pred rest", "Pred task"],
    )

def cpt_dict_to_df(cpt_signal_dict: dict) -> pd.DataFrame:
    return pd.DataFrame(cpt_signal_dict).T


def signal_weights_df(signal_weights: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [{"signal": k, "weight": round(float(v), 4)} for k, v in signal_weights.items()]
    )

def compute_soft_influence_table(bn_trace: dict) -> pd.DataFrame | None:
    if bn_trace.get("mode") != "soft":
        return None

    rows = []
    for class_id, class_name in [(0, "rest"), (1, "task")]:
        for sig, value in bn_trace.get("log_likelihoods", {}).get(class_id, {}).items():
            rows.append(
                {
                    "class": class_name,
                    "signal": sig,
                    "soft_log_contribution": round(float(value), 4),
                    "effective_likelihood": round(
                        float(
                            bn_trace.get("effective_likelihoods", {})
                            .get(class_name, {})
                            .get(sig, np.nan)
                        ),
                        4,
                    ),
                }
            )
    return pd.DataFrame(rows) if rows else None


def compute_hard_debug_contribution_table(bn_trace: dict) -> pd.DataFrame:
    contrib_rows = []
    for class_id, class_name in [(0, "rest"), (1, "task")]:
        for sig, value in bn_trace.get("log_likelihoods", {}).get(class_id, {}).items():
            contrib_rows.append(
                {
                    "class": class_name,
                    "signal": sig,
                    "log_contribution": round(float(value), 4),
                }
            )
    return pd.DataFrame(contrib_rows)
