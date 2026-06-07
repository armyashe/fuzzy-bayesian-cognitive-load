import numpy as np
import pandas as pd

from fbn_pipeline.models.bn_fusion import SIGNAL_NAMES


def get_state_names(n_sets: int) -> list[str]:
    if n_sets == 3:
        return ["L", "M", "H"]
    if n_sets == 5:
        return ["VL", "L", "M", "H", "VH"]
    return [f"S{i}" for i in range(n_sets)]


def get_transition_feature_names(n_sets: int) -> list[str]:
    states = get_state_names(n_sets)
    return [f"{a}->{b}" for a in states for b in states]

def signal_weights_are_uniform(signal_weights: dict) -> bool:
    if not signal_weights:
        return True
    values = [float(v) for v in signal_weights.values()]
    return bool(np.allclose(values, np.ones(len(values))))

def build_raw_df(x_sample: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(x_sample.T, columns=SIGNAL_NAMES).rename_axis("t")

def format_metric(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{float(value):.4f}"
