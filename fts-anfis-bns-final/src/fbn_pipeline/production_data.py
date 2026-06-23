from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import _binarize_labels
from .nested_evaluation import NestedDataset


def load_full_subject_dataset(data_folder_path: str | Path) -> NestedDataset:
    data_dir = Path(data_folder_path)
    label_df = pd.read_excel(data_dir / "labels.xlsx", index_col=0)
    signal_frames = {
        "temp": pd.read_excel(data_dir / "temp.xlsx", index_col=0),
        "hr": pd.read_excel(data_dir / "hr.xlsx", index_col=0),
        "gsr": pd.read_excel(data_dir / "gsr.xlsx", index_col=0),
        "rr": pd.read_excel(data_dir / "rr.xlsx", index_col=0),
    }

    X = np.stack(
        [signal_frames[name].loc[label_df.index].values for name in ("temp", "hr", "gsr", "rr")],
        axis=1,
    )
    y = _binarize_labels(label_df["level"].values)
    subject_ids = label_df["user_id"].astype(str).to_numpy()
    sample_indices = np.arange(len(label_df), dtype=int)
    sample_ids = [f"{idx}:{subject_ids[idx]}:{label_df.iloc[idx]['timestamp']}" for idx in sample_indices]
    return NestedDataset(
        X=X,
        y=y,
        subject_ids=subject_ids,
        sample_ids=sample_ids,
        sample_indices=sample_indices,
    )
