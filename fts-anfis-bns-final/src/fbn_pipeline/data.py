from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_IDS = [
    "1mpau", "2nxs5", "5gpsc", "7swyk", "8a1ep", "b7mrd", "c24ur", "dkhty",
    "e4gay", "ef5rq", "f3j25", "hpbxa", "ibvx8", "iz2ps", "rc1in", "tn4vl",
    "wjxci", "yljm5",
]
TEST_IDS = ["3caqi", "6frz4", "bd47a", "f1gjp", "iz3x1"]


@dataclass(slots=True)
class DatasetSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    user_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    user_test: np.ndarray


@dataclass(slots=True)
class TrainingSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    user_train: np.ndarray


def _binarize_labels(raw_labels: np.ndarray) -> np.ndarray:
    y = raw_labels.astype(object).copy()
    y[y == "rest"] = -1
    y = y.astype(int) + 1
    y[y > 0] = 1
    return y.astype(int)


def load_dataset(data_folder_path: str | Path) -> DatasetSplit:
    data_dir = Path(data_folder_path)

    label_df = pd.read_excel(data_dir / "labels.xlsx", index_col=0)
    signal_frames = {
        "temp": pd.read_excel(data_dir / "temp.xlsx", index_col=0),
        "hr": pd.read_excel(data_dir / "hr.xlsx", index_col=0),
        "gsr": pd.read_excel(data_dir / "gsr.xlsx", index_col=0),
        "rr": pd.read_excel(data_dir / "rr.xlsx", index_col=0),
    }

    X_train, y_train, user_train = [], [], []
    X_test, y_test, user_test = [], [], []

    for user in label_df["user_id"].unique():
        idx = label_df["user_id"] == user
        X_user = np.stack(
            [signal_frames[name].loc[idx].values for name in ("temp", "hr", "gsr", "rr")],
            axis=1,
        )
        y_user = _binarize_labels(label_df.loc[idx, "level"].values)

        if user in TRAIN_IDS:
            X_train.append(X_user)
            y_train.extend(y_user)
            user_train.extend([user] * len(y_user))
        elif user in TEST_IDS:
            X_test.append(X_user)
            y_test.extend(y_user)
            user_test.extend([user] * len(y_user))

    return DatasetSplit(
        X_train=np.vstack(X_train),
        y_train=np.array(y_train, dtype=int),
        user_train=np.array(user_train),
        X_test=np.vstack(X_test),
        y_test=np.array(y_test, dtype=int),
        user_test=np.array(user_test),
    )


def load_training_split(data_folder_path: str | Path) -> TrainingSplit:
    dataset = load_dataset(data_folder_path)
    return TrainingSplit(
        X_train=dataset.X_train,
        y_train=dataset.y_train,
        user_train=dataset.user_train,
    )
