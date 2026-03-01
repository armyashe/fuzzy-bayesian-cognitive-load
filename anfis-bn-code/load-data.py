import numpy as np
import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

def load_data(data_folder_path: str):
    data_dir = Path(data_folder_path)

    label_df = pd.read_excel(data_dir / "labels.xlsx", index_col=0)
    temp_df  = pd.read_excel(data_dir / "temp.xlsx",   index_col=0)
    hr_df    = pd.read_excel(data_dir / "hr.xlsx",     index_col=0)
    gsr_df   = pd.read_excel(data_dir / "gsr.xlsx",    index_col=0)
    rr_df    = pd.read_excel(data_dir / "rr.xlsx",     index_col=0)

    train_ids = ['1mpau','2nxs5','5gpsc','7swyk','8a1ep','b7mrd','c24ur','dkhty',
                 'e4gay','ef5rq','f3j25','hpbxa','ibvx8','iz2ps','rc1in','tn4vl','wjxci','yljm5']
    test_ids  = ['3caqi','6frz4','bd47a','f1gjp','iz3x1']

    X_train, y_train, user_train = [], [], []
    X_test,  y_test,  user_test  = [], [], []

    for user in label_df["user_id"].unique():
        idx = (label_df["user_id"] == user)

        X_user = np.stack([
            temp_df.loc[idx].values,
            hr_df.loc[idx].values,
            gsr_df.loc[idx].values,
            rr_df.loc[idx].values
        ], axis=1)  # (N_user, 4, 30)

        y = label_df.loc[idx, "level"].values
        y[y == "rest"] = -1
        y = y.astype(int) + 1
        y[y > 0] = 1

        if user in train_ids:
            X_train.append(X_user)
            y_train.extend(y)
            user_train.extend([user] * len(y))
        elif user in test_ids:
            X_test.append(X_user)
            y_test.extend(y)
            user_test.extend([user] * len(y))

    X_train = np.vstack(X_train)
    X_test  = np.vstack(X_test)
    y_train = np.array(y_train, dtype=int)
    y_test  = np.array(y_test, dtype=int)
    user_train = np.array(user_train)
    user_test  = np.array(user_test)

    print(f"X_train:{X_train.shape}  X_test:{X_test.shape}")
    print(f"y_train  rest={(y_train==0).sum()}  task={(y_train==1).sum()}")
    print(f"y_test   rest={(y_test==0).sum()}   task={(y_test==1).sum()}")

    return X_train, y_train, user_train, X_test, y_test, user_test


if __name__ == "__main__":
    data_folder_path = r"C:\Users\doqua\Desktop\anfis-bn-code\last_30s_segments"
    load_data(data_folder_path)