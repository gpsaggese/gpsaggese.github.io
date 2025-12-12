# src/components/preprocessor.py
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.config import config_manager
from src.logging.logger import WandbLogger


class Preprocessor:
    def __init__(self, config_path: str = "config", wandb_logger: WandbLogger = None):
        self.config = config_manager
        self.params = self.config.load_params()
        self.train_cfg = self.params["training"]
        self.logger = wandb_logger or WandbLogger(config_path)
        self.target_col = self.train_cfg["target_column"]
        self.horizon = int(self.train_cfg.get("forecast_horizon", 1))
        self.seq_len = self.params["model"]["lstm"]["sequence_length"]

    def time_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        test_size = self.train_cfg["test_size"]
        val_size = self.train_cfg["validation_size"]

        n = len(df)
        test_n = int(n * test_size)
        train_val_n = n - test_n
        val_n = int(train_val_n * val_size)

        train = df.iloc[: train_val_n - val_n]
        val = df.iloc[train_val_n - val_n : train_val_n]
        test = df.iloc[train_val_n:]
        self.logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test

    def scale(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        target_col: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        # Separate target and features.
        y_train = train[target_col].values
        y_val = val[target_col].values
        y_test = test[target_col].values

        # Important: keep current Close as a feature, unless it is the target_col.
        X_train_df = train.drop(columns=[target_col])
        X_val_df = val.drop(columns=[target_col])
        X_test_df = test.drop(columns=[target_col])

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_df)
        X_val = scaler.transform(X_val_df)
        X_test = scaler.transform(X_test_df)

        meta = {
            "feature_names": list(X_train_df.columns),
            "scaler": scaler
        }
        return X_train, X_val, X_test, y_train, y_val, y_test, meta

    def build_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        seq_len = self.seq_len
        X_seq, y_seq = [], []
        for i in range(seq_len, len(X)):
            X_seq.append(X[i - seq_len : i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)

    def run(self, df: pd.DataFrame) -> Dict[str, Any]:
        df = df.copy()
        if self.horizon < 1:
            raise ValueError("forecast_horizon must be >= 1")

        # Supervised forecasting: features at time t -> target at time t+h.
        target_next_col = "__target__"
        df[target_next_col] = df[self.target_col].shift(-self.horizon)
        df = df.iloc[: -self.horizon]

        train, val, test = self.time_split(df)
        X_train, X_val, X_test, y_train, y_val, y_test, meta = self.scale(train, val, test, target_col=target_next_col)

        # Unscaled DataFrames/Series for statsmodels.
        X_train_df = train.drop(columns=[target_next_col])
        X_val_df = val.drop(columns=[target_next_col])
        X_test_df = test.drop(columns=[target_next_col])
        y_train_s = train[target_next_col]
        y_val_s = val[target_next_col]
        y_test_s = test[target_next_col]

        X_train_seq, y_train_seq = self.build_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.build_sequences(X_val, y_val)
        X_test_seq, y_test_seq = self.build_sequences(X_test, y_test)

        self.logger.info(
            f"Flat shapes -> train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}"
        )
        self.logger.info(
            f"Seq shapes  -> train: {X_train_seq.shape}, val: {X_val_seq.shape}, test: {X_test_seq.shape}"
        )

        return {
            # Flat arrays for classical models (linear/trees/boosting)
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            # Sequence arrays for LSTM/CNN/seq models
            "X_train_seq": X_train_seq,
            "y_train_seq": y_train_seq,
            "X_val_seq": X_val_seq,
            "y_val_seq": y_val_seq,
            "X_test_seq": X_test_seq,
            "y_test_seq": y_test_seq,
            "feature_names": meta["feature_names"],
            "scaler": meta["scaler"],
            # Unscaled frames/series (time-indexed) for statsmodels.
            "train_df": train,
            "val_df": val,
            "test_df": test,
            "X_train_df": X_train_df,
            "X_val_df": X_val_df,
            "X_test_df": X_test_df,
            "y_train_s": y_train_s,
            "y_val_s": y_val_s,
            "y_test_s": y_test_s,
            "target_next_col": target_next_col,
        }