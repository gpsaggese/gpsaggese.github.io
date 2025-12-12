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

    def scale(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        # Separate target and features
        y_train = train[self.target_col].values
        y_val = val[self.target_col].values
        y_test = test[self.target_col].values

        X_train_df = train.drop(columns=[self.target_col])
        X_val_df = val.drop(columns=[self.target_col])
        X_test_df = test.drop(columns=[self.target_col])

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
        train, val, test = self.time_split(df)
        X_train, X_val, X_test, y_train, y_val, y_test, meta = self.scale(train, val, test)

        X_train_seq, y_train_seq = self.build_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.build_sequences(X_val, y_val)
        X_test_seq, y_test_seq = self.build_sequences(X_test, y_test)

        self.logger.info(
            f"Sequences shapes -> "
            f"train: {X_train_seq.shape}, val: {X_val_seq.shape}, test: {X_test_seq.shape}"
        )

        return {
            "X_train": X_train_seq,
            "y_train": y_train_seq,
            "X_val": X_val_seq,
            "y_val": y_val_seq,
            "X_test": X_test_seq,
            "y_test": y_test_seq,
            "feature_names": meta["feature_names"],
            "scaler": meta["scaler"],
        }