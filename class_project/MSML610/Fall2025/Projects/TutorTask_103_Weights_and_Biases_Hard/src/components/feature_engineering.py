import os
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import config_manager
from src.logging.logger import WandbLogger


class FeatureEngineering:
    def __init__(self, config_path: str = "config", wandb_logger: Optional[WandbLogger] = None):
        self.config = config_manager
        self.params = self.config.load_params()
        self.fe_cfg = self.params["feature_engineering"]
        self.training_cfg = self.params["training"]
        self.logger = wandb_logger or WandbLogger(config_path)
        self.proc_dir = Path(self.params["data_collection"]["processed_dir"])
        self.artifacts_dir = Path(self.params["evaluation"]["artifacts_dir"])
        self.plots_dir = Path(self.params["evaluation"]["plots_dir"])
        self.proc_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    # ------------ Missing value handling ------------
    def fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_index()
        df = df.ffill().bfill()
        return df

    # ------------ Core helpers ------------
    def _close_series(self, df: pd.DataFrame) -> pd.Series:
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return close

    # ------------ Features ------------
    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        close = self._close_series(df)
        df["return_pct"] = close.pct_change()
        df["return_log"] = np.log(close).diff()
        return df

    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        close = self._close_series(df)
        df[f"sma_{self.fe_cfg['short_window']}"] = close.rolling(self.fe_cfg["short_window"]).mean()
        df[f"sma_{self.fe_cfg['long_window']}"] = close.rolling(self.fe_cfg["long_window"]).mean()
        df[f"sma_{self.fe_cfg['very_long_window']}"] = close.rolling(self.fe_cfg["very_long_window"]).mean()
        return df

    def add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        window = self.fe_cfg["rsi_period"]
        close = self._close_series(df)
        delta = close.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        roll_up = pd.Series(gain, index=df.index).rolling(window).mean()
        roll_down = pd.Series(loss, index=df.index).rolling(window).mean()
        rs = roll_up / (roll_down + 1e-9)
        df[f"rsi_{window}"] = 100 - (100 / (1 + rs))
        return df

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        fast, slow, signal = self.fe_cfg["macd_fast"], self.fe_cfg["macd_slow"], self.fe_cfg["macd_signal"]
        close = self._close_series(df)
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_hist"] = macd_line - signal_line
        return df

    def add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        win = self.fe_cfg["volatility_window"]
        df[f"volatility_{win}"] = df["return_log"].rolling(win).std()
        lw = self.fe_cfg["long_window"]
        close = self._close_series(df)
        sma = close.rolling(lw).mean()
        std = close.rolling(lw).std()
        df[f"bb_upper_{lw}"] = sma + 2 * std
        df[f"bb_lower_{lw}"] = sma - 2 * std
        return df

    def add_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        close = self._close_series(df)
        for lag in self.fe_cfg["lag_features"]:
            df[f"lag_close_{lag}"] = close.shift(lag)
            df[f"lag_return_pct_{lag}"] = df["return_pct"].shift(lag)
        return df

    # ------------ Orchestration ------------
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.fill_missing(df)
        df = self.add_returns(df)
        df = self.add_moving_averages(df)
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_volatility(df)
        df = self.add_lags(df)
        df = df.dropna()
        return df

    def drop_high_corr(self, df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
        # Never drop the target column (needed by Preprocessor)
        target_col = self.training_cfg.get("target_column", "Close")
        protected = {target_col}

        corr = df.select_dtypes(include=[np.number]).corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        to_drop = []
        for col in upper.columns:
            if col in protected:
                continue
            if any(upper[col] > threshold):
                to_drop.append(col)

        if to_drop:
            self.logger.info(f"Dropping highly correlated features (>{threshold}): {to_drop}")
            df = df.drop(columns=to_drop)
        return df

    def summarize(self, df: pd.DataFrame, name: str = "data") -> None:
        self.logger.info(f"{name}: rows={len(df)}, cols={len(df.columns)}")
        self.logger.info(f"{name} features: {list(df.columns)}")

    def save(self, df: pd.DataFrame, name: str = "data") -> Path:
        out_path = self.proc_dir / f"{name}_features.csv"
        df.to_csv(out_path)
        self.logger.info(f"Saved features to {out_path}")
        return out_path

    def plot_correlation(self, df: pd.DataFrame, name: str = "data") -> Optional[Path]:
        corr = df.select_dtypes(include=[np.number]).corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="RdBu_r", center=0)
        plt.title("Feature Correlation")
        plot_path = self.plots_dir / f"{name}_corr.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        self.logger.info(f"Saved correlation heatmap to {plot_path}")
        return plot_path

    def run(self, df: pd.DataFrame, name: str = "data") -> pd.DataFrame:
        feats = self.build_features(df)
        feats = self.drop_high_corr(feats, threshold=self.fe_cfg.get("corr_drop_threshold", 0.9))
        self.summarize(feats, name)
        out_path = self.save(feats, name)
        plot_path = self.plot_correlation(feats, name)
        if self.logger.run:
            self.logger.log_artifact(str(out_path), f"{name}_features", "dataset")
            if plot_path:
                self.logger.log_plot(plot_path, f"{name}_correlation_heatmap")
        return feats