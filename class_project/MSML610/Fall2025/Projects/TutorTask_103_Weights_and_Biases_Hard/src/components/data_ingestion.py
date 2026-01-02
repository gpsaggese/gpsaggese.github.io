# src/components/data_ingestion.py
import os
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf

from src.utils.config import config_manager
from src.logging.logger import WandbLogger


class DataIngestion:
    def __init__(self, config_path: str = "config", wandb_logger: Optional[WandbLogger] = None):
        self.config = config_manager
        self.params = self.config.load_params()
        self.data_cfg = self.params["data_collection"]
        self.logger = wandb_logger or WandbLogger(config_path)
        self.raw_dir = Path(self.data_cfg["data_dir"])
        self.proc_dir = Path(self.data_cfg["processed_dir"])
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.proc_dir.mkdir(parents=True, exist_ok=True)

    def fetch(self) -> pd.DataFrame:
        ticker = self.data_cfg["ticker_symbol"]
        start = self.data_cfg["start_date"]
        end = self.data_cfg["end_date"]
        interval = self.data_cfg["interval"]
        self.logger.info(f"Fetching {ticker} from {start} to {end or 'now'} @ {interval}")
        # yfinance may return MultiIndex columns (e.g. (Price, Ticker)).
        # We normalize to a flat OHLCV DataFrame with columns like "Open/High/Low/Close/Volume".
        df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False)
        if df.empty:
            raise ValueError(f"No data returned for ticker {ticker}")
        if isinstance(df.columns, pd.MultiIndex):
            # Try to select the requested ticker level if present.
            try:
                df = df.xs(ticker, axis=1, level=-1, drop_level=True)
            except Exception:
                # Fallback: stringify columns to avoid downstream crashes.
                df.columns = ["_".join(map(str, c)) for c in df.columns]
        df.index.name = "Date"
        return df

    def validate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
        # Sanity-check only price columns (Volume can be 0; some providers may emit 0s).
        price_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in df.columns]
        if price_cols:
            if (df[price_cols] <= 0).any().any():
                self.logger.warning("Found non-positive price values; check data sanity.")
        return df

    def save(self, df: pd.DataFrame, name: str = "data") -> Tuple[Path, Path]:
        raw_path = self.raw_dir / f"{name}_raw.csv"
        proc_path = self.proc_dir / f"{name}_processed.csv"
        df.to_csv(raw_path)
        df.to_csv(proc_path)
        self.logger.info(f"Saved raw to {raw_path} and processed to {proc_path}")
        return raw_path, proc_path

    def run(self, name: str = "data") -> pd.DataFrame:
        df = self.fetch()
        df = self.validate(df)
        raw_path, proc_path = self.save(df, name)
        if self.logger.run:
            self.logger.log_artifact(str(raw_path), f"{name}_raw", "dataset")
            self.logger.log_artifact(str(proc_path), f"{name}_processed", "dataset")
            self.logger.log_table(f"{name}_head", df.head(20))
        return df