# src/features/txn_features.py
from __future__ import annotations

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 24 * 3600


def add_txn_features(df: pd.DataFrame, account_key: str) -> pd.DataFrame:
    out = df.copy()
    if "TransactionDT" not in out:
        raise ValueError("TransactionDT required for temporal features.")
    if account_key not in out:
        raise ValueError(f"account_key {account_key} missing")

    dt = out["TransactionDT"].astype(np.float32)
    out["hour"] = ((dt // 3600) % 24).astype("int16")
    out["dow"] = ((dt // SECONDS_PER_DAY) % 7).astype("int8")
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24).astype("float32")
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24).astype("float32")

    dt_min = dt.min()
    dt_range = max(dt.max() - dt_min, 1.0)
    out["txn_time_norm"] = ((dt - dt_min) / dt_range).astype("float32")

    group_diff = (
        out[[account_key, "TransactionDT"]]
        .sort_values("TransactionDT")
        .groupby(account_key)["TransactionDT"]
        .diff()
        .fillna(0.0)
    )
    out["hours_since_last_txn"] = (group_diff / 3600.0).astype("float32")

    if "TransactionAmt" in out:
        out["log_amt"] = np.log1p(out["TransactionAmt"]).astype("float32")
    out["has_device_change"] = (
        (out.get("DeviceInfo").notna()) & (out.get("DeviceType").notna())
    ).astype("int8")
    return out
