# src/features/txn_features.py
from __future__ import annotations
import pandas as pd
import numpy as np

def add_txn_features(df: pd.DataFrame, account_key: str) -> pd.DataFrame:
    out = df.copy()
    # time features
    # TransactionDT is seconds from a reference; derive hour-of-day & dow
    out["hour"] = ((out["TransactionDT"] // 3600) % 24).astype("int16")
    out["dow"]  = ((out["TransactionDT"] // (3600*24)) % 7).astype("int8")
    out["hour_sin"] = np.sin(2*np.pi*out["hour"]/24).astype("float32")
    out["hour_cos"] = np.cos(2*np.pi*out["hour"]/24).astype("float32")
    # amount
    if "TransactionAmt" in out:
        out["log_amt"] = np.log1p(out["TransactionAmt"]).astype("float32")
    # simple device/email flags
    out["has_device_change"] = ((out.get("DeviceInfo").notna()) & (out.get("DeviceType").notna())).astype("int8")
    # ensure account key
    if account_key not in out:
        raise ValueError(f"account_key {account_key} missing")
    return out
