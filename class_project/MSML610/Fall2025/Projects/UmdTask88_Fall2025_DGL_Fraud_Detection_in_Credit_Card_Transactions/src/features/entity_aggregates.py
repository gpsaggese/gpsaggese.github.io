# src/features/entity_aggregates.py
from __future__ import annotations
import pandas as pd
import numpy as np

def add_account_aggregates(df: pd.DataFrame, account_key: str, target: str = "isFraud") -> pd.DataFrame:
    out = df.copy()
    grp = out.groupby(account_key, dropna=False)
    out["acc_txn_cnt"] = grp[target].transform("size").astype("int32")
    out["acc_amt_mean"] = grp["TransactionAmt"].transform("mean").astype("float32")
    out["acc_amt_std"]  = grp["TransactionAmt"].transform("std").fillna(0).astype("float32")
    # smoothed fraud rate (beta prior)
    a0, b0 = 1.0, 20.0
    acc_pos = grp[target].transform("sum")
    acc_n   = grp[target].transform("size")
    out["acc_fraud_rate_smooth"] = ((acc_pos + a0) / (acc_n + a0 + b0)).astype("float32")
    return out
