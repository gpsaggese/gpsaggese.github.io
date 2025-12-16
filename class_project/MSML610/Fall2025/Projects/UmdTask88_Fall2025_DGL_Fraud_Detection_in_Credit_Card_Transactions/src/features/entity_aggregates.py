# src/features/entity_aggregates.py
from __future__ import annotations

import numpy as np
import pandas as pd


def _leak_free_account_features(group: pd.DataFrame, target: str) -> pd.DataFrame:
    """Compute cumulative aggregates using only historical transactions per account."""
    sorted_group = group.sort_values("TransactionDT")
    amt = sorted_group["TransactionAmt"].fillna(0.0).to_numpy(dtype=np.float32)
    labels = sorted_group[target].fillna(0.0).to_numpy(dtype=np.float32)

    prev_cnt = np.arange(len(sorted_group), dtype=np.int32)
    cum_amt = np.cumsum(amt) - amt
    mean = np.divide(cum_amt, prev_cnt, out=np.zeros_like(amt), where=prev_cnt > 0)

    cum_sq = np.cumsum(amt ** 2) - amt ** 2
    valid = prev_cnt > 1
    denom = prev_cnt.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        var_vals = np.zeros_like(amt)
        var_vals[valid] = (
            cum_sq[valid] - (cum_amt[valid] ** 2) / denom[valid]
        ) / (denom[valid] - 1.0)
    std = np.sqrt(np.clip(var_vals, a_min=0.0, a_max=None))

    a0, b0 = 1.0, 20.0
    cum_pos = np.cumsum(labels) - labels
    fraud_rate = (cum_pos + a0) / (prev_cnt + a0 + b0)

    return pd.DataFrame(
        {
            "acc_txn_cnt": prev_cnt,
            "acc_amt_mean": mean,
            "acc_amt_std": std,
            "acc_fraud_rate_smooth": fraud_rate,
        },
        index=sorted_group.index,
    )


def add_account_aggregates(
    df: pd.DataFrame,
    account_key: str,
    target: str = "isFraud",
) -> pd.DataFrame:
    required_cols = [account_key, "TransactionDT", "TransactionAmt", target]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for aggregates: {missing}")

    out = df.copy()
    feats = out.groupby(account_key, dropna=False, group_keys=False).apply(
        lambda grp: _leak_free_account_features(grp, target)
    )
    for col in feats.columns:
        out[col] = feats[col].astype("float32" if col != "acc_txn_cnt" else "int32")
    return out
