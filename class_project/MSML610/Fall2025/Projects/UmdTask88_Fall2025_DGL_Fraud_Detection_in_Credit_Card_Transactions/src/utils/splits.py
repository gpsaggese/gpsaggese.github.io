# src/utils/splits.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

SECONDS_PER_DAY = 24 * 3600


@dataclass(frozen=True)
class TemporalSplitIndices:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def _filter_indices_by_accounts(
    df: pd.DataFrame,
    indices: np.ndarray,
    account_key: str,
    blocked_accounts: set,
) -> np.ndarray:
    """Remove indices where the account already appeared in a previous split."""
    if len(indices) == 0 or not blocked_accounts:
        return indices
    subset_accounts = df.iloc[indices][account_key]
    keep_mask = ~subset_accounts.isin(blocked_accounts).to_numpy()
    return indices[keep_mask]


def temporal_group_split_indices(
    df: pd.DataFrame,
    account_key: str,
    val_days: int,
    test_days: int,
    time_col: str = "TransactionDT",
) -> TemporalSplitIndices:
    """
    Split dataframe rows into train/val/test based on time windows and drop accounts
    that would otherwise appear in multiple splits.
    """
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' required for temporal split is missing.")
    if account_key not in df.columns:
        raise ValueError(f"Account key '{account_key}' missing from dataframe.")

    df_base = df.reset_index(drop=True).copy()
    df_base["row_id"] = df_base.index
    df_ord = df_base.sort_values(time_col)

    max_time = df_ord[time_col].max()
    test_cutoff = max_time - test_days * SECONDS_PER_DAY
    val_cutoff = test_cutoff - val_days * SECONDS_PER_DAY

    train_idx = df_ord[df_ord[time_col] < val_cutoff]["row_id"].to_numpy(dtype=np.int64)
    val_idx = df_ord[
        (df_ord[time_col] >= val_cutoff) & (df_ord[time_col] < test_cutoff)
    ]["row_id"].to_numpy(dtype=np.int64)
    test_idx = df_ord[df_ord[time_col] >= test_cutoff]["row_id"].to_numpy(dtype=np.int64)

    train_accounts = set(df_base.iloc[train_idx][account_key].dropna().unique())
    val_idx = _filter_indices_by_accounts(df_base, val_idx, account_key, train_accounts)
    val_accounts = set(df_base.iloc[val_idx][account_key].dropna().unique())
    test_idx = _filter_indices_by_accounts(
        df_base, test_idx, account_key, train_accounts.union(val_accounts)
    )

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            "Temporal split produced an empty subset. "
            "Adjust val_days/test_days or confirm TransactionDT coverage."
        )

    return TemporalSplitIndices(train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


def temporal_group_split_frames(
    df: pd.DataFrame,
    account_key: str,
    val_days: int,
    test_days: int,
    time_col: str = "TransactionDT",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper returning train/val/test dataframes."""
    splits = temporal_group_split_indices(df, account_key, val_days, test_days, time_col)
    df_ord = df.reset_index(drop=True)
    return (
        df_ord.iloc[splits.train_idx].copy(),
        df_ord.iloc[splits.val_idx].copy(),
        df_ord.iloc[splits.test_idx].copy(),
    )


def indices_to_mask(num_rows: int, indices: np.ndarray) -> np.ndarray:
    """Create a boolean mask of length num_rows with True at provided indices."""
    mask = np.zeros(num_rows, dtype=bool)
    if len(indices):
        mask[indices] = True
    return mask
