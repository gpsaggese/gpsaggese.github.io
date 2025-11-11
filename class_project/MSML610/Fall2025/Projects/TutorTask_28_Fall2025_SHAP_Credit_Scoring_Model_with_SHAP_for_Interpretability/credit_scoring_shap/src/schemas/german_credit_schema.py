"""Schema and helpers for the German Credit dataset.

This module centralizes target detection, positive class mapping, and basic
feature typing (numeric vs categorical). Adjust here if your CSV differs.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd

@dataclass(frozen=True)
class TargetSpec:
    candidates: Tuple[str, ...] = (
        "target","Risk","credit_risk","Creditability","Class","default"
    )
    positive_values: Tuple[object, ...] = (
        "bad","Bad","2","default",1,"bad risk"
    )

def detect_and_binarize_target(df: pd.DataFrame, spec: TargetSpec = TargetSpec()) -> Tuple[pd.DataFrame, str]:
    target_col = None
    for c in spec.candidates:
        if c in df.columns:
            target_col = c
            break
    if target_col is None:
        raise KeyError(f"Could not find target column. Looked for: {spec.candidates}")

    y = df[target_col].copy()
    if y.dtype.kind in "OUS":
        y_norm = y.astype(str).str.strip().str.lower()
        y_bin = y_norm.isin([str(v).lower() for v in spec.positive_values]).astype(int)
    else:
        vals = set(pd.unique(y))
        if vals == {0,1}:
            y_bin = y.astype(int)
        elif vals == {1,2}:
            y_bin = (y == 2).astype(int)
        else:
            # Fallback heuristic
            y_bin = (y > y.median()).astype(int)

    df = df.drop(columns=[target_col]).copy()
    df[target_col] = y_bin
    return df, target_col

def basic_feature_types(df: pd.DataFrame, target_col: str):
    cats = df.drop(columns=[target_col]).select_dtypes(include=['object','category']).columns.tolist()
    nums = [c for c in df.columns if c not in cats + [target_col]]
    return nums, cats
