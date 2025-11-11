# tests/test_split_no_group_leak.py
import pandas as pd, yaml
from src.models.tabular_baselines import time_group_split
def test_no_group_leak():
    cfg = yaml.safe_load(open("configs/default.yaml"))
    df = pd.read_parquet(cfg["tabular"]["features_out"])
    tr, va = time_group_split(df, cfg["graph"]["account_key"], cfg["splits"]["val_days"], cfg["tabular"]["target"])
    inter = set(tr[cfg["graph"]["account_key"]].dropna().unique()).intersection(
            set(va[cfg["graph"]["account_key"]].dropna().unique()))
    assert len(inter) == 0
