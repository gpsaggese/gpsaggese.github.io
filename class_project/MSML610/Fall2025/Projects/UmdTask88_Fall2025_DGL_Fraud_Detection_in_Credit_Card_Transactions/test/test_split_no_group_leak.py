# tests/test_split_no_group_leak.py
import pandas as pd, yaml
from src.utils.splits import temporal_group_split_frames
def test_no_group_leak():
    cfg = yaml.safe_load(open("configs/default.yaml"))
    df = pd.read_parquet(cfg["tabular"]["features_out"])
    tr, va, te = temporal_group_split_frames(
        df,
        cfg["graph"]["account_key"],
        cfg["splits"]["val_days"],
        cfg["splits"]["test_days"],
    )
    def uniq_accounts(frame):
        return set(frame[cfg["graph"]["account_key"]].dropna().unique())
    acc_tr = uniq_accounts(tr)
    acc_va = uniq_accounts(va)
    acc_te = uniq_accounts(te)
    assert acc_tr.isdisjoint(acc_va)
    assert acc_tr.isdisjoint(acc_te)
    assert acc_va.isdisjoint(acc_te)
