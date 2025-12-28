# src/features/make_features.py
from __future__ import annotations
import os, argparse, yaml, pandas as pd
from .txn_features import add_txn_features
from .entity_aggregates import add_account_aggregates

def run(in_path: str, out_path: str, account_key: str, target: str) -> str:
    df = pd.read_parquet(in_path)
    df = add_txn_features(df, account_key)
    df = add_account_aggregates(df, account_key, target)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = run(
        in_path=cfg["data"]["merged_file"],
        out_path=cfg["tabular"]["features_out"],
        account_key=cfg["graph"]["account_key"],
        target=cfg["tabular"]["target"],
    )
    print(f"[make_features] wrote {out}")
