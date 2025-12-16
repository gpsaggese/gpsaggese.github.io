# src/data/load_data.py
from __future__ import annotations
import os
import pandas as pd

def _memory_opts():
    return dict(low_memory=False)

def load_raw(raw_dir: str, max_rows: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    tx = pd.read_csv(os.path.join(raw_dir, "train_transaction.csv"), nrows=max_rows, **_memory_opts())
    idt = pd.read_csv(os.path.join(raw_dir, "train_identity.csv"), **_memory_opts())
    return tx, idt

def merge_and_clean(tx: pd.DataFrame, idt: pd.DataFrame) -> pd.DataFrame:
    df = tx.merge(idt, on="TransactionID", how="left")
    # Ensure expected fields exist
    if "TransactionDT" not in df: 
        raise ValueError("TransactionDT missing.")
    # Down-cast common numeric fields
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    # Ensure categorical-ish keys as strings (account/device/email proxies)
    for c in ["card1","card2","card3","card4","card5","card6",
              "addr1","addr2","P_emaildomain","R_emaildomain","DeviceInfo","DeviceType"]:
        if c in df:
            df[c] = df[c].astype("string")
    return df

def run(
    raw_dir: str,
    out_path: str,
    sample_frac: float | None = None,
    random_state: int = 42,
    max_rows: int | None = None,
) -> str:
    tx, idt = load_raw(raw_dir, max_rows=max_rows)
    df = merge_and_clean(tx, idt)
    if sample_frac and 0 < sample_frac < 1.0:
        # Sample by earliest time first to keep temporal structure on tiny runs
        df = df.sort_values("TransactionDT")
        n = max(1, int(len(df) * sample_frac))
        df = df.head(n)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path

if __name__ == "__main__":
    import yaml, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-rows", type=int, default=None, help="Cap number of transaction rows read (memory saver).")
    ap.add_argument("--sample-frac", type=float, default=None, help="Override sample_frac from config.")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = run(
        raw_dir=cfg["data"]["raw_dir"],
        out_path=cfg["data"]["merged_file"],
        sample_frac=args.sample_frac if args.sample_frac is not None else cfg["data"]["sample_frac"],
        random_state=cfg["data"]["random_state"],
        max_rows=args.max_rows,
    )
    print(f"[load_data] wrote {out}")
