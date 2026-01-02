# src/graph/construct_hetero_graph.py
from __future__ import annotations
import os, argparse, yaml
import pandas as pd
import torch
from torch_geometric.data import HeteroData

def build_txn_account_graph(df: pd.DataFrame, account_key: str) -> HeteroData:
    # Node ID spaces
    # transactions: use row index; accounts: categorical codes of account_key
    txn_index = df.reset_index(drop=True).index.to_series()
    acc_codes, acc_uniques = pd.factorize(df[account_key].fillna("NA_ACC").astype(str))
    # Edges transaction -> account
    src = torch.tensor(txn_index.values, dtype=torch.long)
    dst = torch.tensor(acc_codes, dtype=torch.long)

    data = HeteroData()
    data["transaction"].num_nodes = int(len(df))
    data["account"].num_nodes = int(len(acc_uniques))
    data["transaction", "owns", "account"].edge_index = torch.stack([src, dst], dim=0)
    data["account", "rev_owns", "transaction"].edge_index = torch.stack([dst, src], dim=0)

    # Lightweight node features for Phase 1
    import numpy as np
    tx_feats = []
    for col in ["log_amt","hour","dow","hour_sin","hour_cos"]:
        if col in df:
            tx_feats.append(torch.tensor(df[col].fillna(0).to_numpy(), dtype=torch.float32).unsqueeze(1))
    if tx_feats:
        data["transaction"].x = torch.cat(tx_feats, dim=1)  # [N_txn, d]
    # Labels (optional at this stage)
    if "isFraud" in df:
        data["transaction"].y = torch.tensor(df["isFraud"].to_numpy(), dtype=torch.long)
    return data

def run(in_path: str, account_key: str, save_path: str) -> str:
    df = pd.read_parquet(in_path)
    g = build_txn_account_graph(df, account_key)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(g, save_path)
    print(f"[graph] saved hetero graph to {save_path} | "
          f"txn={g['transaction'].num_nodes} acc={g['account'].num_nodes} "
          f"edges={g['transaction','owns','account'].edge_index.shape[1]}")
    return save_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run(cfg["tabular"]["features_out"], cfg["graph"]["account_key"], cfg["graph"]["save_path"])
