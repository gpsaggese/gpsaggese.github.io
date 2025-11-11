# src/models/tabular_baselines.py
from __future__ import annotations
import os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

BASIC_FEATS = [
    "log_amt","hour","dow","hour_sin","hour_cos",
    "acc_txn_cnt","acc_amt_mean","acc_amt_std",
    # "acc_fraud_rate_smooth",  # removed to avoid leakage in Phase 1
]

def time_group_split(df: pd.DataFrame, account_key: str, val_days: int, target: str):
    df = df.sort_values("TransactionDT")
    # Split by last N days; keep groups (accounts) from bleeding across sets
    cutoff = df["TransactionDT"].max() - val_days*24*3600
    tr = df[df["TransactionDT"] <= cutoff].copy()
    va = df[df["TransactionDT"]  > cutoff].copy()
    # drop any account that appears in both (conservative)
    inter = set(tr[account_key].dropna().unique()).intersection(set(va[account_key].dropna().unique()))
    tr = tr[~tr[account_key].isin(inter)]
    va = va[~va[account_key].isin(inter)]
    return tr, va

def run_baseline(features_path: str, report_dir: str, account_key: str, val_days: int, target: str) -> dict:
    os.makedirs(report_dir, exist_ok=True)
    df = pd.read_parquet(features_path)
    tr, va = time_group_split(df, account_key, val_days, target)
    Xtr, ytr = tr[BASIC_FEATS].fillna(0).to_numpy(), tr[target].to_numpy()
    Xva, yva = va[BASIC_FEATS].fillna(0).to_numpy(), va[target].to_numpy()

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xva)[:,1]
    pr_auc = float(average_precision_score(yva, p))
    # default 0.5 threshold for Phase 1
    yhat = (p >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(yva, yhat, average="binary", zero_division=0)

    report = dict(
        model="LogisticRegression(balanced)",
        features=BASIC_FEATS,
        val_days=val_days,
        metrics=dict(PR_AUC=pr_auc, precision=float(prec), recall=float(rec), f1=float(f1)),
        n_train=int(len(tr)), n_val=int(len(va))
    )
    with open(os.path.join(report_dir, "baseline_metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    import argparse, yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run_baseline(
        features_path=cfg["tabular"]["features_out"],
        report_dir=cfg["tabular"]["report_dir"],
        account_key=cfg["graph"]["account_key"],
        val_days=cfg["splits"]["val_days"],
        target=cfg["tabular"]["target"],
    )
