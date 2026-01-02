# src/models/tabular_baselines.py
from __future__ import annotations
import os, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

from src.utils.splits import temporal_group_split_frames

BASIC_FEATS = [
    "log_amt",
    "hour",
    "dow",
    "hour_sin",
    "hour_cos",
    "txn_time_norm",
    "hours_since_last_txn",
    "acc_txn_cnt",
    "acc_amt_mean",
    "acc_amt_std",
    "acc_fraud_rate_smooth",
    "has_device_change",
]

def _pick_threshold_from_pr_curve(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """
    Choose the threshold that maximizes F1 on the precision-recall curve.
    Returns (threshold, f1_at_threshold).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_scores = (2 * precision * recall) / np.clip(precision + recall, 1e-12, None)
    best_idx = np.nanargmax(f1_scores)
    # precision_recall_curve returns len(thresholds) = len(precision)-1
    best_threshold = float(thresholds[min(best_idx, len(thresholds) - 1)])
    return best_threshold, float(f1_scores[best_idx])


def _compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    pr_auc = float(average_precision_score(y_true, probs))
    yhat = (probs >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, yhat, average="binary", zero_division=0
    )
    return dict(PR_AUC=pr_auc, precision=float(prec), recall=float(rec), f1=float(f1))


def run_baseline(
    features_path: str,
    report_dir: str,
    account_key: str,
    val_days: int,
    test_days: int,
    target: str,
    threshold: float = 0.5,
    model_type: str = "logreg",
    tune_threshold: bool = True,
) -> dict:
    os.makedirs(report_dir, exist_ok=True)
    df = pd.read_parquet(features_path)
    try:
        tr, va, te = temporal_group_split_frames(df, account_key, val_days, test_days)
    except ValueError:
        # Fallback for tiny demo samples: simple time-based 60/20/20 split without group filtering
        df_ord = df.sort_values("TransactionDT").reset_index(drop=True)
        n = len(df_ord)
        if n < 10:
            raise ValueError("Not enough rows to build even a tiny split; increase sample or max_rows.")
        tr_end = int(n * 0.6)
        va_end = int(n * 0.8)
        tr, va, te = (
            df_ord.iloc[:tr_end].copy(),
            df_ord.iloc[tr_end:va_end].copy(),
            df_ord.iloc[va_end:].copy(),
        )
        print("[run_baseline] Using fallback 60/20/20 time split for small sample.")

    Xtr_raw, ytr = tr[BASIC_FEATS].fillna(0), tr[target].to_numpy()
    Xva_raw, yva = va[BASIC_FEATS].fillna(0), va[target].to_numpy()
    Xte_raw, yte = te[BASIC_FEATS].fillna(0), te[target].to_numpy()

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr_raw)
    Xva = scaler.transform(Xva_raw)
    Xte = scaler.transform(Xte_raw)

    if model_type == "logreg":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=None)
        clf.fit(Xtr, ytr)
        val_probs = clf.predict_proba(Xva)[:, 1]
        test_probs = clf.predict_proba(Xte)[:, 1]
        model_label = "LogisticRegression(balanced)"
    elif model_type == "lgbm":
        try:
            import lightgbm as lgb  # type: ignore
        except OSError as exc:
            raise RuntimeError(
                "LightGBM is not available in the current environment (libomp missing). "
                "Install libomp / lightgbm or run with model_type='logreg'."
            ) from exc
        except ImportError as exc:
            raise RuntimeError(
                "LightGBM is not installed. Install it or choose model_type='logreg'."
            ) from exc
        pos_weight = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
        train_data = lgb.Dataset(Xtr, label=ytr)
        val_data = lgb.Dataset(Xva, label=yva)
        params = dict(
            objective="binary",
            metric="aucpr",
            boosting="gbdt",
            learning_rate=0.05,
            num_leaves=64,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=1,
            min_data_in_leaf=50,
            scale_pos_weight=pos_weight,
            verbosity=-1,
        )
        model = lgb.train(
            params,
            train_data,
            num_boost_round=300,
            valid_sets=[val_data],
            valid_names=["val"],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        val_probs = model.predict(Xva, num_iteration=model.best_iteration)
        test_probs = model.predict(Xte, num_iteration=model.best_iteration)
        model_label = "LightGBM(scale_pos_weight)"
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    tuned_threshold = threshold
    tuned_val_f1 = None
    if tune_threshold:
        tuned_threshold, tuned_val_f1 = _pick_threshold_from_pr_curve(yva, val_probs)

    report = {
        "model": model_label,
        "features": BASIC_FEATS,
        "val_days": val_days,
        "test_days": test_days,
        "threshold": tuned_threshold,
        "val_best_f1": tuned_val_f1,
        "metrics": {
            "val": _compute_metrics(yva, val_probs, tuned_threshold),
            "test": _compute_metrics(yte, test_probs, tuned_threshold),
        },
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
    }
    fname = "baseline_metrics.json" if model_type == "logreg" else "lgbm_metrics.json"
    with open(os.path.join(report_dir, fname), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return report

if __name__ == "__main__":
    import argparse, yaml
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", default="logreg", choices=["logreg", "lgbm"])
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-threshold-tuning", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    run_baseline(
        features_path=cfg["tabular"]["features_out"],
        report_dir=cfg["tabular"]["report_dir"],
        account_key=cfg["graph"]["account_key"],
        val_days=cfg["splits"]["val_days"],
        test_days=cfg["splits"].get("test_days", cfg["splits"]["val_days"]),
        target=cfg["tabular"]["target"],
        model_type=args.model,
        threshold=args.threshold,
        tune_threshold=not args.no_threshold_tuning,
    )
