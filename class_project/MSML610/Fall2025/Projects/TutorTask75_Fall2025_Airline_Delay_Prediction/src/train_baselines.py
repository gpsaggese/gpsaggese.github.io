# src/train_baselines.py
"""
Train LightGBM and/or CatBoost baselines on the airline-delay dataset with a time-aware split.

Artifacts (one set per model; prefix is `<model>_{tag or _nodep}`):
- models/{prefix}_model.joblib
- models/{prefix}_metrics.json
- models/{prefix}_roc.png, models/{prefix}_pr.png, models/{prefix}_loss_curve.png
- models/{prefix}_confusion_matrix.png  (counts + overall %)
- models/{prefix}_feature_importance.csv, {prefix}_feature_importance.png
"""

import os, json, argparse, warnings
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import joblib
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, confusion_matrix, log_loss
)
from sklearn.model_selection import train_test_split

import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings("ignore")
RANDOM_STATE = 610

BASE_CATEGORICAL: List[str] = [
    "AIRLINE","ORIGIN_AIRPORT","DESTINATION_AIRPORT","ORIGIN_STATE","DEST_STATE"
]
BASE_NUMERIC: List[str] = [
    "DEPARTURE_DELAY","AIR_TIME","DISTANCE",
    "ORIGIN_LAT","ORIGIN_LON","DEST_LAT","DEST_LON",
    "temp","rhum","prcp","snow","wspd","pres"
]

# ---------- IO / utils ----------
def read_parquet(path: str) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()

def coerce(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    for c in numeric_cols + ["is_delayed"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "is_delayed" in df.columns:
        df["is_delayed"] = df["is_delayed"].fillna(0).astype(int)
    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].astype("float32")
    if "FL_DATE" in df.columns:
        df["FL_DATE"] = pd.to_datetime(df["FL_DATE"], errors="coerce")
    for c in BASE_CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df

def time_split(df: pd.DataFrame, eval_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("FL_DATE")
    cut = int((1.0 - eval_size) * len(df))
    return df.iloc[:cut], df.iloc[cut:]

def plot_pr_roc(y_true, proba, out_dir, prefix):
    # PR
    p, r, _ = precision_recall_curve(y_true, proba)
    ap = average_precision_score(y_true, proba)
    plt.figure(figsize=(5,4))
    plt.plot(r, p, lw=2)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR (AP={ap:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{prefix}_pr.png")); plt.close()
    # ROC
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, lw=2); plt.plot([0,1],[0,1],'--', lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={auc:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{prefix}_roc.png")); plt.close()

def cm_plot_counts_overall_pct(cm: np.ndarray, out_path: str, title="Confusion Matrix (valid)"):
    total = cm.sum()
    annot = np.empty_like(cm).astype(object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = 100.0 * cm[i, j] / total if total > 0 else 0.0
            annot[i, j] = f"{cm[i, j]:,}\n({pct:.1f}%)"
    plt.figure(figsize=(5.0, 4.6))
    im = plt.imshow(cm, cmap="Purples")
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.xticks([0,1], ["On-time (0)","Delayed (1)"]); plt.yticks([0,1], ["On-time (0)","Delayed (1)"])
    thresh = cm.max()/2.0 if cm.max()>0 else 0.0
    for (i,j), label in np.ndenumerate(annot):
        color = "white" if cm[i,j] > thresh else "black"
        plt.text(j, i, label, ha="center", va="center", color=color, fontsize=9)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def plot_loss_curve(train_loss: List[float], valid_loss: List[float], out_path: str, title: str):
    plt.figure(figsize=(7,4))
    if train_loss:
        plt.plot(range(len(train_loss)), train_loss, label="train (logloss)")
    if valid_loss:
        plt.plot(range(len(valid_loss)), valid_loss, label="valid (logloss)")
    plt.xlabel("Boosting round"); plt.ylabel("Logloss"); plt.title(title)
    plt.legend(); plt.tight_layout(); plt.savefig(out_path); plt.close()

def pick_threshold(y_true, proba, maximize="f1"):
    p, r, th = precision_recall_curve(y_true, proba)
    th = np.append(th, 1.0)
    f1 = 2 * (p*r) / np.maximum(p + r, 1e-12)
    if maximize == "recall":
        idx = int(np.argmax(r))
    elif maximize == "precision":
        idx = int(np.argmax(p))
    else:
        idx = int(np.nanargmax(f1))
    return float(th[idx]), float(f1[idx]), float(p[idx]), float(r[idx])

def _bool_arg(x: str) -> bool:
    return str(x).strip().lower() in ("1","true","yes","y")

def summarize_every_N_inclusive(
    predict_fn,  # callable(num_iteration)-> (proba_valid, proba_train)
    y_valid: np.ndarray, y_train: np.ndarray,
    train_loss: List[float], valid_loss: List[float],
    N: int
):
    last = len(valid_loss) - 1 if valid_loss else 0
    checkpoints = list(range(0, last + 1, max(1, N)))
    if checkpoints and checkpoints[-1] != last:
        checkpoints.append(last)
    for k in checkpoints:
        pv, pt = predict_fn(k)
        _, f1_v, _, _ = pick_threshold(y_valid, pv, maximize="f1")
        _, f1_t, _, _ = pick_threshold(y_train, pt, maximize="f1")
        llv = valid_loss[k] if k < len(valid_loss) else np.nan
        llt = train_loss[k] if k < len(train_loss) else np.nan
        auc_v = roc_auc_score(y_valid, pv); ap_v = average_precision_score(y_valid, pv)
        auc_t = roc_auc_score(y_train, pt); ap_t = average_precision_score(y_train, pt)
        print(
          f"[{k}]  "
          f"valid_logloss={llv:.5f}  valid_F1={f1_v:.3f}  valid_auc={auc_v:.3f}  valid_aucpr={ap_v:.3f}  "
          f"train_logloss={llt:.5f}  train_F1={f1_t:.3f}  train_auc={auc_t:.3f}  train_aucpr={ap_t:.3f}"
        )

# ---------- LightGBM ----------
def train_lgbm(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_valid: pd.DataFrame, y_valid: np.ndarray,
    categorical_cols: List[str],
    out_dir: str, prefix: str,
    learning_rate: float, n_estimators: int, max_depth: int,
    early_stopping: int, log_period: int
):
    dtrain = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols, free_raw_data=False)
    dvalid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical_cols, reference=dtrain, free_raw_data=False)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss","auc","average_precision"],
        "learning_rate": float(learning_rate),
        "max_depth": int(max_depth),
        "num_leaves": int(2**max(1, max_depth)) if max_depth > 0 else 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 20,
        "seed": RANDOM_STATE,
        "verbose": -1,
    }

    evals_result: Dict[str, Dict[str, List[float]]] = {}

    # Callbacks API (no verbose_eval kwarg in recent LightGBM)
    callbacks = [
        lgb.early_stopping(stopping_rounds=int(early_stopping), verbose=False),
        lgb.record_evaluation(evals_result),
        lgb.log_evaluation(period=0 if log_period <= 0 else log_period),
    ]

    bst = lgb.train(
        params,
        dtrain,
        num_boost_round=int(n_estimators),
        valid_sets=[dtrain, dvalid],
        valid_names=["train","validation_0"],
        callbacks=callbacks,
    )

    tr_ll = evals_result.get("train", {}).get("binary_logloss", [])
    va_ll = evals_result.get("validation_0", {}).get("binary_logloss", [])
    best_iter = int(bst.best_iteration or len(va_ll))

    def predict_at(k: int):
        pv = bst.predict(X_valid, num_iteration=k+1)
        pt = bst.predict(X_train, num_iteration=k+1)
        return pv, pt

    summarize_every_N_inclusive(predict_at, y_valid, y_train, tr_ll, va_ll, log_period)

    proba_valid = bst.predict(X_valid, num_iteration=best_iter)
    proba_train = bst.predict(X_train, num_iteration=best_iter)

    th, f1_v, p_v, r_v = pick_threshold(y_valid, proba_valid, maximize="f1")
    roc_v   = float(roc_auc_score(y_valid, proba_valid))
    prauc_v = float(average_precision_score(y_valid, proba_valid))
    preds_v = (proba_valid >= th).astype(int)
    cm_v    = confusion_matrix(y_valid, preds_v)
    report_v= classification_report(y_valid, preds_v, output_dict=True)

    th_tr, f1_tr, p_tr, r_tr = pick_threshold(y_train, proba_train, maximize="f1")
    roc_tr   = float(roc_auc_score(y_train, proba_train))
    prauc_tr = float(average_precision_score(y_train, proba_train))

    joblib.dump(bst, os.path.join(out_dir, f"{prefix}_model.joblib"))
    with open(os.path.join(out_dir, f"{prefix}_metrics.json"), "w") as f:
        json.dump({
            "mode": "lightgbm",
            "best_iteration": best_iter,
            "features_used": {"categorical": BASE_CATEGORICAL, "numeric": [c for c in BASE_NUMERIC if c in X_train.columns]},
            "roc_auc": roc_v, "pr_auc": prauc_v,
            "threshold": th, "f1_at_threshold": f1_v,
            "precision_at_threshold": p_v, "recall_at_threshold": r_v,
            "train_snapshot": {
                "roc_auc": roc_tr, "pr_auc": prauc_tr,
                "threshold": th_tr, "f1_at_threshold": f1_tr,
                "precision_at_threshold": p_tr, "recall_at_threshold": r_tr
            },
            "classification_report": report_v
        }, f, indent=2)

    plot_pr_roc(y_valid, proba_valid, out_dir, prefix)
    cm_plot_counts_overall_pct(cm_v, os.path.join(out_dir, f"{prefix}_confusion_matrix.png"))
    plot_loss_curve(tr_ll, va_ll, os.path.join(out_dir, f"{prefix}_loss_curve.png"), f"Loss vs Iterations — {prefix}")

    try:
        gain = bst.feature_importance(importance_type="gain")
        names = bst.feature_name()
        fi = pd.DataFrame({"feature": names, "gain": gain}).sort_values("gain", ascending=False)
        fi.to_csv(os.path.join(out_dir, f"{prefix}_feature_importance.csv"), index=False)
        top = fi.head(25)
        plt.figure(figsize=(8, max(4, 0.35*len(top))))
        plt.barh(top["feature"][::-1], top["gain"][::-1])
        plt.xlabel("Gain"); plt.ylabel("Feature"); plt.title("LGBM Feature Importance (gain)")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{prefix}_feature_importance.png")); plt.close()
    except Exception:
        pass

    print(f"[best_iter={best_iter}]  {prefix}: ROC-AUC={roc_v:.3f}  PR-AUC={prauc_v:.3f}  F1={f1_v:.3f}  P={p_v:.3f}  R={r_v:.3f}")


# ---------- CatBoost ----------
    

def train_cat(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_valid: pd.DataFrame, y_valid: np.ndarray,
    categorical_cols: List[str],
    out_dir: str, prefix: str,
    learning_rate: float, n_estimators: int, depth: int,
    early_stopping: int, log_period: int
):
    cat_idx = [X_train.columns.get_loc(c) for c in categorical_cols if c in X_train.columns]

    model = CatBoostClassifier(
        iterations=int(n_estimators),
        learning_rate=float(learning_rate),
        depth=int(depth),
        loss_function="Logloss",
        eval_metric="AUC",
        od_type="Iter",
        od_wait=int(early_stopping),
        random_seed=RANDOM_STATE,
        verbose=False
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_idx)

    model.fit(train_pool, eval_set=valid_pool, use_best_model=True, verbose=False)

    hist = model.get_evals_result()
    tr_ll = hist.get("learn", {}).get("Logloss", [])
    va_ll = hist.get("validation", {}).get("Logloss", [])
    best_iter = int(model.get_best_iteration() or len(va_ll)-1)

    def predict_at(k: int):
        pv = model.predict_proba(valid_pool, ntree_end=k+1)[:, 1]
        pt = model.predict_proba(train_pool, ntree_end=k+1)[:, 1]
        return pv, pt
    summarize_every_N_inclusive(predict_at, y_valid, y_train, tr_ll, va_ll, log_period)

    proba_valid = model.predict_proba(valid_pool)[:, 1]
    proba_train = model.predict_proba(train_pool)[:, 1]

    th, f1_v, p_v, r_v = pick_threshold(y_valid, proba_valid, maximize="f1")
    roc_v   = float(roc_auc_score(y_valid, proba_valid))
    prauc_v = float(average_precision_score(y_valid, proba_valid))
    preds_v = (proba_valid >= th).astype(int)
    cm_v    = confusion_matrix(y_valid, preds_v)
    report_v= classification_report(y_valid, preds_v, output_dict=True)

    th_tr, f1_tr, p_tr, r_tr = pick_threshold(y_train, proba_train, maximize="f1")
    roc_tr   = float(roc_auc_score(y_train, proba_train))
    prauc_tr = float(average_precision_score(y_train, proba_train))

    joblib.dump(model, os.path.join(out_dir, f"{prefix}_model.joblib"))
    with open(os.path.join(out_dir, f"{prefix}_metrics.json"), "w") as f:
        json.dump({
            "mode": "catboost",
            "best_iteration": best_iter,
            "features_used": {"categorical": BASE_CATEGORICAL, "numeric": [c for c in BASE_NUMERIC if c in X_train.columns]},
            "roc_auc": roc_v, "pr_auc": prauc_v,
            "threshold": th, "f1_at_threshold": f1_v,
            "precision_at_threshold": p_v, "recall_at_threshold": r_v,
            "train_snapshot": {
                "roc_auc": roc_tr, "pr_auc": prauc_tr,
                "threshold": th_tr, "f1_at_threshold": f1_tr,
                "precision_at_threshold": p_tr, "recall_at_threshold": r_tr
            },
            "classification_report": report_v
        }, f, indent=2)

    plot_pr_roc(y_valid, proba_valid, out_dir, prefix)
    cm_plot_counts_overall_pct(cm_v, os.path.join(out_dir, f"{prefix}_confusion_matrix.png"))
    plot_loss_curve(tr_ll, va_ll, os.path.join(out_dir, f"{prefix}_loss_curve.png"), f"Loss vs Iterations — {prefix}")

    try:
        fi_vals = model.get_feature_importance(train_pool, type="PredictionValuesChange")
        fi = pd.DataFrame({"feature": X_train.columns, "gain": fi_vals}).sort_values("gain", ascending=False)
        fi.to_csv(os.path.join(out_dir, f"{prefix}_feature_importance.csv"), index=False)
        top = fi.head(25)
        plt.figure(figsize=(8, max(4, 0.35*len(top))))
        plt.barh(top["feature"][::-1], top["gain"][::-1])
        plt.xlabel("Gain"); plt.ylabel("Feature"); plt.title("CatBoost Feature Importance (PredictionValuesChange)")
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{prefix}_feature_importance.png")); plt.close()
    except Exception:
        pass

    print(f"[best_iter={best_iter}]  {prefix}: ROC-AUC={roc_v:.3f}  PR-AUC={prauc_v:.3f}  F1={f1_v:.3f}  P={p_v:.3f}  R={r_v:.3f}")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Train LightGBM / CatBoost baselines (time-aware split).")
    ap.add_argument("--in_path", type=str, default="data/processed/flights_with_weather.parquet")
    ap.add_argument("--out_dir", type=str, default="models")
    ap.add_argument("--eval_size", type=float, default=0.20)
    ap.add_argument("--sample_frac", type=float, default=1.0)
    ap.add_argument("--split", type=str, choices=["time","random"], default="time")
    ap.add_argument("--use_departure_delay", type=str, default="true",
                    help="true/false: include DEPARTURE_DELAY as a feature")
    ap.add_argument("--model", type=str, choices=["lgbm","cat","all"], default="all")
    ap.add_argument("--tag", type=str, default="", help="suffix for artifact filenames (e.g., 'baseline')")
    ap.add_argument("--log_period", type=int, default=25)

    # Common training knobs
    ap.add_argument("--n_estimators", type=int, default=1000)
    ap.add_argument("--learning_rate", type=float, default=0.1)
    ap.add_argument("--early_stopping", type=int, default=100)

    # Model-specific knobs (kept minimal for baselines)
    ap.add_argument("--lgbm_max_depth", type=int, default=7)
    ap.add_argument("--cat_depth", type=int, default=7)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    use_dep = _bool_arg(args.use_departure_delay)
    log_N   = max(1, int(args.log_period))

    numeric = [c for c in BASE_NUMERIC if use_dep or c != "DEPARTURE_DELAY"]
    categorical = [c for c in BASE_CATEGORICAL if c in BASE_CATEGORICAL]

    suffix = f"_{args.tag}" if args.tag else ("" if use_dep else "_nodep")

    # -------- Data
    df = coerce(read_parquet(args.in_path), numeric_cols=numeric)
    if args.sample_frac < 1.0:
        df = df.sample(frac=args.sample_frac, random_state=RANDOM_STATE)

    X_all = df[categorical + numeric + (["FL_DATE"] if "FL_DATE" in df.columns else [])].copy()
    y_all = df["is_delayed"].values

    if args.split == "time" and "FL_DATE" in X_all.columns:
        tmp = X_all.join(pd.Series(y_all, name="y"))
        train_df, valid_df = time_split(tmp, args.eval_size)
        X_train = train_df.drop(columns=["y","FL_DATE"], errors="ignore")
        y_train = train_df["y"].values
        X_valid = valid_df.drop(columns=["y","FL_DATE"], errors="ignore")
        y_valid = valid_df["y"].values
    else:
        X = X_all.drop(columns=["FL_DATE"], errors="ignore")
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y_all, test_size=args.eval_size, random_state=RANDOM_STATE, stratify=y_all
        )

    if args.model in ("lgbm","all"):
        print("Training Light Gradient Boosted Model (GMB) Now... \n")
        prefix = f"lgbm{suffix}"
        train_lgbm(
            X_train, y_train, X_valid, y_valid,
            categorical_cols=[c for c in categorical if c in X_train.columns],
            out_dir=args.out_dir, prefix=prefix,
            learning_rate=args.learning_rate, n_estimators=args.n_estimators, max_depth=args.lgbm_max_depth,
            early_stopping=args.early_stopping, log_period=log_N
        )

    if args.model in ("cat","all"):
        print("Training CatBoost Model Now... \n")
        prefix = f"cat{suffix}"
        train_cat(
            X_train, y_train, X_valid, y_valid,
            categorical_cols=[c for c in categorical if c in X_train.columns],
            out_dir=args.out_dir, prefix=prefix,
            learning_rate=args.learning_rate, n_estimators=args.n_estimators, depth=args.cat_depth,
            early_stopping=args.early_stopping, log_period=log_N
        )

if __name__ == "__main__":
    main()
