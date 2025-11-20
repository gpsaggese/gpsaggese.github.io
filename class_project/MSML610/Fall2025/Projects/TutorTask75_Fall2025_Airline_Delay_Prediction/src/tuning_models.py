# src/tuning_models.py
"""
Bayesian hyperparameter tuning for XGBoost (native API) with time-aware CV using Optuna.

Workflow
--------
1) Load and coerce data; include DEPARTURE_DELAY (default true).
2) Time-aware holdout: first (1-eval_size) as train pool, last eval_size as final valid.
3) Bayesian optimization (Optuna) over user-defined bounds; objective = mean AP (aucpr) across CV folds.
4) Retrain on train pool with best hyperparams, early-stop on final valid; save artifacts.

Outputs (under models/ with --tag prefix)
-----------------------------------------
- {tag}_tune_trials.csv               (all trials: params + CV metrics)
- {tag}_model.joblib                  (best model)
- {tag}_metrics.json                  (final metrics + best hyperparams)
- {tag}_{pr,roc,loss_curve}.png
- {tag}_confusion_matrix.png          (counts + overall %)
- {tag}_feature_importance.csv/.png
"""

import os, json, argparse, warnings, math
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, confusion_matrix, classification_report, log_loss
)
from sklearn.model_selection import train_test_split

import xgboost as xgb
from xgboost.callback import EarlyStopping

import optuna

warnings.filterwarnings("ignore")
RANDOM_STATE = 610
np.random.seed(RANDOM_STATE)

BASE_CATEGORICAL: List[str] = [
    "AIRLINE","ORIGIN_AIRPORT","DESTINATION_AIRPORT","ORIGIN_STATE","DEST_STATE"
]
BASE_NUMERIC: List[str] = [
    "DEPARTURE_DELAY","AIR_TIME","DISTANCE",
    "ORIGIN_LAT","ORIGIN_LON","DEST_LAT","DEST_LON",
    "temp","rhum","prcp","snow","wspd","pres"
]

# ------------------ IO / utils ------------------

def read_parquet(path: str) -> pd.DataFrame:
    import pyarrow.parquet as pq
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

def time_holdout(df: pd.DataFrame, eval_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("FL_DATE")
    cut = int((1.0 - eval_size) * len(df))
    return df.iloc[:cut], df.iloc[cut:]

def make_dmatrix(X: pd.DataFrame, y: np.ndarray) -> xgb.DMatrix:
    try:
        return xgb.DMatrix(X, label=y, feature_names=list(X.columns), enable_categorical=True)
    except TypeError:
        Xc = X.copy()
        for c in Xc.select_dtypes(include="category").columns:
            Xc[c] = Xc[c].cat.codes.astype("int32")
        return xgb.DMatrix(Xc, label=y, feature_names=list(Xc.columns))

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

def cm_plot_counts_overall_pct(cm: np.ndarray, out_path: str, title="XGB Confusion Matrix (valid)"):
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

def feature_importance_df_from_booster(booster: xgb.Booster) -> pd.DataFrame:
    gain  = booster.get_score(importance_type="gain")
    cover = booster.get_score(importance_type="cover")
    weight= booster.get_score(importance_type="weight")
    keys = set(gain) | set(cover) | set(weight)
    rows = []
    for k in keys:
        rows.append({"feature": k, "gain": float(gain.get(k, 0.0)),
                     "cover": float(cover.get(k, 0.0)), "weight": float(weight.get(k, 0.0))})
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["gain","cover","weight"], ascending=False)
    return df

def plot_feature_importance(fi: pd.DataFrame, out_png: str, top=25):
    if fi.empty: return
    topk = fi.head(top)
    plt.figure(figsize=(8, max(4, 0.35*len(topk))))
    plt.barh(topk["feature"][::-1], topk["gain"][::-1])
    plt.xlabel("Gain"); plt.ylabel("Feature"); plt.title("XGB Feature Importance (gain)")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

def plot_loss_curve(history: Dict[str, Dict[str, List[float]]], out_path: str, title: str):
    tr = history.get("train", {}).get("logloss", [])
    va = history.get("validation_0", {}).get("logloss", [])
    if len(tr)==0 and len(va)==0: return
    plt.figure(figsize=(7,4))
    if tr: plt.plot(range(len(tr)), tr, label="train (logloss)")
    if va: plt.plot(range(len(va)), va, label="valid (logloss)")
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

def _clip01(p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    """Clip probabilities to (0,1) for numerical stability (sklearn >=1.5 removed eps kwarg)."""
    return np.clip(p, eps, 1.0 - eps)

# ------------------ time-aware CV splits ------------------

def time_kfold_indices(n: int, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    idx = np.arange(n)
    fold_sizes = np.full(n_splits, n // n_splits, dtype=int)
    fold_sizes[: n % n_splits] += 1
    boundaries = np.cumsum(fold_sizes)
    folds = []
    for f in range(n_splits - 1):  # last block has no future block to validate on
        tr = idx[:boundaries[f]]
        va = idx[boundaries[f]:boundaries[f+1]]
        if len(tr)>0 and len(va)>0:
            folds.append((tr, va))
    if not folds:
        cut = int(0.8 * n)
        folds = [(idx[:cut], idx[cut:])]
    return folds

# ------------------ Native train helper ------------------

def train_native(
    Xtr: pd.DataFrame, ytr: np.ndarray, Xva: pd.DataFrame, yva: np.ndarray,
    params: Dict[str, Any], n_rounds: int, early_stopping: int
):
    dtr = make_dmatrix(Xtr, ytr)
    dva = make_dmatrix(Xva, yva)
    watch = [(dtr, "train"), (dva, "validation_0")]
    evals_result: Dict[str, Dict[str, List[float]]] = {}
    es = EarlyStopping(rounds=int(early_stopping), save_best=True)
    bst = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=int(n_rounds),
        evals=watch,
        evals_result=evals_result,
        callbacks=[es],
        verbose_eval=False
    )
    last_idx = len(evals_result["validation_0"]["logloss"]) - 1
    best_iter = int(getattr(bst, "best_iteration", last_idx))

    def predict_at(dm: xgb.DMatrix, k: int) -> np.ndarray:
        try:    return bst.predict(dm, iteration_range=(0, k+1))
        except TypeError: return bst.predict(dm, ntree_limit=k+1)

    proba_va = predict_at(dva, best_iter)
    proba_tr = predict_at(dtr, best_iter)

    auc  = float(roc_auc_score(yva, proba_va))
    ap   = float(average_precision_score(yva, proba_va))
    # FIX: clip preds; sklearn>=1.5 doesn't accept eps kwarg in log_loss
    ll   = float(log_loss(yva, _clip01(proba_va)))

    return bst, {
        "best_iter": best_iter,
        "valid_auc": auc,
        "valid_ap": ap,
        "valid_logloss": ll,
        "history": evals_result,
        "proba_valid": proba_va,
        "proba_train": proba_tr
    }

# ------------------ Main ------------------

def main():
    ap = argparse.ArgumentParser(description="Bayesian tuning of XGBoost with time-aware CV (Optuna).")
    ap.add_argument("--in_path", type=str, default="data/processed/flights_with_weather.parquet")
    ap.add_argument("--out_dir", type=str, default="models")
    ap.add_argument("--split", type=str, choices=["time","random"], default="time")
    ap.add_argument("--eval_size", type=float, default=0.20)
    ap.add_argument("--use_departure_delay", type=str, default="true")
    ap.add_argument("--tag", type=str, default="tuned_all_features_bo")

    # CV / BO
    ap.add_argument("--cv_folds", type=int, default=5)
    ap.add_argument("--bo_trials", type=int, default=40)
    ap.add_argument("--bo_startup_trials", type=int, default=10)
    ap.add_argument("--bo_timeout", type=int, default=0, help="seconds; 0 = no timeout")

    # Boosting / early stop
    ap.add_argument("--n_rounds", type=int, default=1200)
    ap.add_argument("--early_stopping", type=int, default=100)

    # Search bounds (inclusive)
    ap.add_argument("--lr_low", type=float, default=0.03)
    ap.add_argument("--lr_high", type=float, default=0.2)
    ap.add_argument("--max_depth_low", type=int, default=5)
    ap.add_argument("--max_depth_high", type=int, default=9)
    ap.add_argument("--min_child_weight_low", type=int, default=1)
    ap.add_argument("--min_child_weight_high", type=int, default=8)
    ap.add_argument("--subsample_low", type=float, default=0.6)
    ap.add_argument("--subsample_high", type=float, default=1.0)
    ap.add_argument("--colsample_bytree_low", type=float, default=0.6)
    ap.add_argument("--colsample_bytree_high", type=float, default=1.0)
    ap.add_argument("--reg_alpha_low", type=float, default=1e-8)
    ap.add_argument("--reg_alpha_high", type=float, default=1.0)
    ap.add_argument("--reg_lambda_low", type=float, default=1e-2)
    ap.add_argument("--reg_lambda_high", type=float, default=10.0)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or "tuned_all_features_bo"

    use_dep = _bool_arg(args.use_departure_delay)

    # -------- Data
    numeric = [c for c in BASE_NUMERIC if use_dep or c != "DEPARTURE_DELAY"]
    categorical = BASE_CATEGORICAL.copy()

    df = coerce(read_parquet(args.in_path), numeric_cols=numeric)
    X_all = df[categorical + numeric + (["FL_DATE"] if "FL_DATE" in df.columns else [])].copy()
    y_all = df["is_delayed"].values

    if args.split == "time" and "FL_DATE" in X_all.columns:
        tmp = X_all.join(pd.Series(y_all, name="y"))
        train_pool, final_valid = time_holdout(tmp, args.eval_size)
        X_train = train_pool.drop(columns=["y","FL_DATE"], errors="ignore")
        y_train = train_pool["y"].values
        X_valid = final_valid.drop(columns=["y","FL_DATE"], errors="ignore")
        y_valid = final_valid["y"].values
    else:
        X = X_all.drop(columns=["FL_DATE"], errors="ignore")
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y_all, test_size=args.eval_size, random_state=RANDOM_STATE, stratify=y_all
        )

    # class imbalance
    pos = int((y_train == 1).sum()); neg = int((y_train == 0).sum())
    spw = max(1.0, neg / max(pos, 1))

    # fixed base params
    base_const = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss","auc","aucpr"],
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "seed": RANDOM_STATE,
        "enable_categorical": True,
        "max_cat_to_onehot": 1,
        "verbosity": 0,
        "scale_pos_weight": float(spw)
    }

    n = len(X_train)
    folds = time_kfold_indices(n, max(2, int(args.cv_folds)))

    # -------- Optuna objective: maximize mean AP across folds
    def objective(trial: optuna.Trial) -> float:
        params = {
            **base_const,
            "learning_rate": trial.suggest_float("learning_rate", args.lr_low, args.lr_high, log=False),
            "max_depth": trial.suggest_int("max_depth", args.max_depth_low, args.max_depth_high),
            "min_child_weight": trial.suggest_int("min_child_weight", args.min_child_weight_low, args.min_child_weight_high),
            "subsample": trial.suggest_float("subsample", args.subsample_low, args.subsample_high),
            "colsample_bytree": trial.suggest_float("colsample_bytree", args.colsample_bytree_low, args.colsample_bytree_high),
            "reg_alpha": trial.suggest_float("reg_alpha", args.reg_alpha_low, args.reg_alpha_high, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", args.reg_lambda_low, args.reg_lambda_high, log=True),
        }

        aps, aucs, lls = [], [], []
        for (tr_idx, va_idx) in folds:
            Xtr, Xva = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            ytr, yva = y_train[tr_idx], y_train[va_idx]
            _, out = train_native(Xtr, ytr, Xva, yva, params, n_rounds=args.n_rounds, early_stopping=args.early_stopping)
            aps.append(out["valid_ap"]); aucs.append(out["valid_auc"]); lls.append(out["valid_logloss"])

            trial.report(np.mean(aps), step=len(aps))
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_ap  = float(np.mean(aps))
        mean_auc = float(np.mean(aucs))
        mean_ll  = float(np.mean(lls))
        trial.set_user_attr("cv_auc", mean_auc)
        trial.set_user_attr("cv_logloss", mean_ll)
        return mean_ap

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE, n_startup_trials=int(args.bo_startup_trials))
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=int(args.bo_startup_trials))
    study   = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(
        objective,
        n_trials=int(args.bo_trials),
        timeout=None if int(args.bo_timeout) <= 0 else int(args.bo_timeout),
        show_progress_bar=False,
        gc_after_trial=True,
        n_jobs=1
    )

    # Collect trials to CSV
    records = []
    for t in study.trials:
        row = {
            "trial": t.number,
            "cv_ap": t.value,
            "cv_auc": t.user_attrs.get("cv_auc", np.nan),
            "cv_logloss": t.user_attrs.get("cv_logloss", np.nan),
        }
        row.update(t.params)
        records.append(row)
    df_trials = pd.DataFrame(records)
    df_trials.sort_values(by=["cv_ap","cv_auc","cv_logloss"], ascending=[False, False, True], inplace=True)
    csv_path = os.path.join(args.out_dir, f"{tag}_tune_trials.csv")
    df_trials.to_csv(csv_path, index=False)

    # Best params
    best_row = df_trials.iloc[0]
    best_params = {
        **base_const,
        "learning_rate": float(best_row["learning_rate"]),
        "max_depth": int(best_row["max_depth"]),
        "min_child_weight": int(best_row["min_child_weight"]),
        "subsample": float(best_row["subsample"]),
        "colsample_bytree": float(best_row["colsample_bytree"]),
        "reg_alpha": float(best_row["reg_alpha"]),
        "reg_lambda": float(best_row["reg_lambda"]),
    }
    print("\nBest hyperparams:", {k: best_params[k] for k in [
        "learning_rate","max_depth","min_child_weight","subsample","colsample_bytree","reg_alpha","reg_lambda"
    ]})

    # Final train on train pool; validate on holdout
    bst, final_out = train_native(
        X_train, y_train, X_valid, y_valid,
        params=best_params, n_rounds=args.n_rounds, early_stopping=args.early_stopping
    )

    best_iter = int(final_out["best_iter"])
    proba_v   = final_out["proba_valid"]
    proba_t   = final_out["proba_train"]

    th, f1_v, p_v, r_v = pick_threshold(y_valid, proba_v, maximize="f1")
    roc_v   = float(roc_auc_score(y_valid, proba_v))
    ap_v    = float(average_precision_score(y_valid, proba_v))
    preds_v = (proba_v >= th).astype(int)
    cm_v    = confusion_matrix(y_valid, preds_v)
    report_v= classification_report(y_valid, preds_v, output_dict=True)

    th_t, f1_t, p_t, r_t = pick_threshold(y_train, proba_t, maximize="f1")
    roc_t   = float(roc_auc_score(y_train, proba_t))
    ap_t    = float(average_precision_score(y_train, proba_t))

    prefix = f"{tag}"
    import joblib
    joblib.dump(bst, os.path.join(args.out_dir, f"{prefix}_model.joblib"))

    with open(os.path.join(args.out_dir, f"{prefix}_metrics.json"), "w") as f:
        json.dump({
            "mode": "native",
            "best_iteration": best_iter,
            "hyperparams": {k: best_params[k] for k in [
                "learning_rate","max_depth","min_child_weight","subsample","colsample_bytree","reg_alpha","reg_lambda"
            ]},
            "features_used": {"categorical": BASE_CATEGORICAL, "numeric": numeric},
            "roc_auc": roc_v, "pr_auc": ap_v,
            "threshold": th, "f1_at_threshold": f1_v,
            "precision_at_threshold": p_v, "recall_at_threshold": r_v,
            "train_snapshot": {
                "roc_auc": roc_t, "pr_auc": ap_t,
                "threshold": th_t, "f1_at_threshold": f1_t,
                "precision_at_threshold": p_t, "recall_at_threshold": r_t
            },
            "classification_report": report_v
        }, f, indent=2)

    plot_pr_roc(y_valid, proba_v, args.out_dir, prefix)
    cm_plot_counts_overall_pct(cm_v, os.path.join(args.out_dir, f"{prefix}_confusion_matrix.png"))
    plot_loss_curve(final_out["history"], os.path.join(args.out_dir, f"{prefix}_loss_curve.png"),
                    title=f"Loss vs Iterations — {prefix}")

    fi = feature_importance_df_from_booster(bst)
    if not fi.empty:
        fi.to_csv(os.path.join(args.out_dir, f"{prefix}_feature_importance.csv"), index=False)
        plot_feature_importance(fi, os.path.join(args.out_dir, f"{prefix}_feature_importance.png"), top=25)

    print(f"\n{prefix}: AUC={roc_v:.3f}  AP={ap_v:.3f}  F1={f1_v:.3f}  P={p_v:.3f}  R={r_v:.3f}  (best_iter={best_iter})")
    print(f"Saved: {csv_path} and model/metrics/plots under {args.out_dir}/")

if __name__ == "__main__":
    main()
