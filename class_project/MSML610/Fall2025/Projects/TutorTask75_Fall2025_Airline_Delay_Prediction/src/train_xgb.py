# src/train_xgb.py
"""
Stream per-iteration metrics while training XGBoost (native path) with tidy rows.

Artifacts (prefix uses --tag or _nodep when DEPARTURE_DELAY is excluded):
- models/{prefix}_model.joblib
- models/{prefix}_metrics.json
- models/{prefix}_roc.png, models/{prefix}_pr.png
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
    precision_recall_curve, roc_curve, confusion_matrix
)
from sklearn.model_selection import train_test_split

import xgboost as xgb
from xgboost import XGBClassifier
from xgboost.callback import TrainingCallback, EarlyStopping

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

def plot_curves(y_true, proba, out_dir, prefix):
    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_auc = roc_auc_score(y_true, proba)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, lw=2); plt.plot([0,1],[0,1],'--', lw=1)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={roc_auc:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{prefix}_roc.png")); plt.close()
    p, r, _ = precision_recall_curve(y_true, proba)
    pr_auc = average_precision_score(y_true, proba)
    plt.figure(figsize=(5,4))
    plt.plot(r, p, lw=2)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR (AP={pr_auc:.3f})")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{prefix}_pr.png")); plt.close()

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

def feature_importance_df_from_booster(booster: xgb.Booster) -> pd.DataFrame:
    gain  = booster.get_score(importance_type="gain")
    cover = booster.get_score(importance_type="cover")
    weight= booster.get_score(importance_type="weight")
    keys = set(gain) | set(cover) | set(weight)
    rows = []
    for k in keys:
        rows.append({
            "feature": k, "gain": float(gain.get(k, 0.0)),
            "cover": float(cover.get(k, 0.0)), "weight": float(weight.get(k, 0.0)),
        })
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

def cm_plot_counts_overall_pct(cm: np.ndarray, out_path: str, title="XGB Confusion Matrix (valid)"):
    total = cm.sum()
    annot = np.empty_like(cm).astype(object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = 100.0 * cm[i, j] / total if total > 0 else 0.0
            annot[i, j] = f"{cm[i, j]:,}\n({pct:.1f}%)"
    plt.figure(figsize=(4.9, 4.5))
    im = plt.imshow(cm, cmap="Purples")
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.xticks([0,1], ["On-time (0)","Delayed (1)"]); plt.yticks([0,1], ["On-time (0)","Delayed (1)"])
    thresh = cm.max()/2.0 if cm.max()>0 else 0.0
    for (i,j), label in np.ndenumerate(annot):
        color = "white" if cm[i,j] > thresh else "black"
        plt.text(j, i, label, ha="center", va="center", color=color, fontsize=9)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(out_path); plt.close()

def _bool_arg(x: str) -> bool:
    return str(x).strip().lower() in ("1","true","yes","y")

# ---------- Custom streaming logger callback ----------
class PeriodicLogger(TrainingCallback):
    """Prints tidy metrics every `period` rounds DURING training (native API).
       Also prints the final round if it wasn't a multiple of `period`.
    """
    def __init__(self, period: int, dtrain: xgb.DMatrix, y_train: np.ndarray,
                 dvalid: xgb.DMatrix, y_valid: np.ndarray):
        self.period   = max(1, int(period))
        self.dtrain   = dtrain
        self.y_train  = y_train
        self.dvalid   = dvalid
        self.y_valid  = y_valid
        self.last_printed = -1
        self.last_round   = None  # filled in after_training

    def _predict_at(self, model: xgb.Booster, k: int, dm: xgb.DMatrix) -> np.ndarray:
        try:
            return model.predict(dm, iteration_range=(0, k+1))
        except TypeError:
            return model.predict(dm, ntree_limit=k+1)

    def _maybe_print(self, model: xgb.Booster, epoch: int, evals_log: Dict[str, Dict[str, List[float]]], force=False):
        if (epoch % self.period != 0) and not force:
            return
        # pull current metrics
        v_ll  = evals_log.get("validation_0", {}).get("logloss", [])
        v_auc = evals_log.get("validation_0", {}).get("auc", [])
        v_ap  = evals_log.get("validation_0", {}).get("aucpr", [])
        t_ll  = evals_log.get("train", {}).get("logloss", [])
        t_auc = evals_log.get("train", {}).get("auc", [])
        t_ap  = evals_log.get("train", {}).get("aucpr", [])

        # compute F1 at epoch from predictions up to current trees
        pv = self._predict_at(model, epoch, self.dvalid)
        pt = self._predict_at(model, epoch, self.dtrain)
        _, f1_v, _, _ = pick_threshold(self.y_valid, pv, maximize="f1")
        _, f1_t, _, _ = pick_threshold(self.y_train, pt, maximize="f1")

        # safe indexing
        def safe(arr, i, default=np.nan):
            return arr[i] if i < len(arr) else default

        print(
            f"[{epoch}]  "
            f"valid_logloss={safe(v_ll, epoch, np.nan):.5f}  valid_F1={f1_v:.3f}  "
            f"valid_auc={safe(v_auc, epoch, np.nan):.3f}  valid_aucpr={safe(v_ap, epoch, np.nan):.3f}  "
            f"train_logloss={safe(t_ll, epoch, np.nan):.5f}  train_F1={f1_t:.3f}  "
            f"train_auc={safe(t_auc, epoch, np.nan):.3f}  train_aucpr={safe(t_ap, epoch, np.nan):.3f}"
        )
        self.last_printed = epoch

    # callbacks API
    def before_training(self, model):
        return model

    def after_iteration(self, model, epoch: int, evals_log: Dict[str, Dict[str, List[float]]]):
        self._maybe_print(model, epoch, evals_log, force=False)
        return False  # continue

    def after_training(self, model):
        # remember last round to ensure inclusive printing if needed
        try:
            self.last_round = int(getattr(model, "best_iteration", None))
        except Exception:
            self.last_round = None
        # If we never printed the last iteration seen in evals_log, force-print it.
        # Infer last epoch from any logged metric length.
        try:
            logs = model.evals_result()
            last_epoch = max(len(v) for m in logs.values() for v in m.values()) - 1
        except Exception:
            last_epoch = None
        if last_epoch is not None and last_epoch != self.last_printed:
            # fabricate an evals_log snapshot for safe access
            self._maybe_print(model, last_epoch, model.evals_result(), force=True)
        return model

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Train XGBoost arrival-delay classifier.")
    ap.add_argument("--in_path", type=str, default="data/processed/flights_with_weather.parquet")
    ap.add_argument("--out_dir", type=str, default="models")
    ap.add_argument("--eval_size", type=float, default=0.20)
    ap.add_argument("--sample_frac", type=float, default=1.0)
    ap.add_argument("--split", type=str, choices=["time","random"], default="time")
    ap.add_argument("--early_stopping", type=int, default=100)
    ap.add_argument("--n_estimators", type=int, default=1200)
    ap.add_argument("--log_period", type=int, default=25)
    ap.add_argument("--use_departure_delay", type=str, default="true",
                    help="true/false: include DEPARTURE_DELAY as a feature")
    ap.add_argument("--tag", type=str, default="", help="suffix for artifact filenames")
    ap.add_argument("--native", type=str, default="true",
                    help="true/false: use native xgboost.train (recommended)")
    ap.add_argument("--learning_rate", type=float, default=0.1,
                    help="Boosting learning rate (eta).")
    args = ap.parse_args()

    use_dep = _bool_arg(args.use_departure_delay)
    native  = _bool_arg(args.native)
    log_N   = max(1, int(args.log_period))

    numeric = [c for c in BASE_NUMERIC if use_dep or c != "DEPARTURE_DELAY"]
    categorical = BASE_CATEGORICAL.copy()

    suffix = f"_{args.tag}" if args.tag else ("" if use_dep else "_nodep")
    prefix = f"xgb{suffix}"

    os.makedirs(args.out_dir, exist_ok=True)

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

    pos = int((y_train == 1).sum()); neg = int((y_train == 0).sum())
    spw = max(1.0, neg / max(pos, 1))

    if native:
        # DMatrix with categorical enabled
        def make_dmatrix(X: pd.DataFrame, y: np.ndarray) -> xgb.DMatrix:
            try:
                return xgb.DMatrix(X, label=y, feature_names=list(X.columns), enable_categorical=True)
            except TypeError:
                Xc = X.copy()
                for c in Xc.select_dtypes(include="category").columns:
                    Xc[c] = Xc[c].cat.codes.astype("int32")
                return xgb.DMatrix(Xc, label=y, feature_names=list(Xc.columns))

        dtrain = make_dmatrix(X_train, y_train)
        dvalid = make_dmatrix(X_valid, y_valid)

        params: Dict[str, Any] = {
            "objective": "binary:logistic",
            "eval_metric": ["logloss","auc","aucpr"],
            "learning_rate": float(args.learning_rate),  # <-- from arg
            "max_depth": 7,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
            "seed": RANDOM_STATE,
            "scale_pos_weight": spw,
            "enable_categorical": True,
            "max_cat_to_onehot": 1,
            "verbosity": 0
        }

        watchlist = [(dtrain, "train"), (dvalid, "validation_0")]
        evals_result: Dict[str, Dict[str, List[float]]] = {}

        # Stream metrics DURING training
        logger_cb = PeriodicLogger(
            period=log_N, dtrain=dtrain, y_train=y_train,
            dvalid=dvalid, y_valid=y_valid
        )
        es_cb = EarlyStopping(rounds=int(args.early_stopping), save_best=True)

        bst = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=int(args.n_estimators),
            evals=watchlist,
            evals_result=evals_result,
            callbacks=[logger_cb, es_cb],   # <-- streaming happens here
            verbose_eval=False
        )

        # Final metrics @ best_iter
        def predict_at(dm: xgb.DMatrix, k: int) -> np.ndarray:
            try:    return bst.predict(dm, iteration_range=(0, k+1))
            except TypeError: return bst.predict(dm, ntree_limit=k+1)

        last_idx = len(evals_result["validation_0"]["logloss"]) - 1
        best_iter = int(getattr(bst, "best_iteration", last_idx))

        proba_valid = predict_at(dvalid, best_iter)
        proba_train = predict_at(dtrain, best_iter)

        th, f1_v, p_v, r_v = pick_threshold(y_valid, proba_valid, maximize="f1")
        roc_v   = float(roc_auc_score(y_valid, proba_valid))
        prauc_v = float(average_precision_score(y_valid, proba_valid))
        preds_v = (proba_valid >= th).astype(int)
        cm_v    = confusion_matrix(y_valid, preds_v)
        report_v= classification_report(y_valid, preds_v, output_dict=True)

        th_tr, f1_tr, p_tr, r_tr = pick_threshold(y_train, proba_train, maximize="f1")
        roc_tr   = float(roc_auc_score(y_train, proba_train))
        prauc_tr = float(average_precision_score(y_train, proba_train))

        print(
            f"[best_iter={best_iter}]  final_valid_auc={roc_v:.3f}  final_valid_aucpr={prauc_v:.3f}  "
            f"best_F1={f1_v:.3f}  th={th:.3f}  P={p_v:.3f}  R={r_v:.3f}"
        )

        # Persist
        joblib.dump(bst, os.path.join(args.out_dir, f"{prefix}_model.joblib"))
        with open(os.path.join(args.out_dir, f"{prefix}_metrics.json"), "w") as f:
            json.dump({
                "mode": "native",
                "best_iteration": best_iter,
                "learning_rate": float(args.learning_rate),  # save for reproducibility
                "features_used": {"categorical": BASE_CATEGORICAL, "numeric": numeric},
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

        # Plots
        plot_curves(y_valid, proba_valid, args.out_dir, prefix)
        cm_plot_counts_overall_pct(cm_v, os.path.join(args.out_dir, f"{prefix}_confusion_matrix.png"))
        fi = feature_importance_df_from_booster(bst)
        if not fi.empty:
            fi.to_csv(os.path.join(args.out_dir, f"{prefix}_feature_importance.csv"), index=False)
            plot_feature_importance(fi, os.path.join(args.out_dir, f"{prefix}_feature_importance.png"), top=25)

        print(f"{prefix}: ROC-AUC={roc_v:.3f}  PR-AUC={prauc_v:.3f}  F1={f1_v:.3f}  P={p_v:.3f}  R={r_v:.3f}")
        return

    # -------- Sklearn wrapper fallback (no streaming improvements here) --------
    clf = XGBClassifier(
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),  # <-- from arg
        max_depth=7,
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, reg_alpha=0.0,
        tree_method="hist", random_state=RANDOM_STATE, n_jobs=-1,
        scale_pos_weight=spw, eval_metric=["logloss","auc","aucpr"],
        enable_categorical=True, max_cat_to_onehot=1, verbosity=0
    )
    clf.set_params(early_stopping_rounds=int(args.early_stopping))
    clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    # Minimal summary lines from evals_result if available
    try:
        er = clf.evals_result()
        val_ll, val_auc, val_ap = er["validation_0"]["logloss"], er["validation_0"]["auc"], er["validation_0"]["aucpr"]
        last_idx = len(val_ll)-1
        for i in range(0, last_idx+1, log_N):
            print(f"[{i}]  valid_logloss={val_ll[i]:.5f}  valid_F1=NA  valid_auc={val_auc[i]:.3f}  valid_aucpr={val_ap[i]:.3f}  "
                  f"train_logloss=NA  train_F1=NA  train_auc=NA  train_aucpr=NA")
        if last_idx % log_N != 0:
            i = last_idx
            print(f"[{i}]  valid_logloss={val_ll[i]:.5f}  valid_F1=NA  valid_auc={val_auc[i]:.3f}  valid_aucpr={val_ap[i]:.3f}  "
                  f"train_logloss=NA  train_F1=NA  train_auc=NA  train_aucpr=NA")
    except Exception:
        pass

    proba_valid = clf.predict_proba(X_valid)[:, 1]
    th, f1_v, p_v, r_v = pick_threshold(y_valid, proba_valid, maximize="f1")
    roc_v   = float(roc_auc_score(y_valid, proba_valid))
    prauc_v = float(average_precision_score(y_valid, proba_valid))
    preds_v = (proba_valid >= th).astype(int)
    cm_v    = confusion_matrix(y_valid, preds_v)
    report_v= classification_report(y_valid, preds_v, output_dict=True)

    proba_tr = clf.predict_proba(X_train)[:, 1]
    th_tr, f1_tr, p_tr, r_tr = pick_threshold(y_train, proba_tr, maximize="f1")
    roc_tr   = float(roc_auc_score(y_train, proba_tr))
    prauc_tr = float(average_precision_score(y_train, proba_tr))

    best_iter = int(getattr(clf, "best_iteration", len(er["validation_0"]["auc"])-1 if 'er' in locals() else args.n_estimators-1))
    print(f"[best_iter={best_iter}]  final_valid_auc={roc_v:.3f}  final_valid_aucpr={prauc_v:.3f}  best_F1={f1_v:.3f}  th={th:.3f}  P={p_v:.3f}  R={r_v:.3f}")

    joblib.dump(clf, os.path.join(args.out_dir, f"{prefix}_model.joblib"))
    with open(os.path.join(args.out_dir, f"{prefix}_metrics.json"), "w") as f:
        json.dump({
            "mode": "sklearn",
            "best_iteration": best_iter,
            "learning_rate": float(args.learning_rate),  # save for reproducibility
            "features_used": {"categorical": BASE_CATEGORICAL, "numeric": numeric},
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

    plot_curves(y_valid, proba_valid, args.out_dir, prefix)
    cm_plot_counts_overall_pct(cm_v, os.path.join(args.out_dir, f"{prefix}_confusion_matrix.png"))
    fi = feature_importance_df_from_booster(clf.get_booster())
    if not fi.empty:
        fi.to_csv(os.path.join(args.out_dir, f"{prefix}_feature_importance.csv"), index=False)
        plot_feature_importance(fi, os.path.join(args.out_dir, f"{prefix}_feature_importance.png"), top=25)

    print(f"{prefix}: ROC-AUC={roc_v:.3f}  PR-AUC={prauc_v:.3f}  F1={f1_v:.3f}  P={p_v:.3f}  R={r_v:.3f}")

if __name__ == "__main__":
    main()
