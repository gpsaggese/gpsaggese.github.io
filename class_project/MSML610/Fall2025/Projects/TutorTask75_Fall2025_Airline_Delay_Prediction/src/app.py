# Ran in 05_running_app.ipynb which is located in the Notebooks folder
# app.py
# Streamlit app for Airline Delay Prediction — compare XGBoost (tuned), LightGBM, and CatBoost
# Uses artifacts saved by src/tuning_models.py and src/train_baselines.py

import os
import json
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------
# Project-wide constants
# -----------------------
RANDOM_STATE = 610

BASE_CATEGORICAL = [
    "AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "ORIGIN_STATE", "DEST_STATE"
]
BASE_NUMERIC = [
    "DEPARTURE_DELAY", "AIR_TIME", "DISTANCE",
    "ORIGIN_LAT", "ORIGIN_LON", "DEST_LAT", "DEST_LON",
    "temp", "rhum", "prcp", "snow", "wspd", "pres"
]

ARTIFACTS = {
    # tag -> file prefixes under models/
    "xgb_tuned": {
        "prefix": "tuned_all_features_bo",
        "model": "models/tuned_all_features_bo_model.joblib",
        "metrics": "models/tuned_all_features_bo_metrics.json",
        "plots": {
            "PR": "models/tuned_all_features_bo_pr.png",
            "ROC": "models/tuned_all_features_bo_roc.png",
            "Loss": "models/tuned_all_features_bo_loss_curve.png",
            "CM": "models/tuned_all_features_bo_confusion_matrix.png",
            "FI": "models/tuned_all_features_bo_feature_importance.png",
        },
    },
    "lgbm": {
        "prefix": "lgbm_all_features",
        "model": "models/lgbm_all_features_model.joblib",
        "metrics": "models/lgbm_all_features_metrics.json",
        "plots": {
            "PR": "models/lgbm_all_features_pr.png",
            "ROC": "models/lgbm_all_features_roc.png",
            "Loss": "models/lgbm_all_features_loss_curve.png",
            "CM": "models/lgbm_all_features_confusion_matrix.png",
            "FI": "models/lgbm_all_features_feature_importance.png",
        },
    },
    "cat": {
        "prefix": "cat_all_features",
        "model": "models/cat_all_features_model.joblib",
        "metrics": "models/cat_all_features_metrics.json",
        "plots": {
            "PR": "models/cat_all_features_pr.png",
            "ROC": "models/cat_all_features_roc.png",
            "Loss": "models/cat_all_features_loss_curve.png",
            "CM": "models/cat_all_features_confusion_matrix.png",
            "FI": "models/cat_all_features_feature_importance.png",
        },
    },
}

# -----------------------
# Small utils
# -----------------------

def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same dtype coercions as training."""
    df = df.copy()
    # numerics to float32
    for c in BASE_NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    # categoricals
    for c in BASE_CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].astype("category")
    # target (if present)
    if "is_delayed" in df.columns:
        df["is_delayed"] = pd.to_numeric(df["is_delayed"], errors="coerce").fillna(0).astype(int)
    return df

@st.cache_data(show_spinner=False)
def load_model(tag: str):
    path = ARTIFACTS[tag]["model"]
    if not os.path.exists(path):
        return None
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def load_metrics(tag: str):
    path = ARTIFACTS[tag]["metrics"]
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)

def predict_proba_for(tag: str, model, X: pd.DataFrame) -> np.ndarray:
    """Library-aware probability prediction. Returns p(delay=1)."""
    if model is None:
        return np.full(len(X), np.nan)

    # Ensure schema/dtypes
    Xc = _coerce_schema(X)
    # XGBoost native Booster?
    try:
        import xgboost as xgb
        if tag == "xgb_tuned":
            # DMatrix with categorical support (codes if needed)
            try:
                dmat = xgb.DMatrix(Xc, feature_names=list(Xc.columns), enable_categorical=True)
            except TypeError:
                Xtmp = Xc.copy()
                for c in Xtmp.select_dtypes(include="category").columns:
                    Xtmp[c] = Xtmp[c].cat.codes.astype("int32")
                dmat = xgb.DMatrix(Xtmp, feature_names=list(Xtmp.columns))
            return model.predict(dmat)
    except Exception:
        pass

    # sklearn-like (LGBM/CatBoost sklearn wrappers saved as joblib)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xc)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        # CatBoost sometimes exposes predict_proba similarly
        return np.ravel(proba)

    # CatBoost native model?
    if hasattr(model, "predict"):
        preds = model.predict(Xc, prediction_type="Probability")
        if isinstance(preds, np.ndarray):
            return preds[:, -1] if preds.ndim == 2 else np.ravel(preds)
        if isinstance(preds, list):
            return np.array(preds, dtype="float32")
    return np.full(len(X), np.nan)

def pick_threshold_from_metrics(m: dict, default: float = 0.5) -> float:
    # our metrics JSONs store 'threshold' for the chosen operating point
    th = m.get("threshold")
    if isinstance(th, (int, float)):
        return float(th)
    return float(default)

# -----------------------
# Streamlit UI
# -----------------------

st.set_page_config(page_title="Airline Delay Prediction — GBDT Comparison", layout="wide")

st.title("Airline Delay Prediction")
st.caption("Compare tuned XGBoost vs LightGBM vs CatBoost. Score new flights and inspect artifacts.")

with st.sidebar:
    st.header("Model selection")
    tag = st.selectbox("Choose a model", ["cat", "lgbm", "xgb_tuned"], index=0)
    show_all_cards = st.checkbox("Show all model cards", value=True)
    st.divider()
    st.header("Artifacts directory")
    st.write("`models/` with saved joblib + metrics + plots is expected.")

# Load selected model + metrics
model = load_model(tag)
metrics = load_metrics(tag)
threshold = pick_threshold_from_metrics(metrics, default=0.5)

# -----------------------
# Model cards (metrics)
# -----------------------
def _metric_row(tag_key: str):
    m = load_metrics(tag_key)
    name = {"cat": "CatBoost", "lgbm": "LightGBM", "xgb_tuned": "XGBoost (tuned)"}[tag_key]
    auc = m.get("roc_auc")
    ap = m.get("pr_auc")
    f1 = m.get("f1_at_threshold")
    p = m.get("precision_at_threshold")
    r = m.get("recall_at_threshold")
    ll = m.get("log_loss") or m.get("valid_logloss")  # XGB file used "logloss"; baselines saved "valid_logloss"
    best_it = m.get("best_iteration")
    th = m.get("threshold")

    # normalize log loss key names
    if ll is None:
        # some JSONs we wrote use nested names; try a few places
        ll = m.get("valid_logloss") or m.get("train_snapshot", {}).get("valid_logloss")

    return {
        "model": name,
        "AUC": auc,
        "AP": ap,
        "F1": f1,
        "Precision": p,
        "Recall": r,
        "log_loss": ll,
        "best_iter": best_it,
        "threshold": th,
    }

st.subheader("Model leaderboard (validation hold-out)")
rows = []
for k in ["cat", "lgbm", "xgb_tuned"]:
    try:
        rows.append(_metric_row(k))
    except Exception:
        pass
df_cards = pd.DataFrame(rows)
if not df_cards.empty:
    st.dataframe(df_cards, use_container_width=True)
else:
    st.info("No metrics JSONs found under models/. Train the models first to populate this table.")


# -----------------------
# Artifact preview
# -----------------------
st.subheader(f"Artifacts — { {'cat':'CatBoost','lgbm':'LightGBM','xgb_tuned':'XGBoost (tuned)'}[tag] }")
colA, colB, colC = st.columns([1, 1, 1])
plots = ARTIFACTS[tag]["plots"]

with colA:
    if os.path.exists(plots["PR"]):
        st.image(plots["PR"], caption="Precision–Recall", use_column_width=True)
    if os.path.exists(plots["ROC"]):
        st.image(plots["ROC"], caption="ROC", use_column_width=True)

with colB:
    if os.path.exists(plots["Loss"]):
        st.image(plots["Loss"], caption="Loss curve", use_column_width=True)
    if os.path.exists(plots["CM"]):
        st.image(plots["CM"], caption="Confusion Matrix (at chosen threshold)", use_column_width=True)

with colC:
    if os.path.exists(plots["FI"]):
        st.image(plots["FI"], caption="Top Feature Importance", use_column_width=True)

st.divider()


# -----------------------
# Scoring: Single example form (only)
# -----------------------
st.subheader("Score new flights")

def _example_row() -> dict:
    return {
        "AIRLINE": "SpiriT",
        "ORIGIN_AIRPORT": "IAD",
        "DESTINATION_AIRPORT": "LAX",
        "ORIGIN_STATE": "MD",
        "DEST_STATE": "CA",
        "DEPARTURE_DELAY": 3.0,
        "AIR_TIME": 155.0,
        "DISTANCE": 1000.0,
        "ORIGIN_LAT": 40.64, "ORIGIN_LON": -73.78,
        "DEST_LAT": 33.94, "DEST_LON": -118.41,
        "temp": 22.0, "rhum": 60.0, "prcp": 0.0, "snow": 0.0, "wspd": 4.0, "pres": 1013.0,
    }

ex = _example_row()
with st.form("example"):
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        AIRLINE = st.text_input("AIRLINE", ex["AIRLINE"])
        ORIGIN_AIRPORT = st.text_input("ORIGIN_AIRPORT", ex["ORIGIN_AIRPORT"])
        DESTINATION_AIRPORT = st.text_input("DESTINATION_AIRPORT", ex["DESTINATION_AIRPORT"])
        ORIGIN_STATE = st.text_input("ORIGIN_STATE", ex["ORIGIN_STATE"])
        DEST_STATE = st.text_input("DEST_STATE", ex["DEST_STATE"])
    with cc2:
        DEPARTURE_DELAY = st.number_input("DEPARTURE_DELAY (min)", value=float(ex["DEPARTURE_DELAY"]), step=1.0)
        AIR_TIME = st.number_input("AIR_TIME (min)", value=float(ex["AIR_TIME"]), step=1.0)
        DISTANCE = st.number_input("DISTANCE (mi)", value=float(ex["DISTANCE"]), step=1.0)
        ORIGIN_LAT = st.number_input("ORIGIN_LAT", value=float(ex["ORIGIN_LAT"]))
        ORIGIN_LON = st.number_input("ORIGIN_LON", value=float(ex["ORIGIN_LON"]))
    with cc3:
        DEST_LAT = st.number_input("DEST_LAT", value=float(ex["DEST_LAT"]))
        DEST_LON = st.number_input("DEST_LON", value=float(ex["DEST_LON"]))
        temp = st.number_input("temp (°C)", value=float(ex["temp"]))
        rhum = st.number_input("rhum (%)", value=float(ex["rhum"]))
        prcp = st.number_input("prcp (mm)", value=float(ex["prcp"]))
        snow = st.number_input("snow (mm)", value=float(ex["snow"]))
        wspd = st.number_input("wspd (m/s)", value=float(ex["wspd"]))
        pres = st.number_input("pres (hPa)", value=float(ex["pres"]))
    submitted = st.form_submit_button("Predict")

if submitted:
    row = {
        "AIRLINE": AIRLINE, "ORIGIN_AIRPORT": ORIGIN_AIRPORT, "DESTINATION_AIRPORT": DESTINATION_AIRPORT,
        "ORIGIN_STATE": ORIGIN_STATE, "DEST_STATE": DEST_STATE,
        "DEPARTURE_DELAY": DEPARTURE_DELAY, "AIR_TIME": AIR_TIME, "DISTANCE": DISTANCE,
        "ORIGIN_LAT": ORIGIN_LAT, "ORIGIN_LON": ORIGIN_LON, "DEST_LAT": DEST_LAT, "DEST_LON": DEST_LON,
        "temp": temp, "rhum": rhum, "prcp": prcp, "snow": snow, "wspd": wspd, "pres": pres
    }
    X1 = _coerce_schema(pd.DataFrame([row]))
    proba = float(predict_proba_for(tag, model, X1)[0])
    pred = int(proba >= threshold)
    st.metric("P(delay)", f"{proba:.3f}")
    st.write(f"**Predicted class** (threshold {threshold:.3f}): **{pred}** — 0=on-time, 1=delayed")




st.divider()

# Comparison chart 

st.subheader("Model comparison snapshot")
comp_csv = "models/model_comparison.csv"
if os.path.exists(comp_csv):
    comp = pd.read_csv(comp_csv)
    st.dataframe(comp, use_container_width=True)
else:
    st.info("models/model_comparison.csv not found. Run the comparison cell to generate it.")
