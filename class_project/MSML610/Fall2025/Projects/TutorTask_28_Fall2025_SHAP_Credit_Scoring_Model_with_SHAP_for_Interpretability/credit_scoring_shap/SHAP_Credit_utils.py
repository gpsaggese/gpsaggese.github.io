from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# --- Data loading ---

def load_credit_data(csv_path: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    """Return a small dataframe and a target column name.

    - If `csv_path` is provided and exists, we load it.
    - Otherwise we synthesize a tiny dataset that runs end-to-end.
    """
    target_col = "target"
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
        # Heuristic: try to find a plausible target
        for cand in ("target","Risk","Creditability","default","Class"):
            if cand in df.columns:
                target_col = cand
                break
        # Normalize: map to {0,1} with 1='bad' if possible
        if df[target_col].dtype.kind in "OUS":
            bad_vals = {"bad","default","2","bad risk","Bad"}
            df[target_col] = df[target_col].astype(str).str.strip().str.lower().isin(bad_vals).astype(int)
        else:
            vals = set(pd.unique(df[target_col]))
            if vals == {1,2}: df[target_col] = (df[target_col]==2).astype(int)
        return df, target_col

    # Synthetic fallback
    rng = np.random.default_rng(42)
    n = 800
    age = rng.integers(18, 75, size=n)
    income = rng.normal(60_000, 15_000, size=n)
    purpose = rng.choice(["car","furniture","education","other"], size=n)
    # crude rule to create a label with signal
    risk = (income < 50000).astype(int) | (age < 21).astype(int)
    df = pd.DataFrame({"age": age, "income": income, "purpose": purpose, "target": risk.astype(int)})
    return df, target_col

# --- Features ---

def build_preprocessor(df: pd.DataFrame, target_col: str) -> ColumnTransformer:
    cat_cols = df.drop(columns=[target_col]).select_dtypes(include=['object','category']).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols + [target_col]]
    num_pipe = Pipeline(steps=[('scaler', StandardScaler())])
    cat_pipe = Pipeline(steps=[('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=True))])
    pre = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
    return pre

# --- Modeling ---

def train_xgb(X_train, y_train, seed: int = 42):
    pos = (y_train==1).mean()
    scale_pos_weight = max((1-pos)/(pos+1e-9), 1.0)
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='binary:logistic',
        eval_metric='logloss',
        n_jobs=4,
        tree_method='hist',
        scale_pos_weight=scale_pos_weight,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    return clf

# --- Evaluation ---

def evaluate_model(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:,1]
    preds = (proba>=0.5).astype(int)
    auc = float(roc_auc_score(y_test, proba))
    cm = confusion_matrix(y_test, preds).tolist()
    report = classification_report(y_test, preds, output_dict=True)
    return {"test_auc": auc, "confusion_matrix": cm, "classification_report": report}

# --- SHAP ---

def compute_shap(model, X_sample, preprocessor=None):
    """Placeholder SHAP computation. Returns raw model outputs for now.
    Replace with shap.TreeExplainer(model)(X_sample_dense) during the next PR.
    """
    # A very light stub to keep notebooks runnable without heavy plotting.
    # In the next iteration, import shap and compute summary/beeswarm plots.
    return model.predict_proba(X_sample)[:,1][:10]
