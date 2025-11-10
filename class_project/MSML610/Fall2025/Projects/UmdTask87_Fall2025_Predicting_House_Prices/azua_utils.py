from __future__ import annotations
import os, json, joblib, math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

TARGET = "Price"

def load_melbourne(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[TARGET])
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
    return df

def split_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET in num_cols:
        num_cols.remove(TARGET)
    cat_cols = [c for c in df.columns if c not in num_cols + [TARGET]]
    return num_cols, cat_cols

def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )

def candidate_models(random_state: int = 42) -> Dict[str, Any]:
    return {
        "linreg": LinearRegression(),
        "elastic": ElasticNet(alpha=0.01, l1_ratio=0.1, random_state=random_state),
        "rf": RandomForestRegressor(n_estimators=400, max_depth=None, n_jobs=-1, random_state=random_state),
        "xgb": XGBRegressor(
            n_estimators=600, max_depth=8, subsample=0.8, colsample_bytree=0.8,
            learning_rate=0.05, random_state=random_state, n_jobs=-1, objective="reg:squarederror"
        ),
    }

def cv_scores(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, folds: int = 5) -> tuple[float,float]:
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    neg_mse = cross_val_score(pipe, X, y, scoring="neg_mean_squared_error", cv=kf, n_jobs=-1)
    rmse = np.mean([math.sqrt(-m) for m in neg_mse])
    r2 = np.mean(cross_val_score(pipe, X, y, scoring="r2", cv=kf, n_jobs=-1))
    return rmse, r2

def train_select_best(df: pd.DataFrame) -> Dict[str, Any]:
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(float)
    num_cols, cat_cols = split_features(df)
    pre = build_preprocessor(num_cols, cat_cols)

    results = []
    best = {"name": None, "rmse": float("inf"), "r2": -1e9, "pipeline": None}

    for name, model in candidate_models().items():
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])
        rmse, r2 = cv_scores(pipe, X, y)
        results.append({"model": name, "rmse": rmse, "r2": r2})
        if rmse < best["rmse"]:
            best.update({"name": name, "rmse": rmse, "r2": r2, "pipeline": pipe})

    best["pipeline"].fit(X, y)
    return {"best": best, "all_results": results, "num_cols": num_cols, "cat_cols": cat_cols}

def save_artifacts(best_bundle: Dict[str, Any], outdir: str = "artifacts") -> None:
    os.makedirs(outdir, exist_ok=True)
    joblib.dump(best_bundle["best"]["pipeline"], os.path.join(outdir, "model.joblib"))
    meta = {
        "model": best_bundle["best"]["name"],
        "rmse_cv": best_bundle["best"]["rmse"],
        "r2_cv": best_bundle["best"]["r2"],
        "num_cols": best_bundle["num_cols"],
        "cat_cols": best_bundle["cat_cols"],
    }
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)

def load_model(path: str = "artifacts/model.joblib"):
    return joblib.load(path)
