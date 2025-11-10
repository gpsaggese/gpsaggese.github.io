"""
Attrition_utils.py
Reusable utilities and a lightweight wrapper layer for Employee Attrition Prediction.
This module is imported by both Attrition.API.ipynb and Attrition.example.ipynb.
"""
from dataclasses import dataclass
from typing import Protocol, List, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

# ---- API layer (stable contract) ----
@dataclass
class AttritionConfig:
    seed: int = 42
    test_size: float = 0.25
    learning_rate: float = 0.08
    n_estimators: int = 400
    max_depth: int = 3
    subsample: float = 0.9
    target_col: str = "Attrition"
    drop_id_cols: Tuple[str,...] = ("EmployeeNumber",)

class AttritionAPI(Protocol):
    """Abstract service interface for attrition modeling."""
    def fit(self, df: pd.DataFrame) -> "AttritionService": ...
    def predict(self, df: pd.DataFrame) -> np.ndarray: ...
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray: ...

# ---- Reference implementation (example layer) ----
class AttritionService:
    def __init__(self, cfg: AttritionConfig):
        self.cfg = cfg
        self.pipe: Pipeline | None = None
        self.cat_cols: List[str] = []
        self.num_cols: List[str] = []

    def _split(self, df: pd.DataFrame):
        y = (df[self.cfg.target_col] == "Yes").astype(int)
        X = df.drop(columns=[self.cfg.target_col, *self.cfg.drop_id_cols], errors="ignore")
        self.cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
        self.num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        return train_test_split(
            X, y, test_size=self.cfg.test_size, stratify=y, random_state=self.cfg.seed
        )

    def fit(self, df: pd.DataFrame) -> "AttritionService":
        X_train, X_test, y_train, y_test = self._split(df)
        pre = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_cols),
                ("num", "passthrough", self.num_cols),
            ]
        )
        gbm = GradientBoostingClassifier(
            random_state=self.cfg.seed,
            learning_rate=self.cfg.learning_rate,
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            subsample=self.cfg.subsample
        )
        self.pipe = Pipeline([("prep", pre), ("model", gbm)])
        self.pipe.fit(X_train, y_train)
        proba = self.pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = dict(
            accuracy=float(accuracy_score(y_test, pred)),
            f1=float(f1_score(y_test, pred)),
            roc_auc=float(roc_auc_score(y_test, proba))
        )
        self.last_metrics_ = metrics
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        assert self.pipe is not None, "Model not fitted"
        return self.pipe.predict(df)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        assert self.pipe is not None, "Model not fitted"
        return self.pipe.predict_proba(df)
