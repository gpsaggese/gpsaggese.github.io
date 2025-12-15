from __future__ import annotations

from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

from .config import ModelConfig


def build_model(cfg: ModelConfig) -> xgb.XGBClassifier:
    """
    Construct an XGBoost classifier with sensible defaults for credit scoring.
    """
    model = xgb.XGBClassifier(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_lambda=cfg.reg_lambda,
        reg_alpha=cfg.reg_alpha,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
        use_label_encoder=False,
    )
    return model


def train_model(
    model: xgb.XGBClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> xgb.XGBClassifier:
    """
    Fit the model on the training data.
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """
    Compute AUC, confusion matrix, classification report and return predictions.

    Conventions:
    - y = 1 means Bad (default risk, positive class)
    - y = 0 means Good
    - y_proba = P(Bad)
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(
        y_test, y_pred, target_names=["Good", "Bad"]
    )

    metrics = {
        "auc": float(auc),
        "confusion_matrix": cm,
        "classification_report": report_text,
        "threshold": threshold,
    }
    return metrics, y_proba, y_pred
