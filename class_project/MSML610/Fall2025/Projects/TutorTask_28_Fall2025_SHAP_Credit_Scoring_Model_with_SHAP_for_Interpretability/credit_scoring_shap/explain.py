from __future__ import annotations

from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .config import TrainingConfig


def build_shap_explainer(
    model,
    X_background: pd.DataFrame,
) -> Tuple[shap.TreeExplainer, np.ndarray]:
    """
    Build a SHAP TreeExplainer and pre-compute SHAP values on background data.

    For binary classification, returns SHAP values for the positive class (Bad / default risk).

"""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_background)

    # For XGBoost binary classifier, shap_values can be an array or list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class
    return explainer, shap_values


def _ensure_reports_dir(cfg: TrainingConfig) -> Path:
    return cfg.ensure_reports_dir()


def plot_global_shap_summary(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    cfg: TrainingConfig,
    max_display: int = 20,
) -> None:
    """
    Save both SHAP bar summary and beeswarm (dot) plot.
    """
    reports_dir = _ensure_reports_dir(cfg)

    # Bar plot – global importance
    shap.summary_plot(
        shap_values,
        X,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    fig = plt.gcf()
    fig.savefig(reports_dir / "shap_summary_bar.png", bbox_inches="tight")
    plt.close(fig)

    # Beeswarm – distribution of impact
    shap.summary_plot(
        shap_values,
        X,
        max_display=max_display,
        show=False,
    )
    fig = plt.gcf()
    fig.savefig(reports_dir / "shap_summary_beeswarm.png", bbox_inches="tight")
    plt.close(fig)


def plot_shap_dependence_for_top_feature(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    cfg: TrainingConfig,
) -> str:
    """
    Pick the single most important feature by mean |SHAP| and save a dependence plot.

    Returns:
        The feature name used.
    """
    reports_dir = _ensure_reports_dir(cfg)

    mean_abs = np.abs(shap_values).mean(axis=0)
    idx = int(np.argmax(mean_abs))
    top_feat = X.columns[idx]

    shap.dependence_plot(
        top_feat,
        shap_values,
        X,
        show=False,
    )
    fig = plt.gcf()
    fname = f"shap_dependence_{top_feat.replace(' ', '_')}.png"
    fig.savefig(reports_dir / fname, bbox_inches="tight")
    plt.close(fig)

    return top_feat


def plot_shap_decision_for_index(
    explainer: shap.TreeExplainer,
    shap_values: np.ndarray,
    X: pd.DataFrame,
    index: int,
    cfg: TrainingConfig,
    prefix: str = "example",
) -> None:
    """
    Save a SHAP 'decision' style plot for a single instance.

    This shows how the model moves from the base value to the final prediction.
    """
    reports_dir = _ensure_reports_dir(cfg)

    shap.decision_plot(
        explainer.expected_value,
        shap_values[index],
        X.iloc[index],
        feature_names=X.columns.tolist(),
        show=False,
    )
    fig = plt.gcf()
    fname = f"shap_decision_{prefix}_{index}.png"
    fig.savefig(reports_dir / fname, bbox_inches="tight")
    plt.close(fig)