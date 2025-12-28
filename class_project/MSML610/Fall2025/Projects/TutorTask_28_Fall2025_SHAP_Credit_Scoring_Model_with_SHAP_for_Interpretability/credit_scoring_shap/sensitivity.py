from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import TrainingConfig


def _ensure_reports_dir(cfg: TrainingConfig):
    return cfg.ensure_reports_dir()


def _vary_numeric_feature(
    model,
    X: pd.DataFrame,
    instance_idx: int,
    feature: str,
    num_points: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a 1D sweep for a single numeric feature and record changes in 
    predicted probability of default (bad credit)
    """
    row = X.iloc[instance_idx].copy()
    f_min, f_max = X[feature].min(), X[feature].max()
    values = np.linspace(f_min, f_max, num_points)
    probs = []
    for v in values:
        tmp = row.copy()
        tmp[feature] = v
        prob = model.predict_proba(tmp.values.reshape(1, -1))[0, 1]
        probs.append(prob)
    return values, np.array(probs)


def run_sensitivity_for_instance(
    model,
    X: pd.DataFrame,
    shap_values,
    cfg: TrainingConfig,
    instance_idx: int,
    top_n: int = 3,
) -> Dict[str, List[Tuple[float, float]]]:
    """
    For a single instance, pick top_n features by |SHAP| and run numeric
    sensitivity analysis on them, saving line plots.

    Returns:
        Dict[feature_name, list of (value, prob) tuples]
    """
    reports_dir = _ensure_reports_dir(cfg)

    sv = shap_values[instance_idx]
    mean_abs = np.abs(sv)
    top_idx = np.argsort(mean_abs)[::-1][:top_n]
    top_features = [X.columns[i] for i in top_idx]

    results: Dict[str, List[Tuple[float, float]]] = {}

    for feat in top_features:
        if not np.issubdtype(X[feat].dtype, np.number):
            # For one-hot/categorical features, sensitivity is less intuitive.
            continue

        values, probs = _vary_numeric_feature(model, X, instance_idx, feat)
        results[feat] = list(zip(values.tolist(), probs.tolist()))

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(values, probs, marker="o")
        ax.set_title(f"Sensitivity for '{feat}' (instance {instance_idx})")
        ax.set_xlabel(feat)
        ax.set_ylabel("P(bad credit)")
        fig.tight_layout()
        fname = f"sensitivity_{feat.replace(' ', '_')}_{instance_idx}.png"
        fig.savefig(reports_dir / fname, bbox_inches="tight")
        plt.close(fig)

    return results
