# utils_post_processing.py

from typing import Dict
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def compute_metrics(y_true, y_prob) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
    }
