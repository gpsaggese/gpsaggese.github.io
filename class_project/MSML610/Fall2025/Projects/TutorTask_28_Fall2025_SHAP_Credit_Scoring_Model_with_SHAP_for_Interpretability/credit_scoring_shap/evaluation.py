from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

from .config import TrainingConfig


def _ensure_reports_dir(cfg: TrainingConfig) -> Path:
    return cfg.ensure_reports_dir()


def plot_confusion_matrix(
    cm: np.ndarray,
    cfg: TrainingConfig,
    class_names=("Good", "Bad"),
    filename: str = "confusion_matrix.png",
) -> None:
    """
    Save a simple confusion-matrix heatmap.
    """
    reports_dir = _ensure_reports_dir(cfg)

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="black",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(reports_dir / filename, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    y_test,
    y_proba,
    cfg: TrainingConfig,
    filename: str = "roc_pr_curves.png",
) -> None:
    """
    Save ROC and PR curves side-by-side.
    """
    reports_dir = _ensure_reports_dir(cfg)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[0])
    axes[0].set_title("ROC curve")

    PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=axes[1])
    axes[1].set_title("Precision–Recall curve")

    fig.tight_layout()
    fig.savefig(reports_dir / filename, bbox_inches="tight")
    plt.close(fig)


def save_metrics_text(
    metrics: Dict[str, Any],
    cfg: TrainingConfig,
    filename: str = "metrics.txt",
) -> None:
    """
    Persist AUC, threshold, confusion matrix and classification report
    to a plain-text file for grading / inspection.
    """
    reports_dir = _ensure_reports_dir(cfg)
    out_path = reports_dir / filename
    with out_path.open("w") as f:
        f.write(f"AUC: {metrics['auc']:.4f}\n")
        f.write(f"Threshold: {metrics['threshold']:.2f}\n\n")
        f.write("Confusion matrix:\n")
        for row in metrics["confusion_matrix"]:
            f.write(" ".join(str(int(x)) for x in row) + "\n")
        f.write("\nClassification report:\n")
        f.write(metrics["classification_report"])
