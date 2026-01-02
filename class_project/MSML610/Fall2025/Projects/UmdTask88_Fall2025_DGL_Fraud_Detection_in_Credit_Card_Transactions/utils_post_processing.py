# utils_post_processing.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


def _read_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_baseline_metrics(report_dir: str = "data/artifacts") -> Dict:
    """Return the logistic regression metrics JSON."""
    return _read_json(Path(report_dir) / "baseline_metrics.json")


def load_gnn_metrics(report_dir: str = "data/artifacts") -> Dict:
    """Return the final-epoch GNN metrics JSON."""
    return _read_json(Path(report_dir) / "gnn_metrics.json")


def load_lgbm_metrics(report_dir: str = "data/artifacts") -> Dict:
    """Return the LightGBM baseline metrics JSON if present."""
    path = Path(report_dir) / "lgbm_metrics.json"
    return _read_json(path) if path.exists() else {}


def build_metrics_dataframe(models: Dict[str, Dict]) -> pd.DataFrame:
    """
    Convert nested metric dicts into a tidy dataframe for plotting/markdown tables.
    """
    rows = []
    for model_name, payload in models.items():
        metrics_block = payload.get("metrics", payload)
        for split, values in metrics_block.items():
            if not isinstance(values, dict):
                continue
            row = {"model": model_name, "split": split}
            row.update(values)
            rows.append(row)
    return pd.DataFrame(rows)


def precision_recall_at_k(scores: Iterable[float], labels: Iterable[int], k: float) -> Tuple[float, float]:
    """
    Compute precision@k and recall@k for ranked predictions.
    k can be an integer count or a float fraction (0-1).
    """
    scores = np.asarray(list(scores), dtype=float)
    labels = np.asarray(list(labels), dtype=int)
    order = np.argsort(-scores)
    scores = scores[order]
    labels = labels[order]
    if 0 < k < 1:
        top_k = max(1, int(len(scores) * k))
    else:
        top_k = int(k)
    top_k = min(top_k, len(scores))
    top_labels = labels[:top_k]
    precision = float(top_labels.mean()) if top_k else float("nan")
    recall = float(top_labels.sum() / labels.sum()) if labels.sum() > 0 else float("nan")
    return precision, recall


def compare_models_table(report_dir: str = "data/artifacts") -> pd.DataFrame:
    """
    Helper used by notebooks to show baseline vs. GNN metrics side by side.
    """
    baseline = load_baseline_metrics(report_dir)
    lgbm = load_lgbm_metrics(report_dir)
    gnn_report = load_gnn_metrics(report_dir)
    models = {
        "baseline": baseline["metrics"],
        **({"lgbm": lgbm["metrics"]} if lgbm else {}),
        "gnn": {
            "train": gnn_report["train"],
            "val": gnn_report["val"],
            "test": gnn_report["test"],
        },
    }
    return build_metrics_dataframe(models)


def load_gnn_error_table(report_dir: str = "data/artifacts") -> pd.DataFrame:
    """
    Load the val/test prediction table saved by the GNN trainer for error analysis.
    """
    path = Path(report_dir) / "gnn_val_test_preds.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Could not find predictions at {path}. Run train_gnn first.")
    return pd.read_parquet(path)
