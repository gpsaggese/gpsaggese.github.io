# utils_data_io.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from src.data import load_data
from src.features import make_features
from src.graph import construct_hetero_graph
from src.models.tabular_baselines import run_baseline
from src.train.train_gnn import train_gnn


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Read YAML config once and hand it to notebooks or scripts."""
    with open(config_path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def build_dataset(
    config_path: str = "configs/default.yaml",
    sample_frac: Optional[float] = None,
    max_rows: Optional[int] = None,
) -> Path:
    """
    Run the raw → merged parquet pipeline. Returns path to the merged parquet file.
    """
    cfg = load_config(config_path)
    merged = load_data.run(
        raw_dir=cfg["data"]["raw_dir"],
        out_path=cfg["data"]["merged_file"],
        sample_frac=sample_frac if sample_frac is not None else cfg["data"]["sample_frac"],
        random_state=cfg["data"]["random_state"],
        max_rows=max_rows,
    )
    return Path(merged)


def build_features(config_path: str = "configs/default.yaml") -> Path:
    """
    Run deterministic feature engineering and return the feature parquet path.
    """
    cfg = load_config(config_path)
    out = make_features.run(
        in_path=cfg["data"]["merged_file"],
        out_path=cfg["tabular"]["features_out"],
        account_key=cfg["graph"]["account_key"],
        target=cfg["tabular"]["target"],
    )
    return Path(out)


def build_graph(config_path: str = "configs/default.yaml") -> Path:
    """
    Materialize the heterogeneous transaction↔account graph artifact.
    """
    cfg = load_config(config_path)
    graph_path = construct_hetero_graph.run(
        in_path=cfg["tabular"]["features_out"],
        account_key=cfg["graph"]["account_key"],
        save_path=cfg["graph"]["save_path"],
    )
    return Path(graph_path)


def train_tabular_baseline(
    config_path: str = "configs/default.yaml",
    threshold: float = 0.5,
    val_days: Optional[int] = None,
    test_days: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Convenience wrapper used in notebooks to run the logistic regression baseline.
    """
    cfg = load_config(config_path)
    val_days_use = val_days if val_days is not None else cfg["splits"]["val_days"]
    test_days_use = test_days if test_days is not None else cfg["splits"]["test_days"]
    return run_baseline(
        features_path=cfg["tabular"]["features_out"],
        report_dir=cfg["tabular"]["report_dir"],
        account_key=cfg["graph"]["account_key"],
        val_days=val_days_use,
        test_days=test_days_use,
        target=cfg["tabular"]["target"],
        threshold=threshold,
        model_type="logreg",
        tune_threshold=True,
    )


def train_lgbm_baseline(
    config_path: str = "configs/default.yaml",
    threshold: float = 0.5,
    val_days: Optional[int] = None,
    test_days: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Wrapper to train the LightGBM baseline with class imbalance handling.
    """
    cfg = load_config(config_path)
    val_days_use = val_days if val_days is not None else cfg["splits"]["val_days"]
    test_days_use = test_days if test_days is not None else cfg["splits"]["test_days"]
    return run_baseline(
        features_path=cfg["tabular"]["features_out"],
        report_dir=cfg["tabular"]["report_dir"],
        account_key=cfg["graph"]["account_key"],
        val_days=val_days_use,
        test_days=test_days_use,
        target=cfg["tabular"]["target"],
        threshold=threshold,
        model_type="lgbm",
        tune_threshold=True,
    )


def train_gnn_model(
    config_path: str = "configs/default.yaml",
    epochs: int = 5,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    threshold: float = 0.5,
    device: Optional[str] = None,
    pos_weight_scale: float = 1.2,
    val_days: Optional[int] = None,
    test_days: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Notebook-friendly function to trigger the GraphSAGE training loop with overrides.
    """
    cfg = load_config(config_path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    val_days_use = val_days if val_days is not None else cfg["splits"]["val_days"]
    test_days_use = test_days if test_days is not None else cfg["splits"]["test_days"]
    history = train_gnn(
        num_epochs=epochs,
        lr=lr,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        account_key=cfg["graph"]["account_key"],
        val_days=val_days_use,
        test_days=test_days_use,
        threshold=threshold,
        device=device,
        pos_weight_scale=pos_weight_scale,
    )
    return history
