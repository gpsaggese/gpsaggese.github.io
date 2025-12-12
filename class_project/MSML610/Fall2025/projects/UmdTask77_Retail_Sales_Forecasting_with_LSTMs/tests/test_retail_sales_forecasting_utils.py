"""
Lightweight regression tests for the retail sales forecasting utilities.

These tests rely on the synthetic dataset fallback so they can run in CI without
requiring Kaggle credentials or large downloads.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retail_sales_forecasting_utils import (
    DataSourceConfig,
    ModelConfig,
    TemporalFeatureConfig,
    build_feature_pipeline,
    ensure_data_root,
    evaluate_model,
    prepare_dataloader,
    train_model,
)


def _make_configs(tmp_path: Path) -> tuple[DataSourceConfig, TemporalFeatureConfig, ModelConfig]:
    data_cfg = DataSourceConfig(
        root_dir=tmp_path,
        sales_file="train.parquet",
        calendar_file="holidays_events.csv",
        oil_file="oil.csv",
        transactions_file="transactions.csv",
        id_columns=("store_nbr", "family"),
        date_column="date",
        target_column="sales",
        frequency="D",
        horizon_days=14,
        allow_synthetic=True,
    )
    feature_cfg = TemporalFeatureConfig(
        include_external_regressors=True,
        include_promotions=True,
        lookback_days=60,
        train_ratio=0.7,
    )
    model_cfg = ModelConfig(
        hidden_size=64,
        num_layers=1,
        epochs=2,
        batch_size=64,
        learning_rate=5e-4,
    )
    return data_cfg, feature_cfg, model_cfg


def test_prepare_dataloader_shapes(tmp_path):
    """Verify dataloader pipelines produce non-empty sliding windows."""
    data_cfg, feature_cfg, _ = _make_configs(tmp_path)
    ensure_data_root(data_cfg)
    pipeline = list(build_feature_pipeline(data_cfg, feature_cfg))
    train_ds, val_ds, metadata = prepare_dataloader(data_cfg, pipeline, feature_cfg)

    assert train_ds.inputs.ndim == 3
    assert train_ds.targets.ndim == 2
    assert train_ds.inputs.shape[0] > 0
    assert train_ds.targets.shape[-1] == data_cfg.horizon_days
    assert "feature_columns" in metadata
    assert len(metadata["feature_columns"]) == train_ds.inputs.shape[-1]
    # Validation fallback should yield at least one window.
    assert val_ds.inputs.shape[0] > 0


def test_train_and_evaluate(tmp_path):
    """Smoke test the JAX training loop and metric computation pipeline."""
    data_cfg, feature_cfg, model_cfg = _make_configs(tmp_path)
    pipeline = list(build_feature_pipeline(data_cfg, feature_cfg))
    train_ds, val_ds, metadata = prepare_dataloader(data_cfg, pipeline, feature_cfg)
    trained_state = train_model(train_ds, val_ds, model_cfg, metadata)
    artifacts = evaluate_model(trained_state, val_ds, metadata, metrics=("mae", "rmse", "mape"))

    assert "params" in trained_state
    assert "history" in trained_state
    assert isinstance(artifacts.metrics, dict)
    assert "overall" in artifacts.metrics.get("mae", {})
    preds = artifacts.predictions
    assert not preds.empty
    assert set(["timestamp", "prediction", "target"]).issubset(preds.columns)
    # Metrics should be finite after inverse transforms.
    mae = artifacts.metrics["mae"]["overall"]
    assert np.isfinite(mae)

