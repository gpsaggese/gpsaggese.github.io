"""
Executable entry point illustrating how the forecasting API is applied.

The script mirrors the steps in `retail_sales_forecasting_with_lstms.example.ipynb`
so that automated tests (to be added) can validate the notebooks' logic without
requiring a GUI runtime.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

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

_LOG = logging.getLogger(__name__)


def _default_data_config(data_root: Path) -> DataSourceConfig:
    """Create a configuration pointing at the Kaggle Store Sales dataset layout."""
    return DataSourceConfig(
        root_dir=data_root,
        sales_file="train.parquet",
        calendar_file="holidays_events.csv",
        oil_file="oil.csv",
        transactions_file="transactions.csv",
        id_columns=("store_nbr", "family"),
        date_column="date",
        target_column="sales",
        frequency="D",
        horizon_days=28,
    )


def run_demo(data_root: Path, metrics: Sequence[str] = ("mae", "rmse", "mape")) -> None:
    """
    Orchestrate the end-to-end workflow using configuration defaults.

    Parameters
    ----------
    data_root
        Base directory containing the Kaggle Store Sales dataset.
    metrics
        Metrics to compute during evaluation.
    """
    data_cfg = _default_data_config(data_root)
    feature_cfg = TemporalFeatureConfig()
    model_cfg = ModelConfig()

    _LOG.info("Starting demo run with data root %s", data_root)
    ensure_data_root(data_cfg, strict=False)
    feature_generators = build_feature_pipeline(data_cfg, feature_cfg)
    train_dataset, val_dataset, metadata = prepare_dataloader(
        data_cfg, feature_generators, feature_cfg=feature_cfg
    )
    trained_state = train_model(train_dataset, val_dataset, model_cfg, metadata)
    artifacts = evaluate_model(trained_state, val_dataset, metadata, metrics)

    _LOG.info("Demo metrics: %s", artifacts.metrics)
    _LOG.info("Predictions preview:\n%s", artifacts.predictions.head())
    _LOG.debug("Model params summary: %s", trained_state["state"].params)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_demo(Path("/data/store_sales"))
