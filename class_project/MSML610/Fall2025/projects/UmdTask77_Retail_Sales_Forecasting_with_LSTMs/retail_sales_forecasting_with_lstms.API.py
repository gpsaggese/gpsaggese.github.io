"""
Typed API surface for the Retail Sales Forecasting project.

The module re-exports the configuration dataclasses and high-level functions
implemented in `retail_sales_forecasting_utils.py`. The `.API.ipynb` notebook
will import from here to keep the interface documentation and executable code
aligned.

References:
- `retail_sales_forecasting_with_lstms.API.md`
- Kaggle Store Sales: Time Series Forecasting competition.
"""

from retail_sales_forecasting_utils import (
    DataSourceConfig,
    FeatureGenerator,
    ForecastArtifacts,
    ModelConfig,
    TemporalFeatureConfig,
    WindowedDataset,
    build_feature_pipeline,
    create_rnn_model,
    ensure_data_root,
    evaluate_model,
    prepare_dataloader,
    train_model,
)

__all__ = [
    "DataSourceConfig",
    "TemporalFeatureConfig",
    "ModelConfig",
    "FeatureGenerator",
    "WindowedDataset",
    "ForecastArtifacts",
    "build_feature_pipeline",
    "ensure_data_root",
    "prepare_dataloader",
    "create_rnn_model",
    "train_model",
    "evaluate_model",
]
