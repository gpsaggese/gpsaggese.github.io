"""
Typed API surface for the Retail Sales Forecasting project (JAX).

The API notebook imports from this module to explain the configuration objects,
data preparation utilities, training pipeline, inference helpers, and plotting
functions without diving into the implementation details inside
`retail_sales_forecasting_utils.py`.
"""

from retail_sales_forecasting_utils import (
    DatasetSplits,
    TrainingConfig,
    TrainingResult,
    build_model,
    create_run_directory,
    load_run_config,
    load_run_metrics,
    plot_breakdowns,
    plot_final_metrics_comparison,
    plot_predictions_sample,
    plot_sales_metrics,
    plot_training_curves,
    prepare_dataset,
    run_inference,
    run_training_pipeline,
    save_run_artifacts,
    train_model,
    train_models,
)

__all__ = [
    "TrainingConfig",
    "DatasetSplits",
    "TrainingResult",
    "prepare_dataset",
    "build_model",
    "train_model",
    "train_models",
    "run_training_pipeline",
    "run_inference",
    "create_run_directory",
    "save_run_artifacts",
    "load_run_config",
    "load_run_metrics",
    "plot_training_curves",
    "plot_final_metrics_comparison",
    "plot_sales_metrics",
    "plot_breakdowns",
    "plot_predictions_sample",
]
