# Retail Sales Forecasting API Tutorial (JAX)

# Retail Sales Forecasting API Tutorial (JAX)

This document captures the interface-first view of the toolkit being developed
for the MSML610 project **Retail Sales Forecasting with LSTMs**. The API is
implemented in `retail_sales_forecasting_utils.py` and consumed by the
companion notebooks. The intent of the API tutorial is to explain how a user
should interact with the module without needing to read the notebook cells.

## High-Level Goals

- Provide a typed JAX pipeline that handles data loading, feature creation,
  sequence generation, model training, and evaluation for multi-store retail
  sales data.
- Support both LSTM and GRU recurrent architectures through a shared contract.
- Keep the API notebook focused on explaining configurations while the
  utilities deliver reusable functions.

## Module Overview

| Component | Responsibility | Status |
|-----------|----------------|--------|
| `DataSourceConfig` | Dataset locations, schema metadata | ✅ Implemented |
| `TemporalFeatureConfig` | Seasonalities + feature toggles | ✅ Implemented |
| `ModelConfig` | RNN hyperparameters and optimizer knobs | ✅ Implemented |
| `load_sales_data()` | Loads Kaggle files or synthetic fallback | ✅ Implemented |
| `build_feature_pipeline()` | Ordered callables to add temporal/event features | ✅ Implemented |
| `prepare_dataloader()` | Generates scaled sliding windows for train/val splits | ✅ Implemented |
| `create_rnn_model()` | Builds a Flax LSTM/GRU backbone with dense head | ✅ Implemented |
| `train_model()` | Optax-powered training loop with mini-batching | ✅ Implemented |
| `evaluate_model()` | Computes MAE/RMSE/MAPE and tidy prediction frame | ✅ Implemented |
| `ForecastArtifacts` | Wraps metrics, predictions, params, metadata | ✅ Implemented |

## Usage Workflow

1. **Configure Data Sources**

   ```python
   from retail_sales_forecasting_with_lstms.API import DataSourceConfig

   data_cfg = DataSourceConfig(
       root_dir="/data/store_sales",
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
   ```

2. **Specify Temporal Features and Model Hyperparameters**

   ```python
   feature_cfg = TemporalFeatureConfig(
       include_holidays=True,
       include_promotions=True,
       include_external_regressors=True,
       seasonalities=(7, 28, 365),
   )

   model_cfg = ModelConfig(
       cell_type="lstm",
       hidden_size=128,
       num_layers=2,
       dropout_rate=0.1,
       learning_rate=3e-4,
       weight_decay=1e-4,
       gradient_clip=1.0,
       epochs=5,
       batch_size=64,
   )
   ```

3. **Prepare Features, Train, and Evaluate**

   ```python
   pipeline = build_feature_pipeline(data_cfg, feature_cfg)
   train_ds, val_ds, metadata = prepare_dataloader(
       data_cfg,
       pipeline,
       feature_cfg=feature_cfg,
   )
   training_state = train_model(train_ds, val_ds, model_cfg, metadata)
   artifacts = evaluate_model(training_state, val_ds, metadata, metrics=("mae", "rmse", "mape"))
   ```

4. **Consume Outputs**

   - `artifacts.metrics` contains metric dictionaries keyed by scope (currently `overall`).
   - `artifacts.predictions` is a tidy DataFrame with `(store_nbr, family, horizon_step)` rows.
   - `artifacts.model_params` exposes the trained Flax parameter PyTree for serialization.

## Error Handling Strategy

- `ensure_data_root()` warns when Kaggle files are missing and the utilities
  transparently switch to a reproducible synthetic dataset for notebook demos.
- Sliding-window creation raises informative `ValueError`s when the dataset is
  too short to satisfy the requested context window or horizon.
- Training leverages gradient clipping to avoid exploding gradients and logs the
  training/validation losses every epoch for traceability.

## Dependencies

- `jax`, `flax`, and `optax` for differentiable modeling and optimization.
- `pandas`, `numpy`, and `scikit-learn` for preprocessing and scaling.
- Optional Kaggle CSV/Parquet assets; notebooks run even without them via the
  synthetic generator.

## Testing Plan

- Unit tests validate feature pipeline outputs, sequence generation shapes, and
  metric computations using small synthetic fixtures.
- Smoke tests execute the end-to-end training loop to guarantee JIT compilation
  and inference on CPU-only environments.

## Open Items

- Incorporate real holiday calendars (e.g., Ecuador-specific events) once raw
  files are available.
- Extend the evaluation module with hierarchical roll-ups (store vs family) and
  add forecast visualizations for each major event period.
