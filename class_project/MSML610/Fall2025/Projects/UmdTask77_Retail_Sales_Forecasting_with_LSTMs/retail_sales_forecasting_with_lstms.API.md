# Retail Sales Forecasting API Tutorial (JAX)

The API surface is intentionally thin: notebooks/scripts only need to import a
handful of dataclasses and helper functions from
`retail_sales_forecasting_with_lstms.API`. The underlying implementation lives
in `retail_sales_forecasting_utils.py` and exposes the full lifecycle for the
project:

```
TrainingConfig -> prepare_dataset -> train_models/run_training_pipeline ->
load_run_metrics/run_inference/plot_* helpers
```

## Key Dataclasses

| Object | Description |
| --- | --- |
| `TrainingConfig` | Controls dataset filters (families, stores, windows), RNN hyper-parameters, and whether to fall back to synthetic data when Kaggle CSVs are missing. |
| `DatasetSplits` | Holds the prepared train/validation windows plus metadata arrays used for breakdown metrics. |
| `TrainingResult` | Stores the history, metrics, and pickled parameters for an individual model run (LSTM or GRU). |

Example configuration:
```python
from pathlib import Path
from retail_sales_forecasting_with_lstms.API import TrainingConfig

cfg = TrainingConfig(
    data_dir=Path("data/store-sales-time-series-forecasting"),
    families=("GROCERY I", "BEVERAGES", "PRODUCE", "CLEANING", "DAIRY"),
    max_stores=10,
    context_length=30,
    horizon=7,
    epochs=6,
    batch_size=256,
    synthetic_if_missing=True,
)
```

## Dataset Preparation

```python
from retail_sales_forecasting_with_lstms.API import prepare_dataset

dataset = prepare_dataset(cfg)
print(dataset.train_inputs.shape)  # (num_windows, context, feature_dim)
print(dataset.val_targets.shape)   # (num_windows, horizon)
```
- Automatically merges holidays, transactions, promotions, cyclical encodings,
  and entity embeddings.
- If CSV files are absent the helper generates a reproducible synthetic dataset
  so notebooks/tests can run anywhere.

## Training Workflow

### 1. Train models inside the notebook
```python
from retail_sales_forecasting_with_lstms.API import train_models

results, dataset = train_models(cfg, model_names=("lstm", "gru"))
for result in results:
    print(result.name, result.normalized_metrics)
```
- Returns both the list of `TrainingResult` objects and the dataset used for
  evaluation so that downstream cells can reuse it.

### 2. Persist a full run for documentation/video
```python
from pathlib import Path
from retail_sales_forecasting_with_lstms.API import run_training_pipeline

run_dir, results, dataset = run_training_pipeline(
    cfg,
    output_dir=Path("artifacts"),
    run_name="run_20251215_212247",
)
```
Outputs stored inside `run_dir`:
- `config.json` – serialized `TrainingConfig` used for the run.
- `{model}_metrics.json` – history, metrics (normalized + sales space), and
  breakdowns grouped by store/family/holiday/promotion.
- `{model}_params.pkl` – pickled JAX parameter PyTree for inference.
- `summary.json` – quick reference for the video.

## Inference Helpers

```python
from retail_sales_forecasting_with_lstms.API import run_inference

inference = run_inference(run_dir, model_name="lstm", dataset=dataset)
print(inference["mse"], inference["mae"])
```
- Loads the saved config + parameters and runs validation predictions on CPU or
  GPU (`device="cpu"|"gpu"`).
- Returns numpy arrays so notebooks can slice and visualize forecast horizons.

## Plotting Helpers

```python
from retail_sales_forecasting_with_lstms.API import (
    load_run_metrics,
    plot_training_curves,
    plot_final_metrics_comparison,
    plot_sales_metrics,
    plot_breakdowns,
    plot_predictions_sample,
)

metrics = load_run_metrics(run_dir)
plot_training_curves(metrics, run_dir)
plot_final_metrics_comparison(metrics, run_dir)
plot_sales_metrics(metrics, run_dir)
plot_breakdowns(metrics, run_dir, model_name="lstm")
plot_predictions_sample(
    inference["predictions"],
    inference["targets"],
    dataset,
    run_dir / "lstm_test_predictions.png",
)
```
The generated PNGs are the same ones embedded in the `artifacts/` folder and in
the tutorial markdown.

## Error Handling Strategy

- `prepare_dataset` raises a clear `ValueError` if the requested windows/horizon
  cannot be satisfied (e.g., filtering too aggressively).
- `run_training_pipeline` logs any horizon mismatch and updates the config to
  the dataset-derived horizon to keep the run stable.
- All file loaders fail fast with descriptive messages so students immediately
  know whether they forgot to download the Kaggle CSVs.

## Testing

`tests/test_retail_sales_forecasting_utils.py` validates:
- Synthetic fallback data generation and sliding-window creation.
- A CPU-only training pass with `epochs=2` that saves a run directory.
- Metric JSON parsing via `load_run_metrics`.
These tests run quickly inside CI (no GPU required) and guard against refactors
breaking the notebooks or the scripted tutorial.
