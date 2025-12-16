# Retail Sales Forecasting with LSTMs — Example Walkthrough

This document mirrors the notebook/script pair in `retail_sales_forecasting_with_lstms.example.*`.
It explains the end-to-end workflow shown in the final video: from loading the
Kaggle Store Sales dataset to generating the plots and metrics bundled under
`artifacts/run_20251215_212247/`.

## Storyboard

1. **Frame the problem** – predict weekly demand per `(store_nbr, family)` so
   the retailer can plan promotions, staffing, and inventory while respecting
   holidays and oil-price shocks.
2. **Engineer retail-aware features** – merge transactions, promotions, weekend
   flags, and cyclical encodings to capture seasonality.
3. **Train LSTM/GRU baselines in JAX** – leverage a shared data loader, Optax
   optimizer, and best-checkpoint tracking.
4. **Evaluate & visualize** – compare metrics in normalized vs sales units,
   inspect per-store/per-family breakdowns, and plot holiday/promotion effects.
5. **Package artifacts** – save config/params/plots for reviewers and record the
   video without re-running expensive training.

## 1. Environment + Data

```python
from pathlib import Path
from retail_sales_forecasting_with_lstms.API import TrainingConfig

DATA_DIR = Path("data/store-sales-time-series-forecasting")
ARTIFACTS_DIR = Path("artifacts")

cfg = TrainingConfig(
    data_dir=DATA_DIR,
    families=("GROCERY I", "BEVERAGES", "PRODUCE", "CLEANING", "DAIRY"),
    max_stores=10,
    context_length=30,
    horizon=7,
    epochs=6,
)
```
- If `DATA_DIR` does not contain the Kaggle CSVs the notebooks fall back to the
  synthetic generator so that the workflow remains runnable on any laptop.

## 2. Data Snapshot

```python
from retail_sales_forecasting_with_lstms.API import prepare_dataset

dataset = prepare_dataset(cfg)
print(dataset.train_inputs.shape, dataset.val_inputs.shape)
```
- Inspect `dataset.train_inputs[:3]` to show the engineered columns (promo,
  transactions, sine/cosine features, entity embeddings).
- Display a few rows from `dataset.val_store_ids` and `dataset.val_family_ids`
  to highlight the coverage.

## 3. Train Both Models

```python
from retail_sales_forecasting_with_lstms.API import run_training_pipeline

run_dir, results, dataset = run_training_pipeline(
    cfg,
    output_dir=ARTIFACTS_DIR,
    run_name="run_20251215_212247",
)
for result in results:
    print(result.name, result.normalized_metrics)
```
- LSTM validation RMSE (normalized) ≈ 0.138 | sales RMSE ≈ 1,231 units.
- GRU validation RMSE (normalized) ≈ 0.144 | sales RMSE ≈ 1,305 units.
- The script automatically persists `config.json`, model metrics, and pickled
  parameters under `artifacts/run_20251215_212247/`.

## 4. Evaluation & Visualization

```python
from retail_sales_forecasting_with_lstms.API import (
    load_run_metrics,
    plot_training_curves,
    plot_final_metrics_comparison,
    plot_sales_metrics,
    plot_breakdowns,
)

metrics = load_run_metrics(run_dir)
plot_training_curves(metrics, run_dir)
plot_final_metrics_comparison(metrics, run_dir)
plot_sales_metrics(metrics, run_dir)
plot_breakdowns(metrics, run_dir, model_name="lstm")
plot_breakdowns(metrics, run_dir, model_name="gru")
```
- `training_curves.png` shows both models converging within ~6 epochs.
- `final_metrics_comparison.png` highlights the LSTM beating the GRU in every
  normalized metric, albeit by a narrow margin.
- `sales_metrics.png` translates the errors back to original currency for a
  business-friendly story.
- Breakdown plots highlight which stores/families drive the residual error:
  promotions-heavy weeks remain challenging, whereas Grocery I performs best.

## 5. Forecast Visualization

```python
from retail_sales_forecasting_with_lstms.API import run_inference, plot_predictions_sample

inference = run_inference(run_dir, "lstm", dataset=dataset)
plot_predictions_sample(
    inference["predictions"],
    inference["targets"],
    dataset,
    run_dir / "lstm_test_predictions.png",
)
```
- Overlay actual vs predicted sales for a validation horizon, annotating holiday
  spikes and promotion periods (see `lstm_test_predictions.png`).

## 6. Storytelling Notes

- Holidays introduce higher variance but the model still keeps RMSE under 204
  normalized units for those windows.
- Promotions remain the top driver of error → next steps include richer promo
  features or exogenous regressors (oil, transactions lag).
- Despite shared weights across entities, the scaled store/family embeddings let
  the model tailor behaviour: clusters with similar traffic share similar errors.

## 7. Wrap-up for the Video

1. **Files overview** – show API/example notebooks, utils module, docs, and the
   `artifacts/` folder.
2. **Docker + notebooks** – open the container, run notebooks (Restart & Run All)
   to recreate the key tables/plots.
3. **Results discussion** – walk through the generated PNGs, call out LSTM vs
   GRU performance, and highlight the holiday/promotion analysis.
4. **Documentation** – point reviewers to `docs/architecture_notes.md` for the
   narrative and `README.md` for reproduction instructions.
