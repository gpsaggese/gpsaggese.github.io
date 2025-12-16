# Retail Sales Forecasting with LSTMs (JAX)

This directory contains my MSML610 Fall 2025 class project. The deliverable is a
complete JAX/Flax forecasting toolkit that ingests the Kaggle **Store Sales –
Time Series Forecasting** dataset, engineers retail-aware features, trains LSTM
and GRU baselines, evaluates per-store/per-family accuracy, and ships charts,
metrics, and documentation ready for a PR/video walkthrough.

## Directory Layout

- `retail_sales_forecasting_with_lstms.API.*` – interface-first markdown and
  notebook that document the reusable API surface.
- `retail_sales_forecasting_with_lstms.example.*` – tutorial markdown/notebook +
  the runnable script that mirrors the 60-minute walkthrough.
- `retail_sales_forecasting_utils.py` – all shared logic: dataclasses,
  preprocessing, model builders, Optax training loop, inference helpers, and
  plotting utilities.
- `artifacts/run_20251215_212247/` – saved `config.json`, model metrics, and the
  plots referenced in the docs/video.
- `docs/architecture_notes.md` & `docs/instructions.md` – high-level story and
  the original project description for grading reference.
- `docker_simple/` – lightweight Docker workflow used for development/testing.
- `tests/` – regression tests that exercise the synthetic data fallback, model
  training, and artifact generation on CPU-only CI.

## Data and Environment

1. Download the Kaggle dataset into `data/store-sales-time-series-forecasting/`
   (same layout as the competition ZIP). The utilities automatically fall back
   to the bundled synthetic generator if the CSVs are missing, which keeps the
   notebooks/test suite runnable anywhere.
2. Build the Docker image once:
   ```bash
   cd class_project/MSML610/Fall2025/Projects/UmdTask77_Retail_Sales_Forecasting_with_LSTMs/docker_simple
   bash docker_build.sh
   ```
   *Why JAX 0.4.13?* Ubuntu 20.04 ships Python 3.8, and the Dockerfile pins the
   matching `jax[cpu]` wheel from the Google-hosted index. On Apple/ARM hosts
   the build script automatically targets `linux/amd64` so the wheel resolves
   cleanly (even though you’re running on ARM hardware).
3. Launch a shell or JupyterLab via `bash docker_bash.sh` / `bash docker_jupyter.sh`.
   Pass `-d /Users/.../umd_classes` (or let it default to the repo root) so the
   script mounts your workspace at `/app/project`; otherwise the notebook file
   browser will appear empty.

## Quick Start Recipes

### Run the scripted tutorial
```bash
cd class_project/MSML610/Fall2025/Projects/UmdTask77_Retail_Sales_Forecasting_with_LSTMs
python retail_sales_forecasting_with_lstms.example.py
```
This trains both models (synthetic fallback if needed), saves a run directory
under `artifacts/`, generates all plots (training curves, breakdowns, prediction
samples), and prints the validation metrics used in the presentation.

### Programmatic training from a notebook
```python
from pathlib import Path
from retail_sales_forecasting_with_lstms.API import TrainingConfig, run_training_pipeline, load_run_metrics, plot_training_curves

cfg = TrainingConfig(
    data_dir=Path("data/store-sales-time-series-forecasting"),
    families=("GROCERY I", "BEVERAGES", "PRODUCE", "CLEANING", "DAIRY"),
    max_stores=10,
    context_length=30,
    horizon=7,
    epochs=6,
)
run_dir, results, dataset = run_training_pipeline(cfg, output_dir=Path("artifacts"))
metrics = load_run_metrics(run_dir)
plot_training_curves(metrics, run_dir)
```
### Inspect saved metrics/plots
The shipped run under `artifacts/run_20251215_212247/` already contains:
- `lstm_metrics.json` / `gru_metrics.json` with history, normalized & sales-space
  metrics, and breakdowns (store/family/holiday/promotion).
- `training_curves.png`, `final_metrics_comparison.png`, `sales_metrics.png`,
  and eight breakdown plots along with `*_test_predictions.png` overlays.
These assets are copied directly into the final PR/video so reviewers can verify
results without re-running training.

## What’s Implemented

- **Data pipeline** – merges transactions, holiday calendars, promotion flags,
  cyclical encodings, and entity embeddings. Synthetic fallback keeps tests
  deterministic.
- **Model zoo** – Flax-based LSTM and GRU forecasters with Adam optimizer,
  mini-batching, validation tracking, and best-checkpoint selection.
- **Evaluation** – normalized vs sales-space metrics plus grouped diagnostics by
  store, product family, holiday, and promotion status.
- **Artifacts** – automatic run directory creation, parameter serialization,
  inference helper, and Matplotlib figure factories for all documentation plots.
- **Documentation** – API/example markdowns, notebooks, and architecture notes
  that map directly to the video storyline.


