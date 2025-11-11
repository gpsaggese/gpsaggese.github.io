# Retail Sales Forecasting with LSTMs — Project Workspace

This directory tracks the MSML610 Fall 2025 class project focused on multi-store,
multi-product retail sales forecasting using recurrent neural networks in JAX.
The goal for the midterm PR is to demonstrate working scaffolding **with runnable
code** that trains a JAX LSTM on a synthetic replica of the Kaggle dataset while
adhering to the required project structure.

## Directory Layout

- `retail_sales_forecasting_with_lstms.API.*`: interface-first documentation,
  notebook, and helper module describing the reusable forecasting API surface.
- `retail_sales_forecasting_with_lstms.example.*`: end-to-end tutorial material
  showing how to pull data, train the JAX LSTM/GRU models, and evaluate results.
- `retail_sales_forecasting_utils.py`: shared utility functions (data loading,
  feature engineering, model builders) imported by the notebooks.
- `docker_simple/`: the DATA605-style container scripts for development. These
  will be customized with JAX, pandas, and plotting dependencies during
  implementation.
- `docker_causify_dev_system/`: placeholder for the thin-environment setup if we
  migrate to the advanced docker workflow later in the semester.

## Running the Project (Simple Docker)

```bash
cd /Users/mns/Documents/umd_classes/class_project/MSML610/Fall2025/projects/UmdTask77_Retail_Sales_Forecasting_with_LSTMs/docker_simple
bash docker_build.sh   # builds the Jupyter-ready image with JAX/Flax dependencies
bash docker_jupyter.sh # launches JupyterLab with the project mounted at /app/project
```

The Dockerfile already bundles CPU-enabled JAX, Flax, Optax, pandas/polars,
scikit-learn, and plotting libraries. If you have an NVIDIA GPU available add
the appropriate `jaxlib` wheel before rebuilding.

## Midterm PR Deliverables (In Progress)

1. **Planning & Scope** — documented in `retail_sales_forecasting_with_lstms.API.md`
   and `retail_sales_forecasting_with_lstms.example.md`.
2. **Executable Notebooks** — run end-to-end using the synthetic dataset, including
   feature engineering, training, metrics, and visualizations.
3. **Utility Module** — delivers reusable functions for data ingestion, feature
   creation, LSTM/GRU modeling, training, and evaluation.
4. **Docker Scripts** — customized names plus dependency set aligned with the project.

## Next Milestone Checklist

- Integrate the Kaggle Store Sales dataset ingestion pipeline with holiday and
  promotion feature encoding.
- Add store/family-level aggregation metrics and baseline comparisons.
- Expand exploratory plots (holiday overlays, per-store drill-downs).
- Package trained parameters and scalers for reuse outside the notebooks.

Please do not merge this branch without TA approval. Branch and folder naming
follow the `UmdTask77_Retail_Sales_Forecasting_with_LSTMs` convention required by
the course instructions.
