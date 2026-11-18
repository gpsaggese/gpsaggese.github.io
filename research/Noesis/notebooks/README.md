# Noesis notebooks

## Model-drift analysis and figures

- `model_drift_analysis.ipynb` and `model_drift_analysis.py` reproduce the
  frozen capability-versus-Hamming-MMD comparison and write the derived
  summary, table, and findings report under `docs/findings`
- `model_drift_figures.ipynb` and `model_drift_figures.py` regenerate the five
  paper figures as PNG and SVG files from the checked-in summaries
- Neither notebook makes provider calls; the expensive live collection and
  permutation-test outputs are treated as frozen inputs

From the repository root, sync and execute the notebooks with:

```bash
jupytext --sync research/Noesis/notebooks/model_drift_analysis.ipynb
jupytext --sync research/Noesis/notebooks/model_drift_figures.ipynb
PYTHONPATH="$PWD/helpers_root:$PWD" jupyter nbconvert \
  --execute --to html --ExecutePreprocessor.timeout=120 \
  --output-dir=/tmp/noesis-model-drift \
  research/Noesis/notebooks/model_drift_analysis.ipynb \
  research/Noesis/notebooks/model_drift_figures.ipynb
```

Keep executed outputs and generated HTML out of the PR. The paired notebook
sources contain no saved execution outputs.

## In-memory market demonstration

- `market_in_memory.ipynb`: five seeded scenarios with input tables, auction
  results, per-tier curves, contracts, and unmatched orders
- `market_in_memory.py`: Jupytext pair; edit either source and sync with
  `jupytext --sync research/Noesis/notebooks/market_in_memory.ipynb`
- `market_in_memory_utils.py`: in-memory submission, mocked fulfillment, and
  presentation using M10, M11, and M12 helpers
- All delivery outcomes are mocked; no providers, payments, or databases are used

## Run Locally

- From the repository root, use a Python environment with Noesis dependencies,
  `nbconvert`, `ipykernel`, and `jupytext` already installed
- Execute every cell and produce an HTML report:

  ```bash
  PYTHONPATH="$PWD/helpers_root:$PWD" jupyter nbconvert \
    --execute --to html --ExecutePreprocessor.timeout=60 \
    --output-dir=/tmp/noesis-m13 \
    research/Noesis/notebooks/market_in_memory.ipynb
  ```

- Open `/tmp/noesis-m13/market_in_memory.html`; change `seed` in the notebook
  to explore different values with the same guaranteed regimes
- Keep executed outputs and generated HTML out of the PR; the paired sources
  contain no saved execution outputs
