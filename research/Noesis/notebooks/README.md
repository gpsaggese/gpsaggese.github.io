# In-Memory Market Demonstration

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

# Prompt Tokens Billed per Character

- `prompt_tokens.ipynb`: fits the prompt tokens that the seed sellers billed in
  the run of `scripts/experiment_prompt_tokens.py` (#720), and plots them
- `prompt_tokens.py`: Jupytext pair; edit either source and sync with
  `jupytext --sync research/Noesis/notebooks/prompt_tokens.ipynb`
- `prompt_tokens_utils.py`: the fit, the tables, and the plot, one function per
  cell
- Reads `data/prompt_tokens/prompt_tokens.raw.jsonl`; makes no calls
- Run it top to bottom from the repository root:

  ```bash
  > python research/Noesis/notebooks/prompt_tokens.py
  ```
