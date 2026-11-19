# MSML610: Price-only PatchTST-style benchmark (v0.5)

## Research question
Does a patch-based Transformer improve next-observed-date WTI, Brent, or Henry Hub **log-return RMSE** over our already measured zero/mean, Ridge, ElasticNet, XGBoost, LightGBM, LSTM, GRU and TCN baselines?

## Architecture and attribution

This is a **compact supervised PatchTST-style adaptation**, *not* the official authors' source code, pretrained weights, or a faithful reproduction of the entire original PatchTST architecture.

Its PatchTST-inspired mechanisms are: temporal patching of a univariate return history, learned positional embeddings, multihead Transformer encoder, a flattened regression head, and per-window normalization. Two additional past-only features (momentum_5, volatility_20) are appended at the output head for parity with the course's numerical baselines.

Reference: Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*, ICLR 2023, https://arxiv.org/abs/2211.14730 .

This adaptation forecasts one future log return instead of the original paper's general long-horizon setting. Do not describe it as "official PatchTST" in the report. For exact architectural reproduction, replace this adaptation with the original authors' or Nixtla's implementation and rerun the same protocol.

## Inputs, targets and methodology

- Historical input: 30 log returns **strictly before** the date being predicted, ordered oldest -> newest. Includes 5-period momentum and 20-period volatility, both past-only.
- Output: `log(P_t / P_previous_valid_quote)` for a valid FRED target price observation.
- Missing quotes are removed without inventing prices. Nonpositive prices invalidate surrounding log returns.
- Separate model for WTI `DCOILWTICO`, Brent `DCOILBRENTEU`, and Henry Hub `DHHNGSP`.
- Chronological train 65%, validation 15%, test 20%, with five-observation gaps. Same eligible rows as the existing v0.4 leaderboard.
- Training-only scalers for inputs and outputs, 2 candidate settings, early stopping on validation MSE, then a new model fitted on train+validation for the chosen epoch count. Never fit scalers, tune or pretrain on test data.
- Fixed seed 42. Results vary by random seed; repeat seeds after the first exploratory run.
- Important: FRED observation date != guaranteed real-time price publication date. This is retrospective evaluation, not a ready-to-trade next-day forecasting service.

## Hyperparameter search (initial bounded search, NOT universally optimal)

| Setting | Candidate 1 | Candidate 2 |
|---|---|---|
| Context observations | 30 | 30 |
| Patch length, stride | 6, 3 | 6, 3 |
| Transformer layers, attention heads | 2, 4 | 2, 4 |
| Hidden dimension | 32 | 64 |
| Dropout | 0.10 | 0.10 |
| AdamW learning rate | 0.001 | 0.0003 |
| Weight decay | 0.001 | 0.001 |
| Batch size | 64 | 64 |
| Max epochs | 40 | 40 |
| Early stopping patience | 5 | 5 |

The code writes the selected configuration and epoch count to `results/price_only_patchtst/{asset}_model_config.json`.

## Install and test in WSL

From the top-level CLIP project directory with the existing `.venv` activated, extract this archive **into** `price_only_benchmarks/`. Back up an earlier modified `benchmark_models.py` first if necessary.

```bash
python -m pip install torch
PYTHONPATH=price_only_benchmarks python -m pytest -q --noconftest \
  price_only_benchmarks/test/test_patchtst.py \
  price_only_benchmarks/test/test_benchmark_models.py
```

## Train on all three FRED datasets

```bash
mkdir -p results/price_only_patchtst
set -o pipefail
python price_only_benchmarks/benchmark_models.py \
  --period full --models patchtst \
  --output-dir results/price_only_patchtst \
  2>&1 | tee results/price_only_patchtst/run.log
```

**DO NOT use `--quick` for research results**; quick mode only verifies plumbing with 2 training epochs.

## Merge with our prior results

```bash
python price_only_benchmarks/merge_model_comparisons.py \
 results/price_only_tabular_initial/model_comparison.csv \
 results/price_only_boosted/model_comparison.csv \
 results/price_only_neural/model_comparison.csv \
 results/price_only_patchtst/model_comparison.csv \
 --output results/price_only_leaderboard_with_patchtst.csv
```

The merge script refuses incompatible per-asset test date ranges and sample sizes. Check sample counts and metadata explicitly too.

## Outputs

- `model_comparison.csv`: return RMSE (primary), MAE, OOS R², directional accuracy, implied price RMSE/MAE, test counts/dates.
- `{asset}_model_config.json`: per-commodity train/validation/test sizes, selected hyperparameters and validation loss.
- `{asset}_test_predictions.csv`: paired test target and prediction for confidence intervals and plots.
- `benchmark_metadata.json`: experiment version, seed, split settings and model names.
- `run.log`: terminal record for the professor's research evidence.

No learned weights or large FRED CSVs are committed to Git. Save large outputs to the class Drive and document findings in `progress.md`. One initial code+integration test verification used **synthetic prices only**; actual PatchTST RMSE must come from your WSL run.
