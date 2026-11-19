# MSML610 — Price-Only Energy Forecasting Benchmarks (v0.4)

**Scope:** Independent daily one-step forecasting of spot-price log returns for FRED WTI (`DCOILWTICO`), Brent (`DCOILBRENTEU`), and Henry Hub (`DHHNGSP`). **NO CLIP, NO FinBERT, NO NEWS.** The result is a rigorous price-only benchmark that future news-enhanced models must beat.

## Why this matters

The goal is not to find a model with the lowest price-level RMSE on a different dataset. It is to compare competing models on the **same target, eligible observations, date splits, and evaluation metrics**. A naïve zero-return forecast can be remarkably competitive. Real-data performance has **not** been measured by the package author, because the FRED CSVs are on the user's WSL machine and this build environment has no outbound access to download them.

## Inputs and target

Download (or use your existing) CSVs from FRED. The script supports `observation_date` or `DATE` as the first column.

```bash
mkdir -p data
curl -fL --retry 3 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILWTICO' -o data/DCOILWTICO.csv
curl -fL --retry 3 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU' -o data/DCOILBRENTEU.csv
curl -fL --retry 3 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=DHHNGSP' -o data/DHHNGSP.csv
```

For each **target date** t, predict `log(P_t / P_previous_quote)` from features ending at the previous **available** price observation. Features are 30 prior returns, sum of five prior returns (momentum), and standard deviation of 20 prior returns (volatility). Missing quote dates are skipped without filling prices. Nonpositive prices (e.g., April 2020 WTI) invalidate the two surrounding returns and the needed history. Model inputs are always strictly older than the target price. The script includes a date audit check and a test.

FRED's **observation date is not the date that FRED published the observation**. Therefore these are retrospective experiments, not yet legitimate live forecasts based on FRED publication times. Log returns at the next observed quote may span weekends or holidays.

## Models

- **Naïve:** zero future return; historical mean return.
- **Linear:** Ridge, ElasticNet (scaling within training folds).
- **Tabular nonlinear:** `HistGradientBoostingRegressor`, XGBoost, LightGBM.
- **Neural:** supervised one-layer LSTM, GRU, and a small 4-layer causal dilated TCN (receptive field ~31 observations), taking a 30-observation return window.
- **Self-supervised:** simple GRU encoder trained with two masked/noisy views using InfoNCE, followed by frozen representation + Ridge. **Not a full TS2Vec implementation.**

### Install

From your project directory, in your existing `(.venv)`:

```bash
python -m pip install numpy pandas scikit-learn scipy pytest xgboost lightgbm torch
python -m pytest -q --noconftest test/test_benchmark_models.py
```

### Recommended first real run: statistical and tree models (CPU)

```bash
python benchmark_models.py \
  --period full \
  --models ridge,elasticnet,histgb,xgboost,lightgbm \
  --output-dir results/price_only_tabular
```

### Recommended second run: neural / contrastive models (Colab GPU or local CPU)

```bash
python benchmark_models.py \
  --period full \
  --models lstm,gru,tcn,contrastive_ridge \
  --output-dir results/price_only_neural
```

In Colab, turn on a T4 runtime, install the packages, and upload/mount these project files plus three FRED CSV files. The script selects CUDA automatically when available. The CPU run also works but will be slower.

**Do not** use `--quick` for reported experimental results; this flag only checks the plumbing with tiny grids and epochs.

### Limit to the historical news-covered period later

When you have the full raw ABC headline CSV with column `publish_date`, rerun the **same price-only models** over the covered period:

```bash
python benchmark_models.py --period abc \
  --abc-csv data/abcnews-date-text.csv \
  --models ridge,elasticnet,histgb,xgboost,lightgbm \
  --output-dir results/price_only_abc
```

This is the price-only period to compare against news-enhanced methods. It is not valid to compare a 2019–2026 price-only test against a 2003–2021 news-model test. This script uses the ABC publication range to determine its historical period; a downstream news experiment must further ensure **identical actual eligible test dates**.

## Model fitting and fairness

- Train 65%, validation 15%, test 20% **chronologically**, with five-observation gaps between main periods.
- Tabular models: `TimeSeriesSplit(n_splits=3, gap=5)` on train for grid search, then refit selected parameters on train + validation before evaluating held-out test.
- LSTM/GRU/TCN: choose from `hidden=16, lr=1e-3` and `hidden=32, lr=3e-4` using the chronological validation window; early stop with patience 5; refit on train + validation using the selected epoch count. AdamW, batch size 64, gradient clip 1.0, seed 42. This is a limited **starting sweep**, not a demonstrated optimum.
- Contrastive model: train encoder on **past training windows**, select Ridge alpha using validation only, then refit encoder/head on train+validation; never pretrain on the held-out test. No labels in encoder pretraining.
- Scalers are learned on the relevant fit partition only. No data leakage from test.
- Differences between temporal CV on train (tabular) and a single chronological validation partition (deep) must be disclosed; future stronger benchmark should use equal rolling-origin folds.
- Different assets have different series lengths, so they have separate chronological dates. **Never compare raw RMSE across commodities** due to different units/volatility.

## Results and reporting

Each run writes to `--output-dir`:

- `model_comparison.csv`: asset, model, test dates/count, **log-return RMSE**, return MAE, out-of-sample R² vs training mean, directional accuracy, **price-level RMSE** and price-level MAE.
- `{asset}_test_predictions.csv`: target dates, previous observed price, true observed price, true log return, predictions for each model and implied predicted prices (`P_previous * exp(pred_return)`).
- `{asset}_model_config.json`: selected hyperparameters, CV/validation errors, data counts, first/last dates.
- `benchmark_metadata.json`: reproducibility and split settings.

**Meaning of metrics:** Smallest RMSE is desirable. Out-of-sample R² > 0 indicates improvement over historical-mean forecast. Directional accuracy is undefined for a constant zero-return predictor; it is displayed as `NaN`. An apparent improvement might be noise, so later add paired moving-block bootstrap CIs and repeat across rolling test windows.

## Planned stronger models (not included in this first runnable baseline)

- Official **PatchTST** (ICLR 2023) and **iTransformer** (ICLR 2024), on the exact same windows/date splits.
- **N-HiTS** (AAAI 2023) for weekly/monthly horizons.
- **Chronos-2 / TimesFM 2.5** zero-shot time-series foundation comparisons, with caveats about training-set overlap and appropriate target/horizon transforms.
- **Bayesian VAR / fundamentals** for monthly Henry Hub as motivated by Baumeister et al., NBER 2024; those require additional inputs.

## Notes

- The script was run end-to-end against **synthetic test datasets**, across all three assets and all nine trainable families, and separate automated tests passed. Synthetic output values are not research results.
- Real benchmark scores must come from running the script against the three actual FRED CSVs in your WSL/Colab environment.
- Do not commit raw CSVs, `.venv`, Colab checkpoint weights, or large generated outputs to the course GitHub repo.
