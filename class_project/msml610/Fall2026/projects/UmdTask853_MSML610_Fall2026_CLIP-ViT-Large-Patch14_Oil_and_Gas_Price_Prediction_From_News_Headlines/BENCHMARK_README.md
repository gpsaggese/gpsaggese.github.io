# Frozen CLIP Oil and Natural Gas Forecasting — Research Benchmark (v0.2)

This is a **benchmark implementation**, not a claim of state-of-the-art predictive accuracy. It tests whether frozen CLIP ViT-L/14 text features add next-observation WTI / Henry Hub return information beyond historical price features.

## Research protocol

- **Targets:** next *observed* WTI and Henry Hub log returns; modeled as separate regression problems. Missing FRED quotes (including holidays) are excluded **without forward-filling**, so adjacent valid quotes define a next-observed-session return; zero/negative quotes are retained until return construction and invalidate both touching returns, including April 2020 WTI.
- **Price features:** five current/prior log-return lags, trailing five-observation momentum, trailing 20-observation volatility.
- **News:** energy-keyword filtered ABC Australia headlines. Because the source has **calendar dates but not verifiable publication times**, each headline is assigned to the first price observation **strictly after** publication date. This is deliberately conservative, but still does not validate same-day FRED availability. Benchmark period is limited to the headline source's date coverage.
- **CLIP:** frozen `openai/clip-vit-large-patch14`, inference-only; mean daily normalized headline vectors + 6 zero-shot event similarity features; missing-news days filled with zero and given count/availability indicators. CLIP weights are not fine-tuned.
- **PCA and modeling:** 20-dimensional PCA of CLIP vectors and `StandardScaler` within a `scikit-learn` `Pipeline`/`ColumnTransformer`, ensuring refits are performed **inside** each temporal CV training fold. Three-fold expanding-window CV with five-observation gap tunes Ridge alpha among 0.1, 1, 10, 100, 1000. Train/validation/test splits are 65%/15%/20%, with five omitted forecast-origin rows at each boundary. Models are tuned on training only; validation and test are diagnostic/held-out, not used in tuning.
- **Baselines:** zero return, train mean, price-only Ridge, price + CLIP/PCA Ridge, price + zero-shot event scores Ridge, price + combined CLIP/PCA + event scores Ridge, and a date-shuffled news control. Shuffling is independent within each partition to preserve train/test isolation.
- **Evaluation:** RMSE, OOS R² relative to the original train mean, directional accuracy (undefined for all-zero predictions), paired circular moving-block bootstrap CI (block length 5, 1000 resamples) of `RMSE(price Ridge) - RMSE(CLIP/PCA Ridge)`; positive gains mean CLIP improved RMSE. This CI conditions on already-fitted predictions, not full model-training uncertainty.

**Caution:** This is a daily *spot-price* retrospective experiment, **not** a live trading strategy. Historical FRED observation dates do not prove when EIA-sourced values were published or available. Cross-source time-zone differences and possible historical source revisions require further audit before any causal, intraday, or live-deployment claim.

## Run commands (from this directory)

Download FRED CSVs from the series' official pages, using their CSV export option:

- https://fred.stlouisfed.org/series/DCOILWTICO → `DCOILWTICO.csv`
- https://fred.stlouisfed.org/series/DHHNGSP → `DHHNGSP.csv`

Install benchmark requirements in your existing notebook/container environment:

```bash
python -m pip install numpy pandas scipy scikit-learn pytest matplotlib
```

**Phase A — Price-only benchmarks** (works without news/CLIP weights):

```bash
python run_forecasting.py --wti-csv DCOILWTICO.csv --gas-csv DHHNGSP.csv
```

**Phase B — Headline-enhanced CLIP benchmark** (also install `torch==2.7.1` and `transformers==4.53.2` as used by the current `clip.API.ipynb`):

Download the ABC News Australia dataset from https://www.kaggle.com/datasets/therohk/million-headlines and point to its CSV (`publish_date`, `headline_text`).

```bash
python run_forecasting.py \
  --wti-csv DCOILWTICO.csv \
  --gas-csv DHHNGSP.csv \
  --headlines-csv abcnews-date-text.csv \
  --batch-size 16
```

Results are saved in `results/benchmark/summary.csv`, `*_predictions.csv`, and CLIP bootstrap CI files. Frozen CLIP vectors are cached in `.cache/forecast` under a content-derived key. Exclude `.cache/`, raw licensed datasets, and generated results from Git unless explicitly chosen for lightweight, public result tables.

**Fast checks (no internet / model weights required):**

```bash
PYTHONPATH=. python -m pytest -q test/test_forecast_utils.py test/test_clip_features.py
```

## v0.2 correction after first real FRED evaluation

The first WTI/Henry Hub run incorrectly computed trailing 20-observation volatility on raw FRED calendar rows, where missing holiday/quote values repeatedly reset the 20-consecutive-valid-returns window. It consequently discarded most valid observations (WTI: 2,397/10,654 rows; Henry Hub: 2,024/7,781). **Do not cite those initial RMSE values as validated research findings.** This revision removes absent-quote rows *before* constructing next-observed-session returns, while deliberately keeping zero/negative prices to block invalid cross-negative returns. The change is covered by tests for missing observations and negative-price boundaries. Re-run and check sample sizes/date coverage before interpreting new RMSE values.

This does **not** mean imputing missing prices. A return across a weekend/holiday measures the change to the next actually observed quote. If data gaps are long, a separate elapsed-time audit is warranted.

## Before calling results final

Audit ABC News coverage per day and per split; compare longer event-specific prompts; confirm time alignment for each dataset; add cumulative squared-error plots; investigate semantic novelty as a separately ablated extension; write a running results summary and include instructor feedback. Never claim SOTA based on an untested pipeline or on a positive gain with a CI spanning zero.

**Cumulative-error figure** (after a benchmark run):

```bash
python plot_forecast_errors.py --predictions results/benchmark/wti_predictions.csv --output results/benchmark/wti_test_cumulative_error.png --split test
```
