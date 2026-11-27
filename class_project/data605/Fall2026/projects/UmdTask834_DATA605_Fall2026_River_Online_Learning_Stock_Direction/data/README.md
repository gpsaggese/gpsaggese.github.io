# Data

This directory holds the cached market data. It is not committed: the files
are created on the first run of `river.example.ipynb`, which calls
`river_utils.download_ohlcv`.

## Source
- Yahoo Finance daily bars, downloaded with the `yfinance` package
- Fields: open, high, low, close, split/dividend-adjusted close, volume
- Universe: `river_utils.DEFAULT_TICKERS` (SPY plus seven large caps from
  different sectors)
- Period: `river_utils.DEFAULT_START` to `river_utils.DEFAULT_END`

## Files created
- `ohlcv_<tickers>_<start>_<end>.parquet`: long-format bars, one row per
  `(date, ticker)`
- `ohlcv_<tickers>_<start>_<end>.json`: download time, `yfinance` version and
  row count, so a run can be traced back to the exact snapshot

## Why the data is cached
Yahoo revises historical data (e.g., dividend adjustments), so downloading
again later can change results slightly. The cache freezes one snapshot; pass
`force=True` to refresh it.

## Why other suggested sources are not used
- Alpha Vantage: requires an API key and the free tier has a small daily
  request limit, which makes a reproducible Docker run fragile
- Quandl (now Nasdaq Data Link): its free end-of-day US equity dataset (WIKI)
  stopped updating in 2018
- Kaggle stock dataset: a static snapshot that requires a Kaggle login
