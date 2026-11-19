# Data Sources

Where each dataset in this project comes from and how it is made. The code
caches every download under `data/` (not committed), so each notebook runs
from scratch on a new machine, except where noted.

- Shared copies: `data` folder in the
  [project Drive folder](https://drive.google.com/drive/folders/1m7U-AzT-frDAFdCXyhcTsxOO1d78u59T)

## Datasets

| Data | Source | Range | Made by | Used in |
| :--- | :----- | :---- | :------ | :------ |
| BTCUSDT, ETHUSDT 1h and 1d close | Binance public API `data-api.binance.vision` | 2020-01-01 to 2026-10-01 | `tradability_utils.load_prices(source="binance")` | `exercise1.ipynb`, `exercise1_spy.ipynb` |
| BTCUSDT, ETHUSDT 5-minute OHLCV | Binance monthly files on `data.binance.vision` | 2020-01-01 to 2026-10-01 | `data_utils.load_ohlcv()`, cached as parquet | `tradability.data.ipynb`, `tradability.models.ipynb` |
| SPY 1h close, adjusted | Yahoo Finance via `yfinance` | 2024-11-01 to 2026-10-01 | `tradability_utils.load_prices(source="yahoo")` | `exercise1_spy.ipynb` |
| SPY 1d close, adjusted | Yahoo Finance via `yfinance` | 2020-01-01 and 1993-01-01 to 2026-10-01 | `tradability_utils.load_prices(source="yahoo")` | `exercise1_spy.ipynb` |

## Notes

- Binance
  - `api.binance.com` returns HTTP 451 from the US, so the code uses the
    public endpoints, which serve the same data (`D1`)
  - Each monthly file is checked against the SHA-256 checksum that Binance
    publishes next to it (`.CHECKSUM`)
  - The monthly files use millisecond timestamps before 2025 and microsecond
    timestamps from 2025
  - The 5-minute data has 15 gaps per coin (Binance outages). See
    `data_utils.find_gaps()` and #904
- Yahoo Finance
  - Prices are adjusted for dividends and splits (`auto_adjust=True`)
  - Yahoo only keeps the last 730 days of hourly bars, so the SPY 1h file
    can no longer be downloaded for 2024-11-01. Copy
    `SPY_1h_2024-11-01_2026-10-01.csv` from the Drive `data` folder into
    `data/` before running `exercise1_spy.ipynb`
