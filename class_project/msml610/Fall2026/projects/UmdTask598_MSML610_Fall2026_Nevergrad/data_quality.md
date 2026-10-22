# Data Quality Note

- This note describes the two datasets returned by
  `nvutils.load_returns(data_source)`:
  - `"etf"`: the nine-ETF basket from Yahoo Finance (primary)
  - `"french"`: the Ken French 10 Industry Portfolios (fallback if Yahoo blocks
    requests)
- All numbers below come from the cached files, which are the source of truth for
  every result in the notebooks

## ETF Basket (Primary)

### Source

- **Provider**: Yahoo Finance via `yfinance`, with `auto_adjust=True`, so `Close`
  is adjusted for splits and dividends
- **Tickers**: SPY, EFA, EEM, TLT, IEF, LQD, GLD, VNQ, DBC
- **Requested window**: 2006-01-01 to 2026-01-01, end date excluded
- **Cache**: `data/etf_prices.csv` (git-ignored)
- **Provenance**: fetch date and SHA-256 checksum in `data_provenance.json`

### Coverage and Cleaning

- Yahoo returned 5,031 trading days, from 2006-01-03 to 2025-12-31
- 23 rows were dropped, covering 2006-01-03 to 2006-02-03
  - Every dropped row is a day before DBC's first price on 2006-02-06
  - No other row had a missing value, so after 2006-02-06 all nine tickers have a
    price on every trading day
- Clean prices: 5,008 rows, from 2006-02-06 to 2025-12-31
- Daily simple returns: 5,007 rows, from 2006-02-07 to 2025-12-31
- The data spans 20 calendar years, but 2006 is partial (227 trading days,
  starting in February)
  - The first walk-forward training window is therefore about 4.9 years long

### Checks

- `check_returns()` runs on every load and stops with an error unless:
  - No value is missing
  - Each date appears once, in increasing order
  - No absolute daily return exceeds 25%
- The 25% limit was set by looking at the data:
  - The largest move is EEM +22.8% on 2008-10-13, a real crisis-era move
  - A move above 25% in these ETFs is more likely a data error, such as an
    unadjusted split

### Return Extremes

- The largest daily moves cluster in the fall 2008 crisis and the March 2020 COVID
  crash, matching known market events:

  | Ticker | Min (%) | Min Date   | Max (%) | Max Date   |
  | :----- | :------ | :--------- | :------ | :--------- |
  | SPY    | -10.9   | 2020-03-16 | 14.5    | 2008-10-13 |
  | EFA    | -11.2   | 2008-09-29 | 15.9    | 2008-10-13 |
  | EEM    | -16.2   | 2008-10-15 | 22.8    | 2008-10-13 |
  | TLT    | -6.7    | 2020-03-17 | 7.5     | 2020-03-20 |
  | IEF    | -2.5    | 2020-03-17 | 3.4     | 2009-03-18 |
  | LQD    | -9.1    | 2008-09-29 | 9.8     | 2008-09-30 |
  | GLD    | -8.8    | 2013-04-15 | 11.3    | 2008-09-17 |
  | VNQ    | -19.5   | 2008-12-01 | 17.0    | 2008-11-24 |
  | DBC    | -7.9    | 2022-03-09 | 6.9     | 2008-11-24 |

### Reproducibility

- Yahoo restates its adjusted history, even within the same day
  - Two downloads of the same window about 35 minutes apart on 2026-10-22
    differed in every row, by up to about 2 parts per million
  - The difference is negligible for results (e.g., SPY's annualized volatility
    moved by about 2e-7), but the files had different checksums
- A fresh download therefore never reproduces the cached file byte for byte
- `load_data()` reads the cache and refuses to run if its checksum does not match
  `data_provenance.json`
  - To re-download, delete the cache; the next load writes a new provenance record

## Ken French 10 Industry Portfolios (Fallback)

### Source

- **Provider**: [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html),
  file `10_Industry_Portfolios_daily_CSV.zip`
- **Cache**: `data/10_Industry_Portfolios_daily_CSV.zip`, kept exactly as
  downloaded (git-ignored)
- **Provenance**: fetch date and SHA-256 checksum in `data_provenance_french.json`
- The zip is a static file: two downloads on 2026-10-22 had identical checksums
  - It changes only when the library is updated; the CSV's first line names the
    CRSP database vintage it was built from

### Parsing

- Only the "Average Value Weighted Returns -- Daily" section is used; the
  equal-weighted section is ignored
- `YYYYMMDD` integers become dates, and percent values become decimals
- The file's missing-value codes (-99.99 and -999) become missing values; none
  occur in the window used
- The data is cut to the same window as the ETF basket: from 2006-01-01, with
  2026-01-01 excluded

### Coverage

- 5,031 rows of daily returns for 10 industries, from 2006-01-03 to 2025-12-31,
  with no missing values
  - It starts a month earlier than the ETF returns because no fund launch date
    limits it
- The largest absolute daily move is Energy -19.7% on 2020-03-09, within the 25%
  limit
- It is not comparable to the ETF basket: it covers US equities only, with no
  bonds, gold, real estate, or commodities
  - Use it only when Yahoo is unavailable, and select it explicitly with
    `load_returns("french")`
