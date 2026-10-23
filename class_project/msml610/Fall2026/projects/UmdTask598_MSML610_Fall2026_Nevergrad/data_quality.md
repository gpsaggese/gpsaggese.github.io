# Data Quality Note

- This note describes the two datasets returned by
  `nvutils.load_returns(data_source)`:
  - `"etf"`: the nine-ETF basket from Yahoo Finance (primary)
  - `"french"`: the Ken French 10 Industry Portfolios (fallback if Yahoo blocks
    requests)
- It also describes the risk-free rate returned by `nvutils.load_risk_free_rate()`
  and the walk-forward folds built by `nvutils.generate_folds()`
- All numbers below come from the cached files, which are the source of truth for
  every result in the notebooks
- Fetch dates in the provenance files are UTC dates from the container's clock

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

## Risk-Free Rate

### Source

- **Provider**: [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html),
  file `F-F_Research_Data_Factors_CSV.zip` (monthly Fama-French factors)
- **Series**: the `RF` column, the return of the 1-month T-bill in each month
  - It is a return, in the same units as the asset returns
- **Cache**: `data/F-F_Research_Data_Factors_CSV.zip`, kept exactly as downloaded
  (git-ignored)
- **Provenance**: fetch date and SHA-256 checksum in `data_provenance_rf.json`
- The same rate is used with both data sources, so it does not depend on Yahoo
- Like the industry zip, it is a static file: two downloads on 2026-10-22 had
  identical checksums

### Why Monthly, Not Daily

- The daily factors file rounds `RF` to 0.01% per day, about 2.5% per year
  - From 2006 to 2025 it takes only three values: 0.00%, 0.01%, and 0.02% per day
  - Averaged over a 5-year training window, it differs from the monthly file by up
    to 0.23% per year (training window of test year 2023: 1.39% vs 1.15%)
  - It gives exactly 0% for the training windows of test years 2014 to 2018, while
    the monthly file gives 0.02% to 0.20%
- The monthly file rounds to 0.01% per month, about 0.12% per year
- Sharpe ratios need only the average rate over a period (a training window for a
  training Sharpe, a test year for an out-of-sample Sharpe), so a monthly series is
  enough

### Parsing and Coverage

- Only the monthly table is used; the annual table after it is ignored
- `YYYYMM` integers become the first day of the month (e.g., `200601` becomes
  2006-01-01), and percent values become decimals
- 240 months, from 2006-01 to 2025-12, with no missing values
- Monthly returns range from 0.00% to 0.48% (2024-08), with an average of 0.13% per
  month (about 1.6% per year)
  - 72 months are exactly 0.00%: the near-zero-rate periods of 2008 to 2015 and
    2020 to 2022
- `load_risk_free_rate()` stops with an error if a monthly return exceeds 1% (about
  12% per year), a level last seen in the early 1980s
- Every month with ETF returns has a rate
  - January 2006 has a rate but no ETF returns, since DBC starts in February, so
    the first training window has one extra month of rates

## Walk-Forward Folds

### Construction

- `nvutils.generate_folds(returns)` yields one (`train_returns`, `test_returns`)
  pair per fold:
  - **Train**: 5 consecutive calendar years
  - **Test**: the next calendar year
  - Each fold is one year later than the previous one
- Folds follow calendar years, so each test year is one row of the per-year
  results tables
- `nvutils.split_fold()` cuts any date-indexed data by calendar year, so the daily
  returns and the monthly risk-free rate of a fold share the same boundaries
  - Each training window has 60 months of rates; each test year has 12
- The data from 2006 to 2025 gives 15 folds, with test years 2011 to 2025
  - The first training window starts on 2006-02-07, so it is about 4.9 years long

### Checks

- `split_fold()` stops with an error unless training data strictly precedes test
  data, which keeps test-year data out of every estimate
- `get_test_years()` stops with an error if a calendar year between the first and
  last date has no data, since the gap would stretch a training window beyond 5
  years
- `test/test_nevergrad_utils.py` checks that the folds are contiguous: each
  training window ends on the last trading day before its test year starts

### Fold Boundaries

- ETF basket, from `nvutils.get_fold_table(nvutils.load_returns("etf"))`; days are
  trading days with returns:

  | Test Year | Train Start | Train End  | Train Days | Test Start | Test End   | Test Days |
  | :-------- | :---------- | :--------- | :--------- | :--------- | :--------- | :-------- |
  | 2011      | 2006-02-07  | 2010-12-31 | 1,235      | 2011-01-03 | 2011-12-30 | 252       |
  | 2012      | 2007-01-03  | 2011-12-30 | 1,260      | 2012-01-03 | 2012-12-31 | 250       |
  | 2013      | 2008-01-02  | 2012-12-31 | 1,259      | 2013-01-02 | 2013-12-31 | 252       |
  | 2014      | 2009-01-02  | 2013-12-31 | 1,258      | 2014-01-02 | 2014-12-31 | 252       |
  | 2015      | 2010-01-04  | 2014-12-31 | 1,258      | 2015-01-02 | 2015-12-31 | 252       |
  | 2016      | 2011-01-03  | 2015-12-31 | 1,258      | 2016-01-04 | 2016-12-30 | 252       |
  | 2017      | 2012-01-03  | 2016-12-30 | 1,258      | 2017-01-03 | 2017-12-29 | 251       |
  | 2018      | 2013-01-02  | 2017-12-29 | 1,259      | 2018-01-02 | 2018-12-31 | 251       |
  | 2019      | 2014-01-02  | 2018-12-31 | 1,258      | 2019-01-02 | 2019-12-31 | 252       |
  | 2020      | 2015-01-02  | 2019-12-31 | 1,258      | 2020-01-02 | 2020-12-31 | 253       |
  | 2021      | 2016-01-04  | 2020-12-31 | 1,259      | 2021-01-04 | 2021-12-31 | 252       |
  | 2022      | 2017-01-03  | 2021-12-31 | 1,259      | 2022-01-03 | 2022-12-30 | 251       |
  | 2023      | 2018-01-02  | 2022-12-30 | 1,259      | 2023-01-03 | 2023-12-29 | 250       |
  | 2024      | 2019-01-02  | 2023-12-29 | 1,258      | 2024-01-02 | 2024-12-31 | 252       |
  | 2025      | 2020-01-02  | 2024-12-31 | 1,258      | 2025-01-02 | 2025-12-31 | 250       |

- Ken French fallback: the same boundaries, except that the first training window
  starts on 2006-01-03 (1,259 days), since no fund launch date limits it
