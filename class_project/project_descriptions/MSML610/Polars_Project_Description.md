# Description

Polars is a fast DataFrame library written in Rust, with a Python API built on
expressions, lazy evaluation, and parallel execution. It solves the problem of slow
and memory-hungry data manipulation on datasets that are large for pandas. It is
worth a 60-minute tutorial because moving from pandas to Polars changes how a query
is expressed (expressions and lazy plans instead of eager row operations), and the
gain can be measured.

## Technologies Used

Polars

- High performance with parallel execution and lazy evaluation
- Support for various data formats, including CSV, Parquet, and JSON
- Powerful query capabilities with SQL-like syntax for data transformation
- Memory-efficient operations, making it suitable for handling large datasets

# Tutorial

- Implement the tutorial "Learn Polars in 60 mins", following
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Build it with `.claude/skills/tutorial_in_60_mins.create/SKILL.md`
  - Follow the workflow in `tutorials/README.gp.md` and the quality principles in
    `tutorials/tutorials_checklist.md`
- Check the previous tutorials and projects, listed in the section
  `Existing Tutorials and Projects` of
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - No earlier tutorial or project uses Polars, so imitate the reference tutorials
    listed there
- Create the project dir following the class instructions in
  `class_project/README.md`, section `Contribution to the Repo`
  - Start from `class_project/project_template`
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Compare with pandas from the functional point of view
- Deliverables:
  - `polars_utils.py`
  - `polars.API.ipynb`
  - `polars.example.ipynb`

# Project

## Project 1 (Fall2026): Pandas vs Polars Benchmark on Crypto Trade Data

- **Project Objective**: Create a benchmark of typical market-data workloads, from
  trades to OHLCV bars and rolling volatility, and evaluate pandas and Polars on it
  by run time, peak memory, and scaling with data size
- **Dataset Suggestions**: [Binance Public Data](https://data.binance.vision/)
  - Use the daily spot aggregated trades of `ETHUSDT`, in the path
    `data/spot/daily/aggTrades/ETHUSDT/`, for 1, 7, and 28 days to create three data
    sizes
  - Use the monthly 1-minute klines of `ETHUSDT`, in the path
    `data/spot/monthly/klines/ETHUSDT/1m/`, as the reference to check the bars
- **Tasks**:
  - **Ingest the Data**: Download the zipped CSV files (without a header, so name the
    columns), load them with `pd.read_csv` and `pl.read_csv`, write them with
    `write_parquet`, and compare the load times of CSV and Parquet
  - **Define the Benchmark**: Specify 6-8 workloads (filter by trade size, group-by
    aggregation to 1-minute OHLCV bars, rolling realized volatility of the bar
    returns, as-of join of each trade to the last completed bar, sort, timestamp
    parsing) and the three data sizes
  - **Implement the Workloads**: Write each workload in pandas, in Polars eager mode,
    and in Polars lazy mode (`scan_parquet` and `collect`) with `group_by_dynamic`,
    `rolling_std`, and `join_asof`, express one workload with `pl.SQLContext`, and
    check that the results are equal and that the bars match the Binance klines
  - **Measure the Performance**: Record the median wall-clock time over repeated
    runs and the peak memory of every workload, engine, and data size, and compute
    the speedup over pandas
  - **Visualize the Results**: Plot the run time per workload and the run time as a
    function of the data size, and inspect a lazy query plan with `LazyFrame.explain`
- **Bonus Ideas (Optional)**: Add DuckDB as a third engine; run a workload on data
  larger than memory with the Polars streaming engine

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Ingest the Data
  - Result: project dir created and container running, and the three data sizes
    loaded in both pandas and Polars, from CSV and Parquet, with their load times in
    a table
- Milestone 2: API notebook
  - Project tasks: Implement the Workloads
  - Result: `polars.API.ipynb` covering expressions, contexts (`select`, `filter`,
    `group_by`, `with_columns`), `group_by_dynamic`, rolling expressions,
    `join_asof`, `pl.SQLContext`, and lazy evaluation, each next to the equivalent
    pandas code
- Milestone 3: Example notebook
  - Project tasks: Define the Benchmark, Implement the Workloads, Measure the
    Performance, Visualize the Results
  - Result: `polars.example.ipynb` running end to end, with the benchmark harness
    in `polars_utils.py`

## Project 2: Momentum Portfolios for 500 Stocks

- **Project Objective**: Build monthly momentum portfolios of the S&P 500 stocks with
  Polars window expressions, and compare the top-quintile and long-short portfolios
  with the equal-weight market by Sharpe ratio and maximum drawdown, without
  look-ahead
- **Dataset Suggestions**:
  - [S&P 500 Companies](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
    for the tickers
  - Daily adjusted prices for 2014-2024 from
    [`yfinance`](https://github.com/ranaroussi/yfinance)
  - Today's constituents introduce survivorship bias, so state this limit in the
    analysis
- **Tasks**:
  - **Ingest the Data**: Download the daily adjusted close of the tickers with
    `yfinance`, store it as one long table in Parquet, and read it with
    `pl.scan_parquet`
  - **Engineer Features**: Use expressions with `over("ticker")` to compute the daily
    returns, the 12-month momentum that skips the last month, the 63-day volatility
    with `rolling_std`, and the drawdown from the running maximum with `cum_max`
  - **Build the Portfolios**: Take the last trading day of each month with
    `group_by_dynamic`, rank the tickers into quintiles by momentum, and hold each
    quintile with equal weights over the next month, so the signal uses only past
    prices
  - **Evaluate the Portfolios**: Compute the annualized return, volatility, Sharpe
    ratio, and maximum drawdown of the top quintile, the long-short portfolio, and
    the equal-weight market baseline for 2014-2018 and 2019-2024 separately, and
    check the monthly summary with `pl.SQLContext` and with pandas
  - **Visualize the Results**: Plot the cumulative returns and the drawdowns of the
    portfolios, and the average monthly return by quintile
- **Bonus Ideas (Optional)**: Add a low-volatility portfolio from the 63-day
  volatility and compare it with momentum; time the feature pipeline in pandas and in
  Polars lazy mode

## Project 3: Forecasting GDP Growth with World Bank and Commodity Data

- **Project Objective**: Forecast the real GDP growth of countries one year ahead from
  lagged macro indicators and a global commodity price index, and compare the model
  with the previous-year-growth baseline by RMSE in an expanding-window evaluation
- **Dataset Suggestions**:
  - [World Development Indicators (bulk CSV)](https://databank.worldbank.org/data/download/WDI_CSV.zip),
    with the files `WDICSV.csv` and `WDICountry.csv`
    - Use the indicators `NY.GDP.MKTP.KD.ZG` (GDP growth), `FP.CPI.TOTL.ZG`
      (inflation), `NE.TRD.GNFS.ZS` (trade), `NE.GDI.FTOT.ZS` (investment),
      `BN.CAB.XOKA.GD.ZS` (current account), and `FS.AST.PRVT.GD.ZS` (private
      credit)
  - [Global Price Index of All Commodities](https://fred.stlouisfed.org/series/PALLFNFINDEXM)
    from FRED, monthly, downloaded as CSV from
    `https://fred.stlouisfed.org/graph/fredgraph.csv?id=PALLFNFINDEXM`
- **Tasks**:
  - **Ingest the Data**: Scan `WDICSV.csv` lazily with `pl.scan_csv`, filter the six
    indicator codes, `unpivot` the year columns to rows, and write the result to
    Parquet
  - **Merge the Data**: Join with `WDICountry.csv` to drop the regional aggregates
    and add the income group, and join the yearly change of the commodity index,
    computed from the monthly series with `group_by_dynamic`
  - **Engineer Features**: `pivot` the indicators to one row per country and year,
    and build the lags with `shift` and `over("country")`, so the features of year
    `t-1` explain the growth of year `t`
  - **Forecast Growth**: For each test year from 2010 to 2023, fit a `Ridge`
    regression from scikit-learn on the earlier years only, using the arrays
    converted from the Polars frame
  - **Evaluate the Forecasts**: Compute RMSE and MAE against the previous-year-growth
    baseline by year and by income group, give a bootstrap confidence interval of
    the RMSE difference over countries, and repeat the evaluation without 2020
- **Bonus Ideas (Optional)**: Add a gradient boosting model and compare it with
  `Ridge`; measure the run time and the peak memory of the lazy pipeline against
  pandas
