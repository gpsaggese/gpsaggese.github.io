# Description

DuckDB is an embedded, in-process analytical SQL database with a columnar, vectorized
engine, often described as "SQLite for analytics". It solves the problem of running
fast SQL on files of many gigabytes, e.g., CSV and Parquet, on a laptop without
installing or operating a server. It is worth a 60-minute tutorial because it needs
one `pip install`, it queries data frames and files in place, and its speed against
pandas can be measured.

## Technologies Used

DuckDB

- In-process columnar SQL engine, from `duckdb.connect` and `duckdb.sql`
- Direct queries on CSV, Parquet, and JSON files, including glob patterns
- Zero-copy exchange with pandas, Polars, and Arrow data frames
- Window functions, `SUMMARIZE`, `EXPLAIN ANALYZE`, and export with `COPY`
- Persistent database files, and control of memory and threads with `SET`

# Tutorial

- Implement the tutorial "Learn DuckDB in 60 mins", following
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Build it with `.claude/skills/tutorial_in_60_mins.create/SKILL.md`
  - Follow the workflow in `tutorials/README.gp.md` and the quality principles in
    `tutorials/tutorials_checklist.md`
- Check the previous tutorials and projects, listed in the section
  `Existing Tutorials and Projects` of `.claude/skills/tutorial_in_60_mins.rules.md`
  - No earlier tutorial or project uses DuckDB, so read the closest data processing
    work
  - Read the notebooks of `data605/tutorials/tutorial_pandas/` and
    `data605/tutorials/tutorial_parquet/` for the baseline
  - Read the `README.md` of the ClickHouse projects for the columnar analytics view,
    and reuse what is good
    - `class_project/data605/Spring2025/projects/TutorTask198_Spring2025_Real-time_Bitcoin_Data_Analysis_using_ClickHouse/`
    - `class_project/data605/Spring2026/projects/UmdTask382_DATA605_Spring2026_Clickhouse_user_engagement_prediction/`
  - Read `class_project/project_descriptions/MSML610/Polars_Project_Description.md`,
    since a Polars tutorial is built in the same session
- Create the project dir following the class instructions in
  `class_project/README.md`, section `Contribution to the Repo`
  - Start from `class_project/project_template`
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Compare briefly with pandas and Polars from the query performance point of view,
  e.g., time and peak memory for the same aggregation
- Deliverables:
  - `duckdb_utils.py`
  - `duckdb.API.ipynb`
  - `duckdb.example.ipynb`

# Project

## Project 1 (Fall2026): Rolling Risk of 500 Stocks with Window Functions

- **Project Objective**: Compute rolling risk metrics for the stocks of the S&P 500
  with SQL window functions, and compare the file formats and the engines on the same
  queries
- **Dataset Suggestions**:
  - [yfinance](https://github.com/ranaroussi/yfinance) daily prices for 2015-2024 of
    the members listed on
    [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
  - [FRED VIX close](https://fred.stlouisfed.org/series/VIXCLS)
- **Tasks**:
  - **Ingest the Prices**: Download the daily bars with yfinance, write one CSV file
    per ticker, read them all with a glob pattern in `read_csv`, and convert them to
    one Parquet file with `COPY`
  - **Compute the Returns**: Compute the daily returns with `LAG` partitioned by
    ticker, and check them against the pandas `pct_change` with a maximum absolute
    difference below 1e-9
  - **Compute the Rolling Risk**: Compute the 21-day annualized volatility and the
    drawdown from the running maximum with window frames
    `ROWS BETWEEN ... PRECEDING`, and rank the stocks by monthly volatility with
    `RANK() OVER`
  - **Join the VIX without Look-Ahead**: Attach the VIX close of the previous trading
    day to each date with an `ASOF JOIN` on a strict `>`, and compute the mean
    absolute return by VIX bucket
  - **Compare the Formats and Engines**: Time the heaviest query on the CSV files,
    the Parquet file, and a native `.duckdb` table, query the pandas and Polars data
    frames in place, and report the file sizes
  - **Visualize the Risk**: Plot a heatmap of the median drawdown by sector and year,
    and state the survivorship bias of using today's members
- **Bonus Ideas (Optional)**: Estimate the beta of each stock against `SPY` over a
  rolling window with `REGR_SLOPE`

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Ingest the Prices
  - Result: project dir created and container running, and the Parquet file of the
    daily prices of the S&P 500 stocks, with a table of the rows and the date range
    per ticker
- Milestone 2: API notebook
  - Project tasks: Compute the Returns, Compute the Rolling Risk, Join the VIX
    without Look-Ahead, Compare the Formats and Engines
  - Result: `duckdb.API.ipynb` covering `connect`, `read_csv` with glob patterns,
    queries on CSV and Parquet files, `COPY`, a native `.duckdb` table, the exchange
    with pandas and Polars, window functions (`LAG`, window frames, and
    `RANK() OVER`), and `ASOF JOIN`
- Milestone 3: Example notebook
  - Project tasks: Compute the Returns, Compute the Rolling Risk, Join the VIX
    without Look-Ahead, Compare the Formats and Engines, Visualize the Risk
  - Result: `duckdb.example.ipynb` running end to end

## Project 2: SQL Analytics on Mortgage Lending Data

- **Project Objective**: Answer lending questions about mortgage applications with SQL
  directly on a CSV file, and measure the gain over pandas
- **Dataset Suggestions**:
  [HMDA Data Browser](https://ffiec.cfpb.gov/data-browser/), using the 2022 loan
  application register of one state, e.g., Maryland (about 320,000 applications), from
  `https://ffiec.cfpb.gov/v2/data-browser-api/view/csv?states=MD&years=2022`
- **Tasks**:
  - **Load the Data**: Query the CSV file in place with `duckdb.sql`, and report the
    rows, the types, and the missing values with `SUMMARIZE`
  - **Define the Questions**: Write five lending questions, e.g., the origination and
    denial rate by income band, and the most frequent denial reasons
  - **Write the Queries**: Answer the questions with `GROUP BY`, `CASE`, and window
    functions, e.g., the top five lenders per county by originated volume with
    `ROW_NUMBER() OVER`, using `TRY_CAST` for the `NA` and `Exempt` values
  - **Benchmark against pandas**: Run the same five queries in pandas, check that the
    results match, and compare the time and the peak memory
  - **Export the Results**: Save the aggregates in a `.duckdb` file and as Parquet
    with `COPY`
  - **Visualize the Results**: Plot the denial rate by income band and the loan amount
    by loan purpose from the query outputs, describing the pattern without causal
    claims
- **Bonus Ideas (Optional)**: Read the file over HTTP with the `httpfs` extension;
  repeat the denial rate query on the 2021 file and compare

## Project 3: Scaling and Resource Limits on SEC Filings

- **Project Objective**: Measure how DuckDB scales with the data size, the threads, and
  the memory limit on SEC financial statements, and find where pandas stops working
- **Dataset Suggestions**:
  [SEC Financial Statement Data Sets](https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets),
  using the eight quarterly files of 2023 and 2024 (about 1 GB zipped); the SEC asks
  for a `User-Agent` header with contact information on downloads
- **Tasks**:
  - **Assemble the Dataset**: Download the eight zip files, extract `sub.txt` and
    `num.txt`, and query them as two tables with glob patterns in `read_csv`
  - **Profile the Query Plan**: Use `EXPLAIN ANALYZE` on three queries, i.e., a
    group-by on `tag`, a join of `num.txt` with `sub.txt` on `adsh` for the median
    `NetIncomeLoss` of 10-K filers by `sic` group, and a `RANK() OVER` of the filers
    by `Assets`, and name the operator that dominates the time of each
  - **Measure the Scaling**: Time the group-by and the join on 1, 2, 4, and 8 quarters,
    with five repetitions each
  - **Limit the Resources**: Repeat the group-by with a `memory_limit` of 1, 2, and 4
    GB and with 1, 2, 4, and 8 threads, and record the time and whether it spills to
    disk
  - **Compare with pandas**: Run the group-by in pandas on the same quarters, reading
    only the needed columns, and note where it slows down or runs out of memory
  - **Report the Trade-Offs**: Plot the time vs. the data size and vs. the number of
    threads, with the standard deviation over the repetitions
- **Bonus Ideas (Optional)**: Test the effect of sorting the Parquet files by `tag` on
  the time of a filter query for `NetIncomeLoss`
