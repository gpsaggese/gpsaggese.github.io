# Description

Luigi is a Python package from Spotify to build batch data pipelines out of tasks
that declare their dependencies and their outputs. It solves the problem of running a
chain of jobs in the right order, skipping the steps that are already done, and
resuming after a failure. It is worth a 60-minute tutorial because a complete
ETL-to-model pipeline takes a few small classes, and the idempotence and the
dependency graph can be observed directly in the scheduler.

## Technologies Used

Luigi

- `Task` classes with `requires()`, `output()`, and `run()` to declare a pipeline
- Targets such as `LocalTarget`, which make a task idempotent
- Parameters such as `DateParameter` and `FloatParameter` to parametrize and backfill
  a pipeline
- The central scheduler `luigid` with a dependency graph visualizer, and parallel
  workers

# Tutorial

- Implement the tutorial "Learn Luigi in 60 mins", following
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Build it with `.claude/skills/tutorial_in_60_mins.create/SKILL.md`
  - Follow the workflow in `tutorials/README.gp.md` and the quality principles in
    `tutorials/tutorials_checklist.md`
- Check the previous tutorials and projects, listed in the section
  `Existing Tutorials and Projects` of `.claude/skills/tutorial_in_60_mins.rules.md`
  - No earlier tutorial or project uses Luigi, so read the closest workflow
    orchestration work
  - Read `data605/tutorials/tutorial_airflow/README.md`
  - Read the `README.md` of the Spring2025 Dagster, Prefect, and Airflow projects,
    and reuse what is good
    - `class_project/data605/Spring2025/projects/TutorTask102_Spring2025_Real-time_Bitcoin_Data_Ingestion_and_Analysis_with_Dagster/`
    - `class_project/data605/Spring2025/projects/TutorTask213_Spring2025_Real_Time_Bitcoin_Price_Analysis_Using_Prefect/`
    - `class_project/data605/Spring2025/projects/TutorTask95_Spring2025_Real_time_Bitcoin_Data_processing_with_airflow/`
- Create the project dir following the class instructions in
  `class_project/README.md`, section `Contribution to the Repo`
  - Start from `class_project/project_template`
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Compare briefly with Airflow and Prefect from the workflow orchestration point of
  view, e.g., how dependencies are declared and how failures are retried
- Deliverables:
  - `luigi_utils.py`
  - `luigi.API.ipynb`
  - `luigi.example.ipynb`

# Project

## Project 1 (Fall2026): Earnings Filings and Stock Reaction Pipeline

- **Project Objective**: Join SEC filings with stock prices in one pipeline to
  forecast the size of the stock move after an earnings release, and measure the
  robustness of the pipeline to failures and its speed with parallel workers
- **Dataset Suggestions**:
  - [SEC EDGAR - Submissions API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces),
    for 30 companies, following the
    [fair access rules](https://www.sec.gov/os/accessing-edgar-data)
  - [S&P 500 Constituents](https://github.com/datasets/s-and-p-500-companies), for
    the tickers and the CIK numbers
  - Daily prices from [yfinance](https://github.com/ranaroussi/yfinance), for the
    same tickers and for `SPY`
- **Tasks**:
  - **Ingest Both Sources**: Write a `FetchFilings` task for the EDGAR submissions
    JSON, with a declared `User-Agent` and the older pages in `filings.files`, and a
    `FetchPrices` task for the daily prices, with a checksum of each output
  - **Build the Events**: Keep the `8-K` filings with item `2.02`, start the reaction
    window at the next open when `acceptanceDateTime` is after 16:00 ET, and compute
    the 2-day return minus the `SPY` return
  - **Model the Reaction**: Fit a `Ridge` regression of the absolute abnormal return
    on the trailing 20-day volatility and the previous earnings move, with a
    `DateParameter` cutoff so that earlier events train and later events test
  - **Inject Failures**: Make the download tasks fail at random, use the
    `retry_count` of the task, and record how many runs are needed to finish
  - **Compare the Workers**: Run the pipeline with 1, 2, and 4 workers and plot the
    wall-clock time
  - **Evaluate the Model**: Report MAE and R-squared on the test events, next to the
    training mean and the trailing volatility alone as baselines
- **Bonus Ideas (Optional)**: Add a task that sends an email when the pipeline fails;
  add the `10-K` and `10-Q` filings as events and compare their reactions

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Ingest Both Sources
  - Result: project dir created and container running with `luigid`, and the EDGAR
    submissions JSON files and the price files written by `FetchFilings` and
    `FetchPrices`, with the checksum of each output
- Milestone 2: API notebook
  - Project tasks: Model the Reaction, Inject Failures, Compare the Workers
  - Result: `luigi.API.ipynb` covering `Task`, `requires()`, `output()`, `run()`,
    `LocalTarget`, `DateParameter`, `retry_count`, parallel workers, and the central
    scheduler
- Milestone 3: Example notebook
  - Project tasks: Build the Events, Model the Reaction, Inject Failures, Compare
    the Workers, Evaluate the Model
  - Result: `luigi.example.ipynb` running end to end

## Project 2: Inflation Forecasting Pipeline

- **Project Objective**: Build an end-to-end pipeline that ingests, cleans, and
  models US macro indicators to forecast monthly inflation, and can be re-run
  without repeating the finished steps
- **Dataset Suggestions**:
  - [FRED - Consumer Price Index (CPIAUCSL)](https://fred.stlouisfed.org/series/CPIAUCSL)
  - [FRED - Unemployment Rate (UNRATE)](https://fred.stlouisfed.org/series/UNRATE)
  - [FRED - Federal Funds Rate (FEDFUNDS)](https://fred.stlouisfed.org/series/FEDFUNDS)
- **Tasks**:
  - **Ingest the Data**: Write a `DownloadSeries` task with a `series_id` parameter
    and a `LocalTarget` that fetches a FRED series as CSV from
    `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<series_id>`, and check
    that a second run skips it
  - **Define the Problem**: Predict the annualized month-over-month CPI inflation of
    the next month as a regression problem, with RMSE as the metric, a time split
    (train through 2014, test from 2015), and the last-month inflation and the
    training mean as baselines
  - **Build the Pipeline**: Chain the `Clean`, `Features`, `Train`, and `Evaluate`
    tasks with `requires()` and `output()`, where `Features` builds the lagged
    inflation and the changes of `UNRATE` and `FEDFUNDS`, shifted by one month to
    avoid look-ahead, and run them with `luigi.build`
  - **Tune the Model**: Parametrize `Train` with a `luigi.FloatParameter` for the
    `Ridge` alpha, and run three values
  - **Evaluate the Model**: Save the RMSE and MAE of each alpha and of the baselines
    as a JSON file written by the `Evaluate` task
  - **Visualize the Pipeline**: Show the dependency graph in the `luigid` visualizer
    and plot the predicted vs. actual inflation
- **Bonus Ideas (Optional)**: Add a task with a Random Forest and compare it with the
  linear model; add the
  [FRED - Inflation Expectations (MICH)](https://fred.stlouisfed.org/series/MICH)
  as a feature and check if the RMSE improves

## Project 3: Daily Treasury Yield ETL with Backfill

- **Project Objective**: Build a daily pipeline that fetches Treasury yields,
  tracks the inversion of the yield curve, and can backfill a missing date range
- **Dataset Suggestions**:
  - [FRED - 10-Year Treasury Yield (DGS10)](https://fred.stlouisfed.org/series/DGS10)
  - [FRED - 2-Year Treasury Yield (DGS2)](https://fred.stlouisfed.org/series/DGS2)
  - [FRED - 10-Year Minus 2-Year Spread (T10Y2Y)](https://fred.stlouisfed.org/series/T10Y2Y),
    to validate the output
- **Tasks**:
  - **Fetch the Data**: Write a `FetchDay` task with a `DateParameter` and a
    `series_id` parameter that saves one CSV file per series and weekday from
    `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<series_id>&cosd=<date>&coed=<date>`,
    and treats an empty response as a market holiday
  - **Clean the Data**: Convert each file to a Parquet table with typed columns and
    an `is_holiday` flag
  - **Aggregate the Data**: Build a `WeeklySummary` task that requires the five
    weekday tasks of both series, and computes the mean 10Y-2Y spread and the number
    of inverted days (spread below zero)
  - **Backfill a Range**: Delete three daily files and re-run a range of 60 weekdays,
    then check that Luigi runs only the missing tasks
  - **Validate the Output**: Compare the daily spread computed by the pipeline with
    the FRED `T10Y2Y` series, and report the maximum absolute difference
  - **Report the Results**: Plot the weekly mean spread and the share of inverted
    days, and show the task graph with the completed and pending tasks
- **Bonus Ideas (Optional)**: Add a `RangeDaily` wrapper to automate the daily runs;
  add the 3-month yield
  [FRED - DGS3MO](https://fred.stlouisfed.org/series/DGS3MO) and flag the inversion
  of the 10Y-3M spread
