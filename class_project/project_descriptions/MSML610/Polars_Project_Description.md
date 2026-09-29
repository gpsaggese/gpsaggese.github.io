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
- Create `tutorials/Polars/`, since it does not exist yet
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Compare with pandas from the functional point of view
- Deliverables:
  - `polars_utils.py`
  - `polars.API.ipynb`
  - `polars.example.ipynb`

# Project

## Project 1: Pandas vs Polars Benchmark

- **Difficulty**: 2 (Medium)
- **Project Objective**: Create a benchmark of typical data-analysis workloads and
  evaluate pandas and Polars on it by run time, peak memory, and scaling with data
  size
- **Dataset Suggestions**:
  [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
  - Use the yellow taxi Parquet files for 1, 3, and 12 months, plus the taxi zone
    lookup table, to create three data sizes
- **Tasks**:
  - **Ingest the Data**: Download the Parquet files and the zone lookup, and load
    them with `pd.read_parquet` and `pl.read_parquet`
  - **Define the Benchmark**: Specify 6-8 workloads (filter, group-by aggregation,
    join with the zone lookup, sort, window function, date and string parsing) and
    the three data sizes
  - **Implement the Workloads**: Write each workload in pandas, in Polars eager mode,
    and in Polars lazy mode (`scan_parquet` and `collect`), and check that the
    results are equal
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
  - Result: `tutorials/Polars/` container running, and the three data sizes loaded
    in both pandas and Polars with their load times in a table
- Milestone 2: API notebook
  - Project tasks: Implement the Workloads
  - Result: `polars.API.ipynb` covering expressions, contexts (`select`, `filter`,
    `group_by`, `with_columns`), joins, window functions, and lazy evaluation, each
    next to the equivalent pandas code
- Milestone 3: Example notebook
  - Project tasks: Define the Benchmark, Implement the Workloads, Measure the
    Performance, Visualize the Results
  - Result: `polars.example.ipynb` running end to end, with the benchmark harness
    in `polars_utils.py`

## Project 2: E-commerce Customer Segmentation

- **Difficulty**: 2 (Medium)
- **Project Objective**: Segment customers based on their purchasing behavior using
  clustering techniques, aiming to optimize marketing strategies
- **Dataset Suggestions**:
  [Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)
- **Tasks**:
  - **Ingest the Data**: Load the Online Retail dataset using Polars and perform
    exploratory data analysis to understand customer behavior
  - **Preprocess the Data**: Clean the data by removing duplicates and handling
    missing values, particularly in customer IDs and purchase amounts
  - **Engineer Features**: Generate features such as total spending, frequency of
    purchases, and recency of last purchase
  - **Cluster the Customers**: Implement K-means clustering to segment customers into
    distinct groups based on their purchasing behavior
  - **Evaluate the Clusters**: Analyze the characteristics of each cluster and
    visualize the results to identify patterns and insights
- **Bonus Ideas (Optional)**: Experiment with different clustering algorithms (e.g.,
  DBSCAN, hierarchical clustering) and compare results

## Project 3: COVID-19 Case Prediction

- **Difficulty**: 3 (Hard)
- **Project Objective**: Build a predictive model to forecast COVID-19 cases using
  time-series analysis, focusing on the impact of factors like mobility and public
  health measures
- **Dataset Suggestions**:
  - [COVID-19 Open Data](https://github.com/GoogleCloudPlatform/covid-19-open-data)
    by Google Cloud
  - [Google Mobility Reports](https://www.google.com/covid19/mobility/)
- **Tasks**:
  - **Collect the Data**: Use Polars to load COVID-19 case data and mobility data
    from the respective sources
  - **Merge the Data**: Merge the datasets based on date and geographical location to
    create a comprehensive dataset for analysis
  - **Engineer Features**: Create features such as daily new cases, mobility changes,
    and public health measures implemented
  - **Forecast Cases**: Utilize ARIMA or Prophet models to predict future COVID-19
    cases based on historical data and engineered features
  - **Evaluate the Forecasts**: Assess model performance using RMSE, and visualize
    predictions against actual case numbers
- **Bonus Ideas (Optional)**: Integrate additional datasets such as vaccination rates
  or hospital capacity and assess their impact on predictions
