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
- Create `tutorials/DuckDB/`, since it does not exist yet
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

## Project 1: SQL Analytics on NYC Taxi Trips

- **Difficulty**: 1 (Easy)
- **Project Objective**: Answer business questions about taxi trips with SQL directly
  on a Parquet file, and measure the gain over pandas
- **Dataset Suggestions**:
  [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page),
  using one month of yellow taxi trips
- **Tasks**:
  - **Load the Data**: Query the Parquet file in place with `duckdb.sql`, and report
    the rows, the types, and the missing values with `SUMMARIZE`
  - **Define the Questions**: Write five business questions, e.g., the busiest hours
    and the tip rate by payment type
  - **Write the Queries**: Answer the questions with `GROUP BY`, `CASE`, and window
    functions, e.g., the top pickup zones per hour with `ROW_NUMBER() OVER`
  - **Benchmark against pandas**: Run the same five queries in pandas, and compare
    the time and the peak memory
  - **Export the Results**: Save the aggregates in a `.duckdb` file and as Parquet
    with `COPY`
  - **Visualize the Results**: Plot the trips by hour and the tip rate by payment
    type from the query outputs
- **Bonus Ideas (Optional)**: Join the trips with the taxi zone lookup table; read
  the file over HTTP with the `httpfs` extension

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Load the Data
  - Result: `tutorials/DuckDB/` container running, and the `SUMMARIZE` table of the
    trips
- Milestone 2: API notebook
  - Project tasks: Write the Queries, Export the Results
  - Result: `duckdb.API.ipynb` covering `connect`, queries on Parquet and CSV files,
    the exchange with pandas, Polars, and Arrow, window functions, `EXPLAIN ANALYZE`,
    `COPY`, and `SET`
- Milestone 3: Example notebook
  - Project tasks: Define the Questions, Write the Queries, Benchmark against pandas,
    Visualize the Results
  - Result: `duckdb.example.ipynb` running end to end

## Project 2: Flight Delays from Raw CSV Files

- **Difficulty**: 2 (Medium)
- **Project Objective**: Build a small analytical database from raw CSV files, and
  compare the file formats and the engines on delay queries
- **Dataset Suggestions**:
  [Kaggle - 2015 Flight Delays and Cancellations](https://www.kaggle.com/datasets/usdot/flight-delays)
- **Tasks**:
  - **Ingest the CSV Files**: Read the flights, airlines, and airports files with
    `read_csv`, and convert the flights to Parquet with `COPY`
  - **Join the Tables**: Join the flights with the airlines and the airports, and
    save the result as a view
  - **Compute the Delay Statistics**: Compute the delay rate per airline, airport,
    and month, with a moving average and a rank as window functions
  - **Compare the Formats**: Time the same query on the CSV file, the Parquet file,
    and the native `.duckdb` table, and report the file sizes
  - **Compare the Engines**: Repeat the three heaviest queries in pandas and Polars,
    and report the time
  - **Visualize the Delays**: Plot a heatmap of the mean delay by airport and month
- **Bonus Ideas (Optional)**: Add a query with `PIVOT` to show delays by airline and
  weekday

## Project 3: Scaling and Resource Limits

- **Difficulty**: 3 (Hard)
- **Project Objective**: Measure how DuckDB scales with the data size, the threads,
  and the memory limit, and find where pandas stops working
- **Dataset Suggestions**:
  [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page),
  using the twelve monthly yellow taxi files of one year
- **Tasks**:
  - **Assemble the Dataset**: Download the twelve Parquet files, and query them as
    one table with a glob pattern
  - **Profile the Query Plan**: Use `EXPLAIN ANALYZE` on three queries, and name the
    operator that dominates the time of each
  - **Measure the Scaling**: Time a group-by and a join on 1, 3, 6, and 12 months,
    with five repetitions each
  - **Limit the Resources**: Repeat the group-by with a `memory_limit` of 1, 2, and 4
    GB and with 1, 2, 4, and 8 threads, and record the time and whether it spills to
    disk
  - **Compare with pandas**: Run the group-by in pandas on the same months, and note
    where it slows down or runs out of memory
  - **Report the Trade-Offs**: Plot the time vs. the data size and vs. the number of
    threads, with the standard deviation over the repetitions
- **Bonus Ideas (Optional)**: Test the effect of sorting the Parquet files on the
  time of a filter query
