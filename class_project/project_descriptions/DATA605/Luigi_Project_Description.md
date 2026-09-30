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
- Create `tutorials/Luigi/`, since it does not exist yet
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

## Project 1: Housing Price Pipeline

- **Difficulty**: 1 (Easy)
- **Project Objective**: Build an end-to-end pipeline that ingests, cleans, and
  models housing data, and can be re-run without repeating the finished steps
- **Dataset Suggestions**:
  [California Housing](https://github.com/ageron/handson-ml2/tree/master/datasets/housing)
- **Tasks**:
  - **Ingest the Data**: Write a `DownloadData` task with a `LocalTarget` that
    fetches the CSV, and check that a second run skips it
  - **Define the Problem**: Predict `median_house_value` as a regression problem,
    with RMSE as the metric and a mean predictor as the baseline
  - **Build the Pipeline**: Chain the `Clean`, `Features`, `Train`, and `Evaluate`
    tasks with `requires()` and `output()`, and run them with `luigi.build`
  - **Tune the Model**: Parametrize `Train` with a `luigi.FloatParameter` for the
    `Ridge` alpha, and run three values
  - **Evaluate the Model**: Save the RMSE and R-squared of each alpha and of the
    baseline as a JSON file written by the `Evaluate` task
  - **Visualize the Pipeline**: Show the dependency graph in the `luigid` visualizer
    and plot the predicted vs. actual prices
- **Bonus Ideas (Optional)**: Add a task with a Random Forest and compare it with the
  linear model; expose the trained model with FastAPI

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Ingest the Data
  - Result: `tutorials/Luigi/` container running with `luigid`, and the CSV written
    by `DownloadData`
- Milestone 2: API notebook
  - Project tasks: Build the Pipeline, Tune the Model
  - Result: `luigi.API.ipynb` covering `Task`, `requires()`, `output()`,
    `LocalTarget`, `Parameter`, `WrapperTask`, and the central scheduler
- Milestone 3: Example notebook
  - Project tasks: Define the Problem, Build the Pipeline, Evaluate the Model,
    Visualize the Pipeline
  - Result: `luigi.example.ipynb` running end to end

## Project 2: Daily Weather ETL with Backfill

- **Difficulty**: 2 (Medium)
- **Project Objective**: Build a daily pipeline that fetches weather data for several
  cities and can backfill a missing date range
- **Dataset Suggestions**:
  [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- **Tasks**:
  - **Fetch the Data**: Write a `FetchDay` task with a `DateParameter` and a `city`
    parameter that saves one JSON file per city and day
  - **Clean the Data**: Convert each JSON file to a Parquet table with typed columns
  - **Aggregate the Data**: Build a `WeeklySummary` task that requires seven
    `FetchDay` tasks and computes the mean and max temperature
  - **Backfill a Range**: Delete three daily files and re-run a 30-day range, then
    check that Luigi runs only the missing tasks
  - **Report the Results**: Plot the weekly temperatures per city, and show the task
    graph with the completed and pending tasks
- **Bonus Ideas (Optional)**: Add a `RangeDaily` wrapper to automate the daily runs

## Project 3: Bike Trips and Weather Pipeline

- **Difficulty**: 3 (Hard)
- **Project Objective**: Join two data sources in one pipeline, and measure its
  robustness to failures and its speed with parallel workers
- **Dataset Suggestions**:
  - [Citi Bike System Data](https://citibikenyc.com/system-data), using one month
  - [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- **Tasks**:
  - **Ingest Both Sources**: Write one download task per source, with checksums of
    the outputs
  - **Join the Data**: Aggregate trips per day and join them with the daily weather
  - **Model the Demand**: Fit a regression of the daily trips on the weather, and
    compare it with a day-of-week baseline
  - **Inject Failures**: Make the download task fail at random, use the `retry_count`
    of the task, and record how many runs are needed to finish
  - **Compare the Workers**: Run the pipeline with 1, 2, and 4 workers and plot the
    wall-clock time
  - **Evaluate the Model**: Report MAE and R-squared of the demand model
- **Bonus Ideas (Optional)**: Add a task that sends an email when the pipeline fails
