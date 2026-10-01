# Description

MLflow is an open-source platform to manage the end-to-end machine learning
lifecycle. It solves the problem of untracked experiments and irreproducible models
by logging parameters, metrics, and artifacts for every run, packaging models in a
standard format, and versioning them in a registry. It is worth a 60-minute tutorial
because adding tracking to existing training code takes a few lines, and it makes
every result comparable and reproducible.

## Technologies Used

MLflow

- **Tracking**: Log and query experiments, metrics, parameters, and artifacts
- **Projects**: Package data science code in a reusable, reproducible format
- **Models**: Manage and deploy models from various ML libraries
- **Registry**: Central repository to manage the full lifecycle of MLflow Models

# Tutorial

- Implement the tutorial "Learn MLflow in 60 mins", following
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Build it with `.claude/skills/tutorial_in_60_mins.create/SKILL.md`
  - Follow the workflow in `tutorials/README.gp.md` and the quality principles in
    `tutorials/tutorials_checklist.md`
- Check the previous tutorials and projects, listed in the section
  `Existing Tutorials and Projects` of
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Read the `README.md` of the earlier MLflow projects, and reuse what is good
    - `class_project/msml610/Fall2025/projects/UmdTask15_Fall2025_Renewable_Energy_Production/`
    - `class_project/data605/Spring2026/projects/UmdTask463_DATA605_Spring2026_MLflow/`
  - Read the `README.md` of the Fall2025 Weights & Biases project for a related
    experiment-tracking tool
    - `class_project/msml610/Fall2025/projects/TutorTask_103_Weights_and_Biases_Hard/`
- Create the project dir following the class instructions in
  `class_project/README.md`, section `Contribution to the Repo`
  - Start from `class_project/project_template`
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Deliverables:
  - `mlflow_utils.py`
  - `mlflow.API.ipynb`
  - `mlflow.example.ipynb`

# Project

## Project 1 (Fall2026): Volatility Forecasting with Walk-Forward Model Comparison

- **Project Objective**: Forecast the next-week volatility of the S&P 500 with
  several models, compare them with walk-forward validation, and use MLflow to track
  every fold and register the best model
- **Dataset Suggestions**:
  - [S&P 500 Index (SP500)](https://fred.stlouisfed.org/series/SP500), the last 10
    years of daily closes
  - [CBOE Volatility Index: VIX (VIXCLS)](https://fred.stlouisfed.org/series/VIXCLS)
- **Tasks**:
  - **Preprocess the Data**: Download both series from FRED, compute the daily log
    returns and the realized volatility of the next 5 days as the target, and build
    the features only from data up to each day
  - **Define the Forecasting Problem**: Predict the target from lagged realized
    volatility and the VIX, and compute a persistence baseline that repeats the last
    5-day realized volatility
  - **Track Walk-Forward Runs**: Use a parent run per model and a nested run per
    fold of an expanding `TimeSeriesSplit`, with `mlflow.log_params` and
    `mlflow.log_metric` with the `step` argument, to compare a HAR-style linear
    regression, a `GradientBoostingRegressor`, and a GARCH(1,1) from `arch`
  - **Package the Models**: Wrap the GARCH model in an `mlflow.pyfunc.PythonModel`
    since it has no flavor, log it with `mlflow.pyfunc.log_model`, and run the
    pipeline with an `MLproject` file and `mlflow run`
  - **Compare and Register Models**: Rank the parent runs by mean RMSE and QLIKE loss
    across folds with `mlflow.search_runs`, and register the best one with
    `mlflow.register_model` under the alias `champion`
  - **Visualize the Errors**: Log with `mlflow.log_figure` the plots of predicted vs.
    realized volatility, and the error by fold in calm and crisis periods
- **Bonus Ideas (Optional)**: Add a GJR-GARCH model for the asymmetric response to
  negative returns; run a Mincer-Zarnowitz regression of realized on predicted
  volatility and log its R-squared

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Preprocess the Data
  - Result: project dir created and container running with the MLflow tracking UI,
    and a table of the daily log returns, the lagged features, and the 5-day-ahead
    realized volatility target of the S&P 500
- Milestone 2: API notebook
  - Project tasks: Track Walk-Forward Runs, Package the Models, Compare and
    Register Models
  - Result: `mlflow.API.ipynb` covering parent and nested runs, `mlflow.log_params`,
    `mlflow.log_metric` with `step`, `mlflow.log_figure`,
    `mlflow.pyfunc.PythonModel`, an `MLproject` run, `mlflow.search_runs`, and
    `mlflow.register_model` with an alias on a synthetic time series
- Milestone 3: Example notebook
  - Project tasks: Define the Forecasting Problem, Track Walk-Forward Runs, Package
    the Models, Compare and Register Models, Visualize the Errors
  - Result: `mlflow.example.ipynb` running end to end

## Project 2: Credit Card Default Prediction

- **Project Objective**: Predict if a credit card client defaults next month, and use
  MLflow to track the tuning runs, compare them, register, and serve the best model
- **Dataset Suggestions**:
  [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients),
  with 30,000 clients of a Taiwanese bank, their limits, bills, and payment history
- **Tasks**:
  - **Preprocess the Data**: Replace the undocumented codes of `EDUCATION` and
    `MARRIAGE` with the "other" category, rename `PAY_0` to `PAY_1`, and make a
    stratified 60/20/20 train, validation, and test split
  - **Define the Prediction Problem**: Predict the default from the limit, the
    demographics, and 6 months of payment history, and compute a logistic regression
    with default parameters as the baseline
  - **Track Experiments**: Use `mlflow.start_run` with nested runs per model family,
    `mlflow.log_params`, `mlflow.log_metrics`, and `mlflow.autolog` to log logistic
    regression, random forest, and gradient boosting over hyperparameter grids
  - **Package the Pipeline**: Write an `MLproject` file with the model family and the
    seed as parameters, and run it with `mlflow run`
  - **Evaluate and Register Models**: Compare runs by ROC AUC, PR AUC, and KS
    statistic on the validation set in the MLflow UI, register the best model with
    `mlflow.register_model`, set the alias `champion`, and report the test metrics
    once
  - **Serve and Visualize**: Deploy `models:/credit_default@champion` with
    `mlflow models serve`, query the REST endpoint on the test set, and plot the
    default rate by score decile
- **Bonus Ideas (Optional)**: Choose the approval threshold that minimizes the
  expected cost of a missed default versus a rejected good client, and log the cost
  with `mlflow.log_metric`; log SHAP plots as artifacts

## Project 3: Fraud Detector with Model Versions and Alias Promotion

- **Project Objective**: Detect fraudulent card transactions with a time-ordered
  split, and use the MLflow Model Registry to version two detectors, promote the
  better one, and roll back
- **Dataset Suggestions**:
  [Credit Card Fraud Detection](https://www.openml.org/d/1597), with 284,807
  transactions of European cardholders over two days and 492 frauds
- **Tasks**:
  - **Load and Split the Data**: Sort the transactions by `Time`, use the first 70%
    for training, the next 15% for validation, and the last 15% for testing, and
    report the fraud rate of each split
  - **Build a Baseline**: Fit a logistic regression with `class_weight="balanced"`,
    and log its parameters and metrics with `mlflow.start_run` and
    `mlflow.sklearn.autolog`
  - **Track a Second Detector**: Fit a `RandomForestClassifier` as a second detector,
    and run both detectors as entry points of an `MLproject` file with `mlflow run`,
    logging each model with `mlflow.sklearn.log_model` and a signature
  - **Register and Promote Models**: Register both models as versions of
    `fraud_detector` with `mlflow.register_model`, and set the alias `champion` on
    the version with the best validation PR AUC using `MlflowClient`
  - **Load by Alias and Roll Back**: Score the test set with
    `mlflow.pyfunc.load_model("models:/fraud_detector@champion")`, then move the
    alias back to version 1 and compare the metrics
  - **Evaluate and Visualize**: Report PR AUC and the recall at 1% false positive
    rate of both versions on the test set, and plot their precision-recall curves
- **Bonus Ideas (Optional)**: Add a promotion gate that moves the alias only if the
  new version improves the validation PR AUC; log the cost of missed frauds as a
  metric
