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
- Create `tutorials/MLflow/`, since it does not exist yet
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Deliverables:
  - `mlflow_utils.py`
  - `mlflow.API.ipynb`
  - `mlflow.example.ipynb`

# Project

## Project 1: Forecasting Renewable Energy Production

- **Difficulty**: 2 (Medium)
- **Project Objective**: Build a system to forecast the production of a wind turbine,
  and use MLflow to track, compare, register, and serve the best model
- **Dataset Suggestions**:
  [Wind Power Forecasting](https://www.kaggle.com/datasets/theforcecoder/wind-power-forecasting)
- **Tasks**:
  - **Preprocess the Data**: Explore the turbine series, resample it to hourly steps,
    handle missing values, and split it chronologically into train, validation, and
    test sets
  - **Define the Forecasting Problem**: Predict the active power one hour ahead from
    lagged power, weather variables, and time of day, and compute a persistence
    baseline
  - **Track Experiments**: Use `mlflow.start_run`, `mlflow.log_params`,
    `mlflow.log_metrics`, and `mlflow.autolog` to log LSTM and GRU models across
    window lengths and hidden sizes
  - **Evaluate and Register Models**: Compare runs by MAE and RMSE on the validation
    set in the MLflow UI, and register the best model in the Model Registry with
    `mlflow.register_model`
  - **Serve and Visualize**: Deploy the registered model with `mlflow models serve`,
    query the REST endpoint on the test set, and plot predictions vs. actual power
- **Bonus Ideas (Optional)**: Integrate an external API (e.g., OpenWeatherMap) to
  enhance forecasting with real-time weather data

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Preprocess the Data
  - Result: `tutorials/MLflow/` container running with the MLflow tracking UI, and
    the hourly turbine table with the chronological splits
- Milestone 2: API notebook
  - Project tasks: Track Experiments, Evaluate and Register Models
  - Result: `mlflow.API.ipynb` covering tracking (parameters, metrics, artifacts),
    autologging, models, and the Model Registry on a synthetic regression
- Milestone 3: Example notebook
  - Project tasks: Define the Forecasting Problem, Track Experiments, Evaluate and
    Register Models, Serve and Visualize
  - Result: `mlflow.example.ipynb` running end to end

## Project 2: Identifying Anomalies in Network Traffic

- **Difficulty**: 2 (Medium)
- **Project Objective**: Detect anomalies in network traffic that could indicate
  potential security threats
- **Dataset Suggestions**:
  [CICIDS2017](https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset)
- **Tasks**:
  - **Preprocess the Data**: Handle missing values and normalize features
  - **Train Anomaly Detectors**: Train an ensemble of anomaly detection models like
    Isolation Forest or One-Class SVM
  - **Track Experiments**: Use MLflow to log parameters and metrics, and compare
    model performances
  - **Register Models**: Implement a model registry with MLflow to manage different
    versions
- **Bonus Ideas (Optional)**: Test the model on streaming data using a simulated
  network traffic generator

## Project 3: Predicting Air Quality Index Using Time Series Data

- **Difficulty**: 1 (Easy)
- **Project Objective**: Predict the Air Quality Index (AQI) in a given city using
  historical data
- **Dataset Suggestions**:
  [Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)
  (select the Delhi station)
- **Tasks**:
  - **Load and Explore the Data**: Use pandas and Matplotlib to explore the dataset
  - **Build a Baseline**: Implement a simple baseline model and log its performance
    metrics
  - **Track Forecasting Experiments**: Use MLflow to track time series models like
    ARIMA or Prophet, including hyperparameters like the ARIMA order or the Prophet
    seasonality mode
  - **Compare Models**: Use MLflow to save and compare the performance of the models
- **Bonus Ideas (Optional)**: Implement a dashboard using Flask that visualizes
  predictions vs. actual data over time
