**Description**

MLflow is an open-source platform designed to manage the end-to-end machine learning
lifecycle. It streamlines the process of tracking experiments, packaging code into
reproducible runs, and sharing and deploying models.

Technologies Used
MLflow

- **Tracking**: Log and query experiments, metrics, parameters, and artifacts
- **Projects**: Package data science code in a reusable, reproducible format
- **Models**: Manage and deploy models from various ML libraries
- **Registry**: Central repository to manage the full lifecycle of MLflow Models

### Fall2026

#### Tutorial

- Usual tutorial "Learn MLflow in 60 mins", following
  `.claude/skills/tutorials_in_60_mins.rules.md`
- Check if there is an existing tutorial and make it better
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Deliverables:
  - `mlflow_utils.py`
  - `mlflow.API.ipynb`
  - `mlflow.example.ipynb`

#### Project: Forecasting Renewable Energy Production

- **Project Objective**: Build a system to forecast the production of renewable
  energy sources like wind or solar.
- **Dataset Suggestions**: "Wind Power Forecasting" dataset from Kaggle.
- **Tasks**:
  - **Explore the Data**: Conduct exploratory data analysis using Python libraries
    to understand patterns in energy production.
  - **Engineer Features**: Engineer features such as weather data or time of day to
    enhance model accuracy.
  - **Track Experiments**: Use MLflow to log experiments across various deep
    learning models (e.g., LSTM, GRU) for time series forecasting.
  - **Evaluate and Deploy**: Evaluate and compare model performance; deploy the best
    model using MLflow's deployment capabilities.
- **Bonus Ideas (Optional)**: Integrate an external API (e.g., OpenWeatherMap) to
  enhance forecasting with real-time weather data.
- TODO(ai_gp): Improve this

#### Milestones

- TODO(ai_gp): Add milestones related to the project

### Project 1: Predicting Air Quality Index Using Time Series Data

- **Difficulty**: 1 (Easy)
- **Project Objective**: Predict the Air Quality Index (AQI) in a given city using
  historical data.
- **Dataset Suggestions**: "Delhi Air Quality Data" available on Kaggle.
- **Tasks**:
  - Load and explore the dataset using Pandas and Matplotlib.
  - Use MLflow to track experiments with different time series forecasting models
    like ARIMA or Prophet.
  - Implement a simple baseline model and log its performance metrics.
  - Utilize MLflow to save and compare the models' performance.
  - Also use MLflow to track hyperparameters like ARIMA order or Prophet seasonality
    modes.
- **Bonus Ideas (Optional)**: Implement a dashboard using Flask that visualizes
  predictions vs. actual data over time.

### Project 2: Identifying Anomalies in Network Traffic

- **Difficulty**: 2 (Medium)
- **Project Objective**: Detect anomalies in network traffic that could indicate
  potential security threats.
- **Dataset Suggestions**: "CICIDS2017" dataset from Kaggle.
- **Tasks**:
  - Preprocess the dataset to handle missing values and normalize features.
  - Train an ensemble of anomaly detection models like Isolation Forest or One-Class
    SVM.
  - Use MLflow to log parameters and metrics, comparing model performances.
  - Implement a model registry with MLflow to manage different versions.
- **Bonus Ideas (Optional)**: Test the model on streaming data using a simulated
  network traffic generator.
