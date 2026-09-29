# Description

MLlib is the scalable machine learning library of Apache Spark, with algorithms for
classification, regression, clustering, and collaborative filtering, plus utilities
for feature processing and pipelines. It solves the problem of training models on
datasets that do not fit in the memory of one machine, with the same API from a
laptop to a cluster. It is worth a 60-minute tutorial because the `Pipeline` API
turns a feature-engineering and modeling workflow into one reproducible object that
Spark distributes.

## Technologies Used

MLlib

- Offers a wide range of algorithms for classification, regression, clustering, and
  collaborative filtering
- Supports both batch and streaming data processing, making it versatile for
  different data scenarios
- Integrates seamlessly with Apache Spark for distributed computing, enabling
  handling of large-scale datasets efficiently
- Provides `Pipeline`, feature transformers, and `CrossValidator` to build and tune
  reproducible workflows

# Tutorial

- Usual tutorial "Learn MLlib in 60 mins", following
  `.claude/skills/tutorials_in_60_mins.rules.md`
- Create `tutorials/MLlib/` with
  `.claude/skills/tutorials_in_60_mins.create/SKILL.md`
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Deliverables:
  - `mllib_utils.py`
  - `mllib.API.ipynb`
  - `mllib.example.ipynb`

# Project

## Project 1: S&P 500 Next-Day Direction Prediction

- **Difficulty**: 2 (Medium)
- **Project Objective**: Predict whether a stock closes up or down the next day from
  its own price history, and compare MLlib classifiers by out-of-sample AUC against a
  majority-class baseline
- **Dataset Suggestions**:
  [S&P 500 Stock Data](https://www.kaggle.com/datasets/camnugent/sandp500)
- **Tasks**:
  - **Ingest and Engineer Features**: Read the prices into a Spark DataFrame with an
    explicit schema, and use `pyspark.sql.Window` per ticker to compute lagged
    returns, rolling volatility, and moving-average ratios
  - **Define the Problem**: Label each day with the direction of the next-day close
    and split the data chronologically into train and test periods
  - **Train MLlib Models**: Build a `Pipeline` with `VectorAssembler` and
    `LogisticRegression`, `RandomForestClassifier`, or `GBTClassifier`, and tune it
    with `ParamGridBuilder` and `TrainValidationSplit`
  - **Evaluate the Models**: Compute AUC with `BinaryClassificationEvaluator`, and
    accuracy and F1 against the majority-class baseline on the test period
  - **Visualize the Results**: Plot feature importances, ROC curves, and the
    cumulative return of a long-or-flat strategy vs. buy-and-hold
- **Bonus Ideas (Optional)**: Score new daily prices with Structured Streaming;
  compare the run time of Spark with scikit-learn for growing numbers of tickers

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Ingest and Engineer Features
  - Result: `tutorials/MLlib/` container running Spark, and a feature table per
    ticker and day written as Parquet
- Milestone 2: API notebook
  - Project tasks: Train MLlib Models
  - Result: `mllib.API.ipynb` covering `VectorAssembler`, `Pipeline`, classifiers,
    and `ParamGridBuilder` with `TrainValidationSplit` on a synthetic dataset
- Milestone 3: Example notebook
  - Project tasks: Define the Problem, Train MLlib Models, Evaluate the Models,
    Visualize the Results
  - Result: `mllib.example.ipynb` running end to end

## Project 2: Customer Segmentation

- **Difficulty**: 2 (Medium)
- **Project Objective**: Segment customers based on purchasing behavior using
  clustering techniques to identify distinct customer groups for targeted marketing
  strategies
- **Dataset Suggestions**:
  [Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)
- **Tasks**:
  - **Clean the Data**: Clean the dataset by handling missing values and outliers
  - **Engineer Features**: Create features such as total purchase amount, frequency
    of purchases, and recency of last purchase
  - **Cluster the Customers**: Use MLlib's `KMeans` to segment customers into
    distinct groups
  - **Analyze the Clusters**: Visualize and interpret the clusters to derive
    actionable insights for marketing strategies
- **Bonus Ideas (Optional)**: Experiment with different clustering algorithms (e.g.,
  `GaussianMixture`) and compare results; integrate demographic data to enhance the
  clustering process

## Project 3: Predictive Maintenance

- **Difficulty**: 3 (Hard)
- **Project Objective**: Build a predictive maintenance model to forecast equipment
  failures based on sensor data, optimizing maintenance schedules and reducing
  downtime
- **Dataset Suggestions**:
  [NASA Turbofan Engine Degradation Simulation Data Set](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
- **Tasks**:
  - **Preprocess the Data**: Load the sensor data, and select and normalize features
  - **Engineer Features**: Extract features from the time-series sensor data to
    capture trends and anomalies
  - **Predict Failures**: Use MLlib's `RandomForestClassifier` or `GBTClassifier` to
    predict whether an engine fails within a fixed number of cycles
  - **Evaluate the Model**: Evaluate model performance using precision, recall, and
    F1-score on a test dataset
- **Bonus Ideas (Optional)**: Implement a real-time monitoring dashboard using the
  model predictions for proactive maintenance alerts; explore unsupervised learning
  techniques to identify patterns in the sensor data before failures occur
