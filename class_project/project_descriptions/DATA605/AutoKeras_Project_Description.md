# Description

AutoKeras is an open-source AutoML library for deep learning, built on Keras and
TensorFlow. It solves the problem of hand-designing and tuning a neural network by
searching architectures and hyperparameters automatically for tabular, image, text,
and time-series data. It is worth a 60-minute tutorial because a few lines of code
launch a full architecture search, and the trade-off between search budget and
accuracy can be measured.

## Technologies Used

AutoKeras

- Task APIs: `StructuredDataRegressor`, `StructuredDataClassifier`,
  `ImageClassifier`, `TextClassifier`, and `TimeseriesForecaster`
- Automatic architecture search and hyperparameter tuning, controlled by `max_trials`
  and the choice of tuner
- `AutoModel` with input and head blocks to define a custom search space
- Export of the best model as a plain Keras model

# Tutorial

- Implement the tutorial "Learn AutoKeras in 60 mins", following
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Build it with `.claude/skills/tutorial_in_60_mins.create/SKILL.md`
  - Follow the workflow in `tutorials/README.gp.md` and the quality principles in
    `tutorials/tutorials_checklist.md`
- Check the previous tutorials and projects, listed in the section
  `Existing Tutorials and Projects` of `.claude/skills/tutorial_in_60_mins.rules.md`
  - Read the `README.md` of the Fall2025 AutoKeras image project, and reuse what is
    good
    - `class_project/msml610/Fall2025/projects/UmdTask123_Fall2025_Fashion_Product_Image_Classification_AutoKeras/`
  - Read the scripts of the Fall2025 AutoKeras forecasting project, which has no
    `README.md`
    - `class_project/msml610/Fall2025/projects/TutorTask_67_Fall2025_AutoKeras_Electricity_Load_Forecasting/`
  - Read the `README.md` of `tutorials/TensorFlow/` for the Keras basics
- Create the project dir following the class instructions in
  `class_project/README.md`, section `Contribution to the Repo`
  - Start from `class_project/project_template`
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Compare briefly with FLAML and AutoGluon from the AutoML point of view, e.g.,
  accuracy for the same search time
- Deliverables:
  - `autokeras_utils.py`
  - `autokeras.API.ipynb`
  - `autokeras.example.ipynb`

# Project

## Project 1 (Fall2026): Volatility Forecasting with a Time-Series Search

- **Project Objective**: Forecast the next-day volatility of the S&P 500 with
  `TimeseriesForecaster`, and measure if the search beats standard volatility
  models and how stable the result is across seeds
- **Dataset Suggestions**: Daily prices of the `SPY` ETF from
  [yfinance](https://github.com/ranaroussi/yfinance), from 2005 to 2025
- **Tasks**:
  - **Build the Volatility Series**: Download the open, high, low, and close prices
    once, save them to a CSV file, and compute the daily log return and the
    Garman-Klass estimator of the daily variance
  - **Define the Baselines**: Compute the persistence forecast, the 22-day moving
    average, and a HAR regression on the daily, weekly, and monthly average
    volatility
  - **Split the Data by Time**: Use three expanding walk-forward folds that test on
    2020-2021, 2022-2023, and 2024-2025, validate on the last 20% of each training
    window, and fit the scalers on the training window only
  - **Search the Forecaster**: Fit `TimeseriesForecaster` with a `lookback` of 22
    days, a fixed `max_trials`, a cap on the epochs, and five random seeds, and
    export the best model
  - **Evaluate the Forecasts**: Report the mean and standard deviation across the
    seeds of the test RMSE of the log volatility and of the QLIKE loss for each
    fold, next to the baselines
  - **Analyze the Trade-Offs**: Plot the RMSE vs. the search time for `max_trials`
    of 5, 10, and 20 on the last fold
- **Bonus Ideas (Optional)**: Add the
  [FRED - VIX Index](https://fred.stlouisfed.org/series/VIXCLS) as an extra input
  feature and test the gain

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Build the Volatility Series
  - Result: project dir created and container running, and the CSV file of the `SPY`
    prices with the table of the daily log return and the Garman-Klass variance
- Milestone 2: API notebook
  - Project tasks: Search the Forecaster
  - Result: `autokeras.API.ipynb` covering `TimeseriesForecaster`, `lookback`,
    `max_trials`, the cap on the epochs, the random seeds, and `export_model()`
- Milestone 3: Example notebook
  - Project tasks: Define the Baselines, Split the Data by Time, Search the
    Forecaster, Evaluate the Forecasts, Analyze the Trade-Offs
  - Result: `autokeras.example.ipynb` running end to end

## Project 2: Credit Card Default Prediction with AutoML

- **Project Objective**: Predict which credit card clients will miss their next
  payment with an automatically searched neural network, and measure how much the
  search beats simple baselines
- **Dataset Suggestions**:
  [UCI - Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- **Tasks**:
  - **Preprocess the Data**: Load the 30,000 clients, make a stratified train,
    validation, and test split, and report the default rate and the missing values
    per column
  - **Define the Problem**: Predict `default payment next month` as an imbalanced
    binary classification, and fit a majority-class predictor and a
    `LogisticRegression` as baselines
  - **Search with AutoKeras**: Fit `StructuredDataClassifier` with `max_trials` of
    10 and 30 and `objective="val_auc"`, and record the search time and the
    validation AUC of each trial
  - **Customize the Search Space**: Define an `AutoModel` with
    `StructuredDataInput`, `StructuredDataBlock`, and `ClassificationHead`, and
    compare it with the default classifier search
  - **Evaluate the Models**: Export the best model with `export_model()`, reload it
    in Keras, and compare ROC-AUC, PR-AUC, and the recall of defaulters among the
    20% highest risk scores with the baselines on the test set
  - **Visualize the Results**: Plot the ROC and precision-recall curves of all
    models, and the validation AUC vs. the trial number
- **Bonus Ideas (Optional)**: Compare the AUC across `AGE` and `SEX` groups to audit
  the model for bias; repeat the study on the Give Me Some Credit data from Kaggle

## Project 3: Financial News Sentiment with Architecture Search

- **Project Objective**: Classify financial news headlines as bearish, bullish, or
  neutral with `TextClassifier`, and compare the searched network with simple
  baselines and a hand-designed Keras model
- **Dataset Suggestions**:
  [Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment),
  with about 9,500 training and 2,400 validation headlines
- **Tasks**:
  - **Load the Headlines**: Load the data with `datasets`, remove the URLs and the
    texts that appear in both splits, and report the class counts
  - **Build the Baselines**: Fit a majority-class predictor, a TF-IDF with
    `LogisticRegression`, and a small Keras model with `TextVectorization` and
    `Embedding`
  - **Search the Architecture**: Run `TextClassifier` with the `greedy` and
    `hyperband` tuners for the same `max_trials`, using 20% of the training set for
    validation
  - **Evaluate the Models**: Compare macro-F1 and the recall of the bearish class
    on the official validation split, with the parameter count and the search time
    of each model
  - **Inspect the Best Model**: Export the best model, print its layers with
    `model.summary()`, and list ten bearish headlines predicted as neutral
- **Bonus Ideas (Optional)**: Define an `AutoModel` with `TextInput`, `TextBlock`,
  and `ClassificationHead` to restrict the search space; test the best model on the
  sentences of
  [FinanceInc/auditor_sentiment](https://huggingface.co/datasets/FinanceInc/auditor_sentiment)
  and measure the drop in macro-F1
