# Description

W&B Weave is a toolkit from Weights & Biases to build, trace, and evaluate
applications that use models, in particular LLM applications. It solves the problem
of not knowing what a pipeline did and whether a change made it better: every call of
an instrumented function is logged as a trace, and datasets, models, and evaluations
are versioned and compared in a web UI. It is worth a 60-minute tutorial because
adding `weave.init` and `@weave.op` to existing code gives full traces and comparable
evaluations at almost no cost.

## Technologies Used

W&B Weave

- Tracing of function calls with `weave.init` and `@weave.op`
- Versioned datasets and models (`weave.Dataset`, `weave.Model`)
- Evaluations with scorers (`weave.Evaluation`) to compare versions of a pipeline
- Web UI to inspect traces, compare evaluations, and share results with a team

# Tutorial

- Usual tutorial "Learn W&B Weave in 60 mins", following
  `.claude/skills/tutorials_in_60_mins.rules.md`
- Create `tutorials/WB_Weave/` with
  `.claude/skills/tutorials_in_60_mins.create/SKILL.md`
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Deliverables:
  - `weave_utils.py`
  - `weave.API.ipynb`
  - `weave.example.ipynb`

# Project

## Project 1: Real-Time Sentiment Analysis for Stock Market Prediction

- **Difficulty**: 3 (Hard)
- **Project Objective**: Build a sentiment analysis pipeline to predict stock price
  movements from news sentiment, and use Weave to trace it and to compare sentiment
  scorers by the quality of the resulting predictions
- **Dataset Suggestions**: Financial news and stock prices
  - [NewsAPI](https://newsapi.org) for news articles (free tier available; needs an
    API key)
  - Yahoo Finance: use the [yfinance](https://pypi.org/project/yfinance/) library to
    fetch stock prices
  - [Daily News for Stock Market Prediction](https://www.kaggle.com/datasets/aaron7sun/stocknews)
    as a static fallback, so that the notebooks run without API keys
- **Tasks**:
  - **Ingest News and Prices**: Fetch news and prices, cache them on disk, and align
    the daily sentiment scores with the price data by timestamp
  - **Define the Problem**: Predict the next-day direction of a stock from the
    aggregated sentiment and lagged returns, with a chronological split and a
    lagged-returns baseline
  - **Trace the Sentiment Scorers**: Wrap the scorers (e.g., VADER, TextBlob) in
    `@weave.op` after `weave.init`, and inspect the traces of every call
  - **Train and Evaluate**: Train an LSTM on the sentiment features, and compare the
    scorers and the baseline with `weave.Evaluation` using directional accuracy and
    RMSE as scorers
  - **Visualize and Report**: Compare the evaluations in the Weave UI, and plot the
    cumulative return of the sentiment strategy vs. buy-and-hold
- **Bonus Ideas (Optional)**: Implement a reinforcement learning strategy for trading
  based on sentiment predictions and visualize trading performance over time

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Ingest News and Prices
  - Result: `tutorials/WB_Weave/` container running with a Weave project, and a
    cached table with the news, the sentiment inputs, and the prices aligned by day
- Milestone 2: API notebook
  - Project tasks: Trace the Sentiment Scorers, Train and Evaluate
  - Result: `weave.API.ipynb` covering `weave.init`, `@weave.op`, `weave.Dataset`,
    `weave.Model`, and `weave.Evaluation` on a toy classification function
- Milestone 3: Example notebook
  - Project tasks: Define the Problem, Trace the Sentiment Scorers, Train and
    Evaluate, Visualize and Report
  - Result: `weave.example.ipynb` running end to end

## Project 2: Customer Segmentation Analysis

- **Difficulty**: 2 (Medium)
- **Project Objective**: Perform customer segmentation on e-commerce data to identify
  distinct customer groups and tailor marketing strategies accordingly
- **Dataset Suggestions**:
  [Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)
- **Tasks**:
  - **Load the Data**: Import the dataset and examine the structure and contents
  - **Clean the Data**: Remove duplicates and handle missing values, and perform
    exploratory data analysis (EDA)
  - **Engineer Features**: Create features such as purchase frequency, average order
    value, and recency
  - **Cluster the Customers**: Apply K-Means clustering to segment customers based on
    the engineered features
  - **Track the Clustering**: Use W&B Weave to trace the clustering runs and log the
    clustering performance metrics
  - **Visualize the Insights**: Visualize the clusters in 2D space and share the
    findings with stakeholders
- **Bonus Ideas (Optional)**: Experiment with hierarchical clustering and compare
  results with K-Means; visualize dendrograms

## Project 3: Predictive Maintenance in Manufacturing

- **Difficulty**: 2 (Medium)
- **Project Objective**: Develop a predictive maintenance model to forecast equipment
  failures in a manufacturing setting, optimizing maintenance schedules and reducing
  downtime
- **Dataset Suggestions**:
  [NASA Turbofan Engine Degradation Simulation Data Set](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)
- **Tasks**:
  - **Ingest the Data**: Load the dataset into a pandas DataFrame and perform initial
    exploration
  - **Preprocess the Data**: Clean the data, handle missing values, and create
    relevant features for modeling
  - **Train the Model**: Train a classification model (e.g., Random Forest) to
    predict failure events
  - **Track the Experiments**: Use W&B Weave to trace the training and evaluation
    functions and log the model parameters, metrics, and evaluation results
  - **Visualize the Results**: Create visualizations to analyze model performance and
    feature importance
- **Bonus Ideas (Optional)**: Compare model performance with different algorithms
  (e.g., SVM, Gradient Boosting) and visualize results
