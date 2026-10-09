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

- Implement the tutorial "Learn MLlib in 60 mins", following
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - Build it with `.claude/skills/tutorial_in_60_mins.create/SKILL.md`
  - Follow the workflow in `tutorials/README.gp.md` and the quality principles in
    `tutorials/tutorials_checklist.md`
- Check the previous tutorials and projects, listed in the section
  `Existing Tutorials and Projects` of
  `.claude/skills/tutorial_in_60_mins.rules.md`
  - No earlier tutorial or project uses MLlib, so read the earlier Spark projects of
    DATA605 for the Spark setup in Docker
    - `class_project/data605/Spring2025/projects/TutorTask516_Spring2025_Real_Time_Bitcoin_Price_Analysis_with_Apache_Spark/`
    - `class_project/data605/Spring2025/projects/TutorTask94_Spring2025_Real_time_Bitcoin_Data_Processing_with_PySpark/`
    - `class_project/data605/Spring2025/projects/TutorTask108_Spring2025_Implementing_Real-Time_Bitcoin_Price_Analysis_with_Spark_SQL/`
- Create the project dir following the class instructions in
  `class_project/README.md`, section `Contribution to the Repo`
  - Start from `class_project/project_template`
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Deliverables:
  - `mllib_utils.py`
  - `mllib.API.ipynb`
  - `mllib.example.ipynb`

# Project

## Project 1 (Fall2026): Stock Clustering for Portfolio Diversification

- **Project Objective**: Group S&P 500 stocks by risk and return profile with MLlib
  clustering, and test out of sample whether a portfolio with one stock per cluster
  has lower volatility than random portfolios of the same size
- **Dataset Suggestions**:
  - [S&P 500 Companies](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
    for the tickers and the GICS sectors
  - Daily adjusted prices for 2015-2024 from
    [`yfinance`](https://github.com/ranaroussi/yfinance)
  - Today's constituents introduce survivorship bias, so state this limit in the
    analysis
- **Tasks**:
  - **Compute Stock Features**: Load the prices into a Spark DataFrame, and use
    `pyspark.sql.Window` per ticker to compute daily returns, annualized volatility,
    beta to `SPY`, 12-month momentum, and maximum drawdown over 2015-2019
  - **Prepare the Vectors**: Combine the features with `VectorAssembler`, standardize
    them with `StandardScaler`, and project them with `PCA` for plotting
  - **Cluster the Stocks**: Fit `KMeans` for several values of `k`, choose `k` with
    the silhouette score of `ClusteringEvaluator`, and compare with
    `BisectingKMeans`
  - **Evaluate the Portfolios**: Build an equal-weight portfolio with the
    lowest-volatility stock of each cluster, and compare its volatility, Sharpe
    ratio, and maximum drawdown over 2020-2024 against the equal-weight S&P 500, one
    stock per GICS sector, and the average of 200 random portfolios of the same size
  - **Visualize the Clusters**: Plot the clusters in the `PCA` plane and the cluster
    composition by GICS sector, and report the adjusted Rand index against the
    sectors
- **Bonus Ideas (Optional)**: Compare with `GaussianMixture` soft assignments; refit
  the clusters every year and measure how much the cluster membership changes

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Compute Stock Features
  - Result: project dir created and container running Spark, and a table with the
    annualized volatility, beta, momentum, and maximum drawdown of each S&P 500
    stock over 2015-2019 written as Parquet
- Milestone 2: API notebook
  - Project tasks: Prepare the Vectors, Cluster the Stocks
  - Result: `mllib.API.ipynb` covering `pyspark.sql.Window`, `VectorAssembler`,
    `StandardScaler`, `PCA`, `KMeans`, `BisectingKMeans`, and `ClusteringEvaluator`
    on a synthetic dataset
- Milestone 3: Example notebook
  - Project tasks: Prepare the Vectors, Cluster the Stocks, Evaluate the
    Portfolios, Visualize the Clusters
  - Result: `mllib.example.ipynb` running end to end

## Project 2: Lending Club Loan Default Prediction

- **Project Objective**: Predict whether a 36-month consumer loan is charged off from
  the information known at origination, and compare MLlib classifiers by out-of-time
  AUC and AUC-PR against a majority-class baseline and a `grade`-only baseline
- **Dataset Suggestions**:
  [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
  - Use the accepted loans file, and keep a sample of the issue years if memory is
    tight
- **Tasks**:
  - **Ingest and Engineer Features**: Read the accepted loans into a Spark DataFrame
    with an explicit schema, keep the `Fully Paid` and `Charged Off` loans with a
    36-month term, and derive the issue year, the credit history length, and the
    loan-to-income ratio
  - **Define the Problem**: Label `Charged Off` as default, drop the columns known
    only after origination (e.g., payments and recoveries) to avoid leakage, and
    split by issue date: train on the 2012-2014 loans and test on the 2015 loans
  - **Train MLlib Models**: Build a `Pipeline` with `Imputer`, `StringIndexer`,
    `OneHotEncoder`, `VectorAssembler`, and `LogisticRegression`,
    `RandomForestClassifier`, or `GBTClassifier`, and tune it on the training years
    with `ParamGridBuilder` and `CrossValidator`
  - **Evaluate the Models**: Compute AUC and AUC-PR with
    `BinaryClassificationEvaluator` on the 2015 loans, against the majority-class
    baseline and a `LogisticRegression` on `grade` only
  - **Visualize the Results**: Plot the ROC curves, the feature importances, and the
    observed default rate by decile of predicted risk
- **Bonus Ideas (Optional)**: Score new loan applications with Structured Streaming;
  predict the loss on charged-off loans with `GBTRegressor`

## Project 3: Stock Recommendations for Institutional Investors

- **Project Objective**: Recommend new stock positions to institutional investment
  managers with collaborative filtering on SEC 13F holdings, and compare `ALS` with a
  popularity baseline by precision@10 and NDCG@10 on the next quarter
- **Dataset Suggestions**:
  [SEC Form 13F Data Sets](https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets)
  - Use 3-4 consecutive quarterly zip files, each about 75 MB compressed
  - Keep the 2,000 managers with the most holdings and the 5,000 most held CUSIPs if
    memory is tight
- **Tasks**:
  - **Ingest the Holdings**: Read the `INFOTABLE`, `SUBMISSION`, and `COVERPAGE`
    tables with explicit schemas, and join them to get the manager, the report
    period, the CUSIP, and the position value of the latest filing per manager and
    period
  - **Define the Problem**: Build an implicit-feedback matrix of manager by CUSIP with
    `log(1 + value)` as strength, and define the test set of a quarter as the
    positions held in the next quarter and not in this one
  - **Train the Recommender**: Index the managers and the CUSIPs with
    `StringIndexer`, fit `ALS` with `implicitPrefs=True`, and tune `rank`,
    `regParam`, and `alpha` with `ParamGridBuilder` on an earlier pair of quarters
  - **Evaluate the Recommendations**: Get the top 10 stocks per manager with
    `recommendForAllUsers`, and compute precision@10 and NDCG@10 with
    `RankingEvaluator` against a most-held-stocks baseline, for each pair of
    consecutive quarters and by manager size
  - **Inspect the Latent Factors**: List the nearest neighbors of a few well-known
    stocks in the item-factor space, and cluster the item factors with `KMeans`
- **Bonus Ideas (Optional)**: Compare with explicit `ALS` on the portfolio weights;
  measure the effect of `rank` on precision@10 and on the run time
