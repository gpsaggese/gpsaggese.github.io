**Description**

In this project, students will utilize Pyro, a probabilistic programming library built on PyTorch, to perform Bayesian inference and modeling. Pyro allows for flexible probabilistic modeling and offers powerful tools for building complex models with ease. Its key features include:

- Flexible modeling with a focus on Bayesian statistics.
- Support for stochastic variational inference and MCMC.
- Integration with PyTorch for automatic differentiation and GPU acceleration.
- Rich set of built-in distributions and inference algorithms.

---

**Project 1: Predicting Housing Prices with Bayesian Regression**  
**Difficulty**: 1 (Easy)  
**Project Objective**: The goal is to build a Bayesian regression model to predict housing prices based on various features such as location, size, and number of bedrooms. The model will optimize the prediction accuracy while providing uncertainty estimates for the predictions.

**Dataset Suggestions**: Use the "California Housing Prices" dataset available on Kaggle [California Housing Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).

**Tasks**:
- **Data Preprocessing**: Clean the dataset, handle missing values, and encode categorical features.
- **Model Specification**: Define a Bayesian linear regression model using Pyro.
- **Inference**: Implement variational inference to estimate the posterior distribution of the model parameters.
- **Prediction**: Make predictions on a test set and calculate the uncertainty intervals for predictions.
- **Evaluation**: Assess model performance using metrics such as RMSE and visualize predicted vs actual prices.

---

**Project 2: Topic Modeling with Bayesian Mixture Models**  
**Difficulty**: 2 (Medium)  
**Project Objective**: Create a Bayesian mixture model to identify latent topics in a collection of documents. The project aims to optimize the identification of topics and their distributions across the documents.

**Dataset Suggestions**: Use the "20 Newsgroups" dataset, which can be accessed through the sklearn library or directly from the UCI Machine Learning Repository [20 Newsgroups](http://qwone.com/~jason/20Newsgroups/).

**Tasks**:
- **Data Preparation**: Preprocess the text data by tokenizing, removing stop words, and applying TF-IDF.
- **Model Development**: Define a Bayesian mixture model to represent the topics in the documents using Pyro.
- **Inference**: Utilize MCMC or variational inference to estimate the distribution of topics and their representations.
- **Analysis**: Analyze the identified topics and their prevalence in the dataset.
- **Visualization**: Create visualizations to represent the topics and their relationships, such as word clouds or topic distribution plots.

---

**Project 3: Anomaly Detection in Time-Series Data with Bayesian Networks**  
**Difficulty**: 3 (Hard)  
**Project Objective**: The aim is to develop a Bayesian network model for detecting anomalies in time-series sensor data. The project will focus on optimizing the model's ability to identify outliers effectively while quantifying the uncertainty of the predictions.

**Dataset Suggestions**: Use the "NASA Turbofan Engine Degradation Simulation Data Set" available on Kaggle [NASA Turbofan Engine Dataset](https://www.kaggle.com/datasets/behnamfakharzadeh/nasa-turbofan-engine-degradation-simulation-data-set).

**Tasks**:
- **Data Exploration**: Analyze the time-series data to understand its structure and identify potential anomalies visually.
- **Model Construction**: Build a Bayesian network model that captures the relationships between different sensor readings over time.
- **Anomaly Detection**: Implement inference methods to identify anomalies based on the posterior probabilities of the model.
- **Model Evaluation**: Compare the detected anomalies against known anomaly labels in the dataset and calculate precision and recall.
- **Performance Analysis**: Assess the model's performance under different conditions and visualize the detected anomalies in the time series.

**Bonus Ideas (Optional)**: For Project 1, consider adding feature selection techniques; for Project 2, explore dynamic topic modeling; and for Project 3, implement real-time anomaly detection using streaming data techniques.

