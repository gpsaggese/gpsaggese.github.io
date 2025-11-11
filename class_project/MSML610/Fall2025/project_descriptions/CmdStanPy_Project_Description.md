## Description  
CmdStanPy is a Python interface for Stan, a powerful probabilistic programming language that allows users to perform Bayesian statistical modeling and data analysis. It provides a simple and efficient way to fit complex models using Markov Chain Monte Carlo (MCMC) methods. CmdStanPy is particularly useful for hierarchical modeling, Bayesian regression, and any analysis requiring probabilistic inference.  

**Features of CmdStanPy:**  
- Interfaces with Stan for Bayesian modeling and inference.  
- Supports a variety of sampling algorithms, including HMC and NUTS.  
- Allows users to define custom models in Stan's modeling language.  

---

## Project 1: Predicting Housing Prices Using Bayesian Regression  
**Difficulty**: 1 (Easy)  

**Project Objective**: Build a Bayesian regression model to predict housing prices based on various features such as location, size, and number of bedrooms, optimizing for predictive accuracy.  

**Dataset Suggestions**:  
- **Dataset**: "Melbourne Housing Market" dataset on Kaggle  
- **Link**: [Melbourne Housing Market](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot)  

**Tasks**:  
- **Data Preprocessing**: Clean and preprocess the dataset, handling missing values and categorical variables.  
- **Model Specification**: Define a Bayesian linear regression model in Stan to predict housing prices.  
- **Model Fitting**: Use CmdStanPy to fit the model and sample from the posterior distribution.  
- **Model Evaluation**: Evaluate performance using RMSE and compare with a frequentist linear regression model.  
- **Visualization**: Visualize posterior distributions of model parameters and predictions.  

**Bonus Idea**: Explore the effect of adding interaction terms in the regression model.  

---

## Project 2: Hierarchical Modeling of Student Test Scores  
**Difficulty**: 2 (Medium)  

**Project Objective**: Analyze student test scores across schools and countries using a hierarchical Bayesian model, to understand how both school-level and student-level factors influence performance.  

**Dataset Suggestions**:  
- **Dataset**: "PISA 2012 Educational Attainment Dataset" on Kaggle  
- **Link**: [PISA 2012 Dataset](https://www.kaggle.com/datasets/larsen0966/pisa2012)  

**Tasks**:  
- **Data Preparation**: Extract relevant variables (test scores, school IDs, country IDs) and preprocess.  
- **Hierarchical Model Specification**: Create a multilevel Bayesian model in Stan with students nested within schools, and schools nested within countries.  
- **Model Fitting**: Fit the model using CmdStanPy and interpret the hierarchical variance components.  
- **Posterior Predictive Checks**: Conduct posterior predictive checks to validate model fit.  
- **Insights and Reporting**: Summarize findings on how student- and school-level factors impact test scores.  

**NOTE**: The full dataset is large and can be slow to fit with MCMC. Students are encouraged to **select a subset of countries or schools** (e.g., 2–3 countries) to make modeling and inference computationally feasible on a laptop or Google Colab.  

**Bonus Idea**: Extend the model to include random slopes for socio-economic status effects across countries.  

---

## Project 3: Time Series Forecasting with Bayesian Structural Time Series  
**Difficulty**: 3 (Hard)  

**Project Objective**: Forecast retail sales using a Bayesian structural time series model, capturing trend and seasonality for improved predictive performance.  

**Dataset Suggestions**:  
- **Dataset**: "Store Item Demand Forecasting Challenge" on Kaggle  
- **Link**: [Store Item Demand Forecasting](https://www.kaggle.com/c/demand-forecasting-kernels-only/data)  

**Tasks**:  
- **Data Exploration**: Perform EDA to identify seasonality and trends in the sales data.  
- **Model Specification**: Define a Bayesian structural time series model in Stan with components for trend and seasonality.  
- **Model Fitting**: Fit the model using CmdStanPy and analyze posterior distributions of parameters.  
- **Forecasting**: Generate forecasts with prediction intervals for future sales.  
- **Model Comparison**: Compare the Bayesian structural model’s performance with ARIMA using out-of-sample predictions.  

**NOTE**: Structural time series models with MCMC can be computationally expensive. To keep runtimes manageable, students should **focus on one store or a small subset of products** rather than modeling the entire dataset at once.  

**Bonus Idea**: Investigate the effect of external events (e.g., promotions, holidays) through Bayesian intervention analysis.  
