**Description**

BoTorch is a library built on PyTorch that facilitates Bayesian optimization, enabling users to optimize expensive-to-evaluate functions efficiently. It provides a flexible interface for defining and optimizing objective functions using Gaussian processes, allowing for the incorporation of prior knowledge and uncertainty quantification.

Technologies Used
BoTorch

- Implements advanced Bayesian optimization techniques.
- Supports multi-fidelity and multi-objective optimization.
- Integrates seamlessly with PyTorch for deep learning applications.

---

### Project 1: Optimizing Hyperparameters for Machine Learning Models
**Difficulty**: 1 (Easy)

**Project Objective**: The goal is to optimize the hyperparameters of a simple machine learning model (e.g., Random Forest) using BoTorch to achieve the best performance on a classification task.

**Dataset Suggestions**: 
- Use the "Wine Quality" dataset available on Kaggle (https://www.kaggle.com/datasets/uciml/wine-quality).

**Tasks**:
- **Data Preprocessing**: Load the dataset, handle missing values, and encode categorical variables.
- **Define Objective Function**: Create an objective function that evaluates model performance (e.g., accuracy) based on hyperparameters.
- **Set Up BoTorch**: Utilize BoTorch to perform Bayesian optimization on hyperparameters like `n_estimators`, `max_depth`, and `min_samples_split`.
- **Model Evaluation**: Assess the best hyperparameters using cross-validation and compare against a grid search baseline.
- **Visualization**: Visualize the optimization process and performance metrics using Matplotlib.

---

### Project 2: Multi-Objective Optimization for Portfolio Allocation
**Difficulty**: 2 (Medium)

**Project Objective**: The goal is to optimize a stock portfolio's allocation by maximizing returns while minimizing risk using BoTorch's multi-objective optimization capabilities.

**Dataset Suggestions**: 
- Use the "S&P 500 Stock Data" dataset available on Yahoo Finance (via the yfinance library) to gather historical stock prices.

**Tasks**:
- **Data Collection**: Gather historical stock prices for a selection of S&P 500 companies using the yfinance library.
- **Define Objectives**: Create an objective function that calculates expected returns and volatility (risk) based on portfolio weights.
- **Set Up BoTorch**: Implement multi-objective optimization to find the optimal portfolio allocation that balances return and risk.
- **Constraint Handling**: Ensure the weights sum to 1 and are non-negative.
- **Performance Analysis**: Analyze the Pareto front and visualize the trade-off between return and risk.

---

### Project 3: Optimizing Experimental Design for Drug Discovery
**Difficulty**: 3 (Hard)

**Project Objective**: The goal is to optimize the experimental design for a drug discovery process by identifying the most promising compounds to test, maximizing effectiveness while minimizing cost using BoTorch.

**Dataset Suggestions**: 
- Use the "ChEMBL" dataset available on the ChEMBL database (https://www.ebi.ac.uk/chembl/) for compound bioactivity data.

**Tasks**:
- **Data Preparation**: Extract relevant bioactivity data and preprocess it for modeling.
- **Define Objective Function**: Create an objective function that evaluates the expected efficacy and cost of testing specific compounds.
- **Set Up BoTorch**: Utilize BoTorch for Bayesian optimization to select compounds based on the defined objectives.
- **Incorporate Uncertainty**: Model uncertainty in efficacy predictions and cost estimates using Gaussian processes.
- **Results Interpretation**: Analyze and interpret the optimized compound selections, focusing on the trade-offs between efficacy and cost.

**Bonus Ideas**:
- For Project 1, compare the Bayesian optimization results to other optimization techniques like random search or grid search.
- For Project 2, extend the analysis to include transaction costs and rebalancing strategies.
- For Project 3, integrate additional data sources (e.g., genetic information) to enhance the optimization process.

