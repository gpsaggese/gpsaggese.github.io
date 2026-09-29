# Description

Nevergrad is an optimization library developed by Facebook Research for
derivative-free optimization: it minimizes a function using only evaluations, without
gradients. It solves black-box problems such as hyperparameter tuning, noisy or
discrete objectives, and multi-objective trade-offs, with one API over many
algorithms (evolution strategies, differential evolution, Bayesian-style methods, and
the `NGOpt` meta-optimizer). It is worth a 60-minute tutorial because the `ask` and
`tell` interface works on any function, and comparing optimizers under the same
budget takes a few lines.

## Technologies Used

Nevergrad

- Offers a collection of optimization algorithms for black-box optimization problems
- Supports both single-objective and multi-objective optimization
- Provides easy integration with Python, allowing for seamless use in machine
  learning workflows
- Defines search spaces with parametrization classes (`ng.p.Scalar`, `ng.p.Log`,
  `ng.p.Choice`, `ng.p.Dict`)

# Tutorial

- Usual tutorial "Learn Nevergrad in 60 mins", following
  `.claude/skills/tutorials_in_60_mins.rules.md`
- Create `tutorials/nevergrad/` with
  `.claude/skills/tutorials_in_60_mins.create/SKILL.md`
- Make it look like `msml610/tutorials/L03_knowledge_representation/`
- Use the skills in `.claude/skills/notebook.*` to automate part of the work, and
  document how you used them
- Compare briefly with Optuna and `tutorials/Ax_Multi_Objective_Optimization/` from
  the hyperparameter search point of view
- Deliverables:
  - `nevergrad_utils.py`
  - `nevergrad.API.ipynb`
  - `nevergrad.example.ipynb`

# Project

## Project 1: Hyperparameter Tuning for Machine Learning Models

- **Difficulty**: 1 (Easy)
- **Project Objective**: Optimize the hyperparameters of a machine learning model to
  improve classification accuracy on the MNIST handwritten digits dataset
- **Dataset Suggestions**:
  [MNIST Handwritten Digits](https://www.kaggle.com/c/digit-recognizer/data)
  - Use a subsample of the images, so that one model evaluation takes a few seconds
- **Tasks**:
  - **Load the Dataset**: Import the MNIST dataset, preprocess the images for
    training, and split them into train, validation, and test sets
  - **Define the Model**: Choose a classification model (e.g., Random Forest, SVM)
    and compute the accuracy with its default hyperparameters as a baseline
  - **Optimize with Nevergrad**: Define the search space with `ng.p.Scalar`,
    `ng.p.Log`, and `ng.p.Choice`, and run `NGOpt` with `minimize` on the validation
    error, comparing it with `OnePlusOne` and `RandomSearch` under the same budget
  - **Train and Evaluate**: Train the model with the optimized hyperparameters and
    evaluate its accuracy on the test set against the baseline
  - **Visualize the Results**: Plot the best validation accuracy as a function of the
    budget for each optimizer, and the explored hyperparameter configurations
- **Bonus Ideas (Optional)**: Compare the performance of different models after
  hyperparameter tuning; implement cross-validation to ensure robustness in the
  evaluation process

### Milestones

- Milestone 1: Set up the container and the data
  - Project tasks: Load the Dataset, Define the Model
  - Result: `tutorials/nevergrad/` container running, and the baseline accuracy of
    the model with default hyperparameters on the MNIST subsample
- Milestone 2: API notebook
  - Project tasks: Optimize with Nevergrad
  - Result: `nevergrad.API.ipynb` covering the parametrization classes, optimizers,
    the `ask` and `tell` loop, and multi-objective optimization with `pareto_front`
    on a toy function
- Milestone 3: Example notebook
  - Project tasks: Define the Model, Optimize with Nevergrad, Train and Evaluate,
    Visualize the Results
  - Result: `nevergrad.example.ipynb` running end to end

## Project 2: Feature Selection for Regression Models

- **Difficulty**: 2 (Medium)
- **Project Objective**: Identify the most important features influencing house
  prices using a regression model, optimizing the feature selection process with
  Nevergrad
- **Dataset Suggestions**:
  [Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
- **Tasks**:
  - **Preprocess the Data**: Clean the dataset and handle missing values, categorical
    features, and scaling
  - **Define the Regression Model**: Choose a regression model (e.g., Linear
    Regression, Gradient Boosting)
  - **Select Features with Nevergrad**: Implement Nevergrad to optimize the selection
    of features based on model performance metrics
  - **Train the Model**: Train the regression model using the selected features and
    evaluate its performance using metrics like RMSE
  - **Analyze Feature Importance**: Visualize the importance of the selected features
    and their impact on predictions
- **Bonus Ideas (Optional)**: Experiment with different regression models and compare
  their performance with the optimized features; implement a recursive feature
  elimination approach alongside Nevergrad for comparison

## Project 3: Multi-Objective Optimization for Portfolio Management

- **Difficulty**: 3 (Hard)
- **Project Objective**: Optimize a stock portfolio by balancing risk and return
  using the multi-objective optimization of Nevergrad
- **Dataset Suggestions**: Historical stock prices from Yahoo Finance, fetched with
  [yfinance](https://pypi.org/project/yfinance/), for selected companies (e.g.,
  Apple, Microsoft, Amazon)
- **Tasks**:
  - **Acquire the Data**: Use `yfinance` to fetch historical stock price data for the
    selected companies
  - **Define the Optimization Problem**: Formulate the objectives for maximizing
    returns while minimizing risk (e.g., variance of returns)
  - **Optimize with Nevergrad**: Optimize the weights of each stock in the portfolio
    with multi-objective optimization
  - **Evaluate the Portfolio**: Calculate the expected return and risk of the
    optimized portfolio and visualize the efficient frontier
  - **Analyze the Sensitivity**: Analyze how changes in stock weights affect the
    overall portfolio performance and risk
- **Bonus Ideas (Optional)**: Compare the optimized portfolio against a benchmark
  (e.g., S&P 500) to evaluate performance; implement additional constraints (e.g.,
  maximum investment per stock) and observe the impact on optimization results
