**Description**

In this project, students will utilize Ax, a powerful tool for adaptive experimentation and optimization, to design and analyze experiments efficiently. Ax enables users to explore different configurations and optimize outcomes through Bayesian optimization and multi-armed bandit strategies. This tool is particularly useful for applications in machine learning model tuning, product feature optimization, and resource allocation.

Technologies Used
Ax

- Provides a framework for adaptive experimentation and optimization.
- Supports Bayesian optimization for efficient hyperparameter tuning.
- Facilitates multi-armed bandit strategies for real-time decision-making.

---

### Project 1: Optimizing Hyperparameters for a Machine Learning Model
**Difficulty**: 1 (Easy)

**Project Objective**: The goal of this project is to optimize the hyperparameters of a Random Forest classifier to improve its accuracy on a dataset of handwritten digits (MNIST).

**Dataset Suggestions**: Use the MNIST dataset available on Kaggle: [MNIST Handwritten Digits](https://www.kaggle.com/c/digit-recognizer/data).

**Tasks**:
- **Data Preprocessing**: Load and preprocess the MNIST dataset, including normalization and splitting into training and test sets.
- **Define the Objective Function**: Create a function that trains the Random Forest model and returns accuracy based on the hyperparameters.
- **Set Up Ax Experiment**: Initialize an Ax experiment to optimize hyperparameters such as the number of trees and maximum depth.
- **Run Optimization**: Execute the optimization process using Ax to find the best hyperparameter configuration.
- **Evaluate Performance**: Compare the optimized model's accuracy with the baseline model and visualize the results.

**Bonus Ideas**: Experiment with different classifiers (e.g., SVM, KNN) and compare their optimized performances using Ax. 

---

### Project 2: Feature Selection for Predicting House Prices
**Difficulty**: 2 (Medium)

**Project Objective**: The objective is to identify the most significant features affecting house prices using Ax to optimize a feature selection process on the Ames Housing dataset.

**Dataset Suggestions**: Use the Ames Housing dataset available on Kaggle: [Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data).

**Tasks**:
- **Data Cleaning and Exploration**: Load the dataset, handle missing values, and perform exploratory data analysis to understand feature distributions.
- **Define the Feature Selection Process**: Create a model (e.g., Linear Regression) that evaluates feature importance based on the selected features.
- **Set Up Ax Experiment**: Use Ax to optimize the selection of features based on their contribution to the model’s R-squared score.
- **Run Optimization**: Execute the Ax experiment to determine the best subset of features for predicting house prices.
- **Model Evaluation**: Train the final model with the selected features and evaluate its performance using cross-validation.

**Bonus Ideas**: Extend the project by implementing different feature selection techniques (e.g., Recursive Feature Elimination) and compare results with Ax-optimized selections.

---

### Project 3: Multi-Objective Optimization for Marketing Campaigns
**Difficulty**: 3 (Hard)

**Project Objective**: The goal is to optimize multiple objectives (e.g., reach and conversion rate) for a digital marketing campaign using Ax to balance trade-offs effectively.

**Dataset Suggestions**: Use the "Online Retail" dataset available on UCI Machine Learning Repository: [Online Retail](https://archive.ics.uci.edu/ml/datasets/Online+Retail).

**Tasks**:
- **Data Preprocessing**: Load the dataset and preprocess it to extract relevant features such as customer demographics, purchase history, and campaign data.
- **Define Objectives**: Create a multi-objective function that evaluates performance based on reach and conversion rates for different campaign strategies.
- **Set Up Ax Experiment**: Initialize an Ax experiment to optimize marketing parameters such as budget allocation, target audience, and ad formats.
- **Run Optimization**: Utilize Ax to find the optimal campaign configuration that maximizes both reach and conversion rates.
- **Analyze Trade-offs**: Visualize the Pareto front to understand the trade-offs between the two objectives and make recommendations for campaign strategies.

**Bonus Ideas**: Incorporate additional objectives (e.g., customer retention) and perform a sensitivity analysis on how changes in budget affect the overall performance of the campaign.

