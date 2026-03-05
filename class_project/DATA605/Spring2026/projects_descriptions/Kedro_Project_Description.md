# Kedro

## Description
- Kedro is an open-source Python framework designed to facilitate the
  development of reproducible, maintainable, and modular data science code.
- It promotes best practices in data science projects, including project
  structure, data management, and pipeline orchestration.
- The tool encourages the use of data catalogs, allowing users to define,
  manage, and access datasets in a consistent manner.
- Kedro integrates seamlessly with various data science libraries and tools,
  enabling efficient data processing and model training workflows.
- It supports version control for data and models, making it easier to track
  changes and collaborate within teams.
- The framework includes a visual interface for pipeline visualization, helping
  users understand data flows and dependencies.

## Project Objective
The goal of this project is to build a machine learning pipeline using Kedro to
predict house prices based on various features such as location, size, and
amenities. The project will focus on optimizing the model for accuracy and
interpretability.

## Dataset Suggestions
1. **Kaggle House Prices Dataset**
   - **Source Name**: Kaggle
   - **URL**:
     [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Features related to house specifications and sale
     prices.
   - **Access Requirements**: Free to use with a Kaggle account (registration
     required).

2. **Boston Housing Dataset**
   - **Source Name**: UCI Machine Learning Repository
   - **URL**:
     [Boston Housing Data](https://archive.ics.uci.edu/ml/datasets/Housing)
   - **Data Contains**: Various attributes of housing in Boston, including crime
     rates and average number of rooms.
   - **Access Requirements**: Publicly accessible without authentication.

3. **California Housing Prices Dataset**
   - **Source Name**: California Department of Housing and Community Development
   - **URL**:
     [California Housing Data](https://www.dhcd.ca.gov/Pages/CaliforniaHousingData.aspx)
   - **Data Contains**: Information on housing prices, demographics, and
     socio-economic factors in California.
   - **Access Requirements**: Open access, no authentication needed.

## Tasks
- **Project Setup**: Initialize a new Kedro project and set up the directory
  structure to follow best practices.
- **Data Ingestion**: Create a data catalog to ingest datasets from the chosen
  sources and ensure they are properly formatted.
- **Data Preprocessing**: Implement data cleaning and preprocessing steps,
  including handling missing values and feature encoding.
- **Model Development**: Select an appropriate regression model (e.g., Linear
  Regression, Random Forest) and set up a Kedro pipeline for training.
- **Model Evaluation**: Evaluate model performance using metrics like RMSE and
  R², and visualize results using Kedro's built-in tools.
- **Documentation**: Document the project thoroughly, including code comments
  and a project report outlining the methodology and findings.

## Bonus Ideas
- **Hyperparameter Tuning**: Implement hyperparameter tuning using tools like
  Optuna or GridSearchCV to optimize the model further.
- **Model Comparison**: Compare multiple regression models and analyze their
  performance differences.
- **Feature Importance Analysis**: Use techniques like SHAP or LIME to interpret
  model predictions and understand feature contributions.

## Useful Resources
- [Kedro Documentation](https://kedro.readthedocs.io/en/stable/)
- [Kedro GitHub Repository](https://github.com/kedro-org/kedro)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [California Housing Data](https://www.dhcd.ca.gov/Pages/CaliforniaHousingData.aspx)
