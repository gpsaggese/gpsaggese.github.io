# Django ORM

## Description
- Django ORM (Object-Relational Mapping) is a powerful database abstraction
  layer that allows developers to interact with databases using Python code
  instead of SQL.
- It simplifies database operations by allowing users to define their data
  models as Python classes, which are then translated into database tables.
- Key features include support for multiple database backends (e.g., SQLite,
  PostgreSQL, MySQL), automatic schema migrations, and built-in query
  capabilities.
- Django ORM supports complex queries, relationships between models, and allows
  for easy data retrieval and manipulation, making it suitable for data-driven
  applications.
- It integrates seamlessly with the Django web framework, providing a robust
  environment for building full-stack applications with a data-centric focus.

## Project Objective
The goal of this project is to build a web application that predicts house
prices based on various features such as location, size, and amenities. Students
will implement a regression model using Django ORM to manage data and facilitate
user interactions.

## Dataset Suggestions
1. **Kaggle House Prices Dataset**
   - **Source**: Kaggle
   - **URL**:
     [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Features of houses (e.g., size, location, number of
     rooms) and their sale prices.
   - **Access Requirements**: Free account on Kaggle to download the dataset.

2. **Open Data Portal - City of Chicago**
   - **Source**: City of Chicago
   - **URL**:
     [Chicago Housing Prices](https://data.cityofchicago.org/Community-Health/Chicago-Housing-Prices/7t8t-5j7q)
   - **Data Contains**: Housing price data along with various features such as
     neighborhood and building type.
   - **Access Requirements**: Publicly available data; no authentication
     required.

3. **Zillow Home Value Index**
   - **Source**: Zillow
   - **URL**: [Zillow Research Data](https://www.zillow.com/research/data/)
   - **Data Contains**: Home values, rental prices, and property characteristics
     across various U.S. regions.
   - **Access Requirements**: Publicly available data; no authentication
     required.

## Tasks
- **Set Up Django Project**: Create a new Django project and configure the
  settings to use the chosen database (SQLite for simplicity).
- **Define Models**: Create Django models representing the housing data,
  including fields for features and target prices.
- **Data Ingestion**: Write scripts to load the dataset into the Django ORM,
  populating the database with the housing data.
- **Implement Regression Model**: Use a machine learning library (e.g.,
  scikit-learn) to build a regression model that predicts house prices based on
  the features.
- **Create Views and Templates**: Develop web pages that allow users to input
  house features and receive predicted prices from the model.
- **Evaluate and Optimize**: Analyze model performance using metrics such as
  RMSE and refine the model based on evaluation results.

## Bonus Ideas
- **User Authentication**: Implement user accounts where users can save their
  predictions.
- **Data Visualization**: Add charts to visualize the distribution of house
  prices and feature correlations.
- **Model Comparison**: Experiment with different regression algorithms (e.g.,
  linear regression, decision trees) and compare their performances.
- **Deployment**: Deploy the application using a platform like Heroku to make it
  accessible online.

## Useful Resources
- [Django ORM Documentation](https://docs.djangoproject.com/en/stable/topics/db/models/)
- [Django REST Framework](https://www.django-rest-framework.org/) for building
  APIs.
- [Scikit-learn Documentation](https://scikit-learn.org/stable/) for machine
  learning tasks.
- [Kaggle API Documentation](https://www.kaggle.com/docs/api) for accessing
  datasets programmatically.
- [Django Deployment Checklist](https://docs.djangoproject.com/en/stable/howto/deployment/checklist/)
  for best practices in deploying Django applications.
