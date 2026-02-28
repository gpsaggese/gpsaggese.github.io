# FastAPI

## Description
- FastAPI is a modern, fast (high-performance) web framework for building APIs
  with Python 3.6+ based on standard Python type hints.
- It allows for automatic generation of OpenAPI documentation, making it easy to
  understand and interact with your API.
- FastAPI is designed for building APIs quickly with minimal code, enabling
  developers to focus on functionality rather than boilerplate.
- It supports asynchronous programming, making it suitable for high-performance
  applications that require handling multiple requests simultaneously.
- Built-in validation and serialization features simplify data handling and
  ensure data integrity with minimal effort.

## Project Objective
The goal of this project is to develop a RESTful API using FastAPI that serves a
machine learning model for predicting house prices based on various features.
The project will focus on optimizing the model's accuracy and response time for
real-time predictions.

## Dataset Suggestions
1. **Kaggle - House Prices: Advanced Regression Techniques**
   - **URL**:
     [House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Features related to house characteristics (e.g., square
     footage, number of bedrooms, location).
   - **Access Requirements**: Free account on Kaggle required to download the
     dataset.

2. **Kaggle - California Housing Prices**
   - **URL**:
     [California Housing Prices Dataset](https://www.kaggle.com/c/california-housing-prices)
   - **Data Contains**: Various features of houses in California, including
     location, size, and demographics.
   - **Access Requirements**: Free account on Kaggle required to download the
     dataset.

3. **UCI Machine Learning Repository - Boston Housing Dataset**
   - **URL**:
     [Boston Housing Dataset](https://archive.ics.uci.edu/ml/datasets/Housing)
   - **Data Contains**: Information on housing in Boston, including features
     like crime rate, number of rooms, and property age.
   - **Access Requirements**: Publicly accessible without any authentication.

## Tasks
- **Environment Setup**: Set up a FastAPI environment and install necessary
  libraries (e.g., FastAPI, Uvicorn, scikit-learn).
- **Data Preprocessing**: Load the dataset, clean it, and prepare it for model
  training (handling missing values, encoding categorical variables).
- **Model Training**: Choose a regression model (e.g., Linear Regression, Random
  Forest) and train it using the prepared dataset.
- **API Development**: Create endpoints for the API, including an endpoint to
  accept input data and return house price predictions.
- **Testing and Validation**: Validate the API endpoints and ensure that they
  return accurate predictions based on test data.
- **Documentation**: Generate and customize API documentation using FastAPI's
  built-in features to make it user-friendly.

## Bonus Ideas
- Implement user authentication for the API to restrict access to specific
  endpoints.
- Enhance the API to include additional endpoints for model evaluation metrics
  (e.g., RMSE, R²).
- Explore deploying the FastAPI app on a cloud platform (e.g., Heroku, AWS) for
  real-world accessibility.
- Create a front-end interface using a framework like React or Vue.js to
  interact with the FastAPI backend.

## Useful Resources
- [FastAPI Official Documentation](https://fastapi.tiangolo.com/)
- [FastAPI GitHub Repository](https://github.com/tiangolo/fastapi)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
