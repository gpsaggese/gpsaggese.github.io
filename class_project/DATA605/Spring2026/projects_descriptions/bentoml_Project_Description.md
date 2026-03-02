# BentoML

## Description
- BentoML is an open-source framework designed for deploying machine learning
  models as APIs.
- It simplifies the process of packaging, serving, and managing models in
  production environments.
- The tool supports various frameworks, including TensorFlow, PyTorch, and
  Scikit-learn, allowing for versatile model integration.
- BentoML provides a user-friendly interface for creating RESTful APIs to serve
  models, making it accessible for developers and data scientists.
- It includes built-in features for model versioning, monitoring, and scaling,
  enhancing the deployment workflow.
- The framework supports Docker and Kubernetes, facilitating cloud deployment
  and scalability.

## Project Objective
The goal of this project is to build and deploy a machine learning model that
predicts house prices based on various features such as location, size, and
amenities. Students will optimize the model's performance and evaluate its
accuracy using metrics like Mean Absolute Error (MAE).

## Dataset Suggestions
1. **Ames Housing Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Ames Housing Dataset](https://www.kaggle.com/datasets/prestonvong/AmesHousing)
   - **Data Contains**: Features about house characteristics and sale prices.
   - **Access Requirements**: Free access with a Kaggle account.

2. **California Housing Prices**
   - **Source**: OpenML
   - **URL**: [California Housing Dataset](https://www.openml.org/d/42165)
   - **Data Contains**: Housing data including location, number of rooms, and
     prices.
   - **Access Requirements**: Publicly available without authentication.

3. **Boston Housing Dataset**
   - **Source**: UCI Machine Learning Repository
   - **URL**:
     [Boston Housing Dataset](https://archive.ics.uci.edu/ml/datasets/Housing)
   - **Data Contains**: Features about housing in Boston, including prices.
   - **Access Requirements**: Publicly available for educational use.

4. **Kaggle House Prices: Advanced Regression Techniques**
   - **Source**: Kaggle
   - **URL**:
     [House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Detailed attributes of houses in Ames, Iowa, and their
     sale prices.
   - **Access Requirements**: Free access with a Kaggle account.

## Tasks
- **Data Exploration**: Analyze the selected dataset to understand its structure
  and identify important features for predicting house prices.
- **Data Preprocessing**: Clean the dataset by handling missing values, encoding
  categorical variables, and normalizing numerical features.
- **Model Selection**: Choose a suitable regression model (e.g., Linear
  Regression, Random Forest) to predict house prices.
- **Model Training**: Train the selected model using the processed dataset and
  evaluate its performance using cross-validation.
- **Model Deployment**: Use BentoML to package the trained model and create a
  RESTful API for serving predictions.
- **Model Evaluation**: Test the API with sample inputs and evaluate the model's
  performance using metrics such as MAE or RMSE.

## Bonus Ideas
- Experiment with ensemble methods to improve model accuracy and compare their
  performance against the initial model.
- Implement a user-friendly web interface using Flask or Streamlit to visualize
  predictions and model performance metrics.
- Explore hyperparameter tuning techniques to optimize the model further.
- Add logging and monitoring capabilities to the BentoML deployment to track
  usage and performance metrics.

## Useful Resources
- [BentoML Official Documentation](https://docs.bentoml.org/)
- [BentoML GitHub Repository](https://github.com/bentoml/BentoML)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [OpenML API Documentation](https://www.openml.org/api_docs/)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
