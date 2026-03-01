# Argo Workflows

## Description
- Argo Workflows is an open-source container-native workflow engine for
  orchestrating parallel jobs on Kubernetes.
- It allows users to define complex workflows using a YAML-based syntax,
  enabling the execution of multiple tasks in sequence or in parallel.
- The tool supports a wide array of tasks, including data processing, machine
  learning model training, and deployment, making it versatile for data science
  projects.
- Argo provides features like retries, dependencies, and resource management,
  allowing for robust and fault-tolerant workflow execution.
- It integrates seamlessly with Kubernetes, enabling scalability and efficient
  resource utilization for data-intensive applications.

## Project Objective
The goal of this project is to create a data processing and machine learning
pipeline that predicts housing prices based on various features. The project
will optimize the model's accuracy by fine-tuning hyperparameters and evaluating
different algorithms.

## Dataset Suggestions
1. **Kaggle Housing Prices Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Housing Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Various features of houses (e.g., square footage, number
     of bedrooms, etc.) and their corresponding sale prices.
   - **Access Requirements**: Free account on Kaggle required for download.

2. **California Housing Prices Dataset**
   - **Source**: California Department of Housing and Community Development
   - **URL**:
     [California Housing Data](https://www.kaggle.com/c/california-housing-prices/data)
   - **Data Contains**: Information about housing prices, demographics, and
     geographic data in California.
   - **Access Requirements**: Free account on Kaggle required for download.

3. **OpenStreetMap (OSM) Housing Data**
   - **Source**: OpenStreetMap
   - **URL**: [OSM Data](https://download.geofabrik.de/)
   - **Data Contains**: Geographic and feature data that can be used to enrich
     housing datasets (e.g., proximity to amenities).
   - **Access Requirements**: Publicly accessible data; no authentication
     needed.

## Tasks
- **Task 1**: Workflow Design
  - Design a YAML file to define the workflow for data ingestion, preprocessing,
    model training, and evaluation.
- **Task 2**: Data Ingestion
  - Implement a containerized step to fetch and preprocess the housing dataset,
    handling missing values and feature scaling.
- **Task 3**: Model Training
  - Create a step in the workflow for training multiple regression models (e.g.,
    Linear Regression, Random Forest) and fine-tuning hyperparameters.
- **Task 4**: Model Evaluation
  - Add a step to evaluate model performance using metrics like RMSE and R², and
    select the best-performing model for predictions.
- **Task 5**: Results Visualization
  - Implement a final step to visualize the predictions against actual prices
    using libraries like Matplotlib or Seaborn.

## Bonus Ideas
- Extend the project by integrating additional data sources, such as local
  economic indicators, to improve model accuracy.
- Compare the performance of traditional machine learning models with a
  pre-trained deep learning model for regression tasks.
- Implement a real-time prediction service using the trained model and Argo
  Workflows for continuous integration and deployment.

## Useful Resources
- [Argo Workflows Official Documentation](https://argoproj.github.io/argo-workflows/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Kubernetes Official Documentation](https://kubernetes.io/docs/home/)
- [GitHub Repository for Argo Workflows](https://github.com/argoproj/argo-workflows)
- [Introduction to Machine Learning with Scikit-Learn](https://scikit-learn.org/stable/supervised_learning.html)
