# Luigi

## Description
- Luigi is a Python package designed to build complex data pipelines and
  workflows in a manageable and scalable way.
- It provides a framework to define tasks and dependencies, ensuring that tasks
  are executed in the correct order.
- With built-in support for task visualization, users can easily monitor the
  status and progress of their workflows.
- Luigi integrates well with various data sources and tools, making it suitable
  for ETL (Extract, Transform, Load) processes.
- It facilitates the orchestration of machine learning workflows, from data
  ingestion and preprocessing to model training and evaluation.

## Project Objective
The goal of this project is to create an end-to-end data pipeline that ingests,
processes, and models data to predict housing prices in a specific city.
Students will optimize a linear regression model's performance by fine-tuning
hyperparameters and evaluating its accuracy.

## Dataset Suggestions
1. **Kaggle Housing Prices Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Housing Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Features of houses (e.g., size, number of rooms,
     location) and their sale prices.
   - **Access Requirements**: Free account on Kaggle.

2. **California Housing Prices Dataset**
   - **Source**: UCI Machine Learning Repository
   - **URL**:
     [California Housing](https://archive.ics.uci.edu/ml/datasets/California+Housing+Prices)
   - **Data Contains**: Housing data from California, including median house
     values and various demographic factors.
   - **Access Requirements**: Publicly accessible with no authentication needed.

3. **OpenStreetMap Data for Urban Areas**
   - **Source**: Geofabrik
   - **URL**: [Geofabrik Download Server](https://download.geofabrik.de/)
   - **Data Contains**: Geospatial data including urban area boundaries that can
     be related to housing prices.
   - **Access Requirements**: Open access, no authentication required.

4. **Zillow Home Value Index**
   - **Source**: Zillow
   - **URL**: [Zillow API](https://www.zillow.com/howto/api/APIOverview.htm)
     (check for free access)
   - **Data Contains**: Historical home values and trends for various regions.
   - **Access Requirements**: Free access with limited data points; registration
     may be required.

## Tasks
- **Task 1**: Data Ingestion
  - Use Luigi to create a task that fetches and ingests data from the selected
    datasets into a local or cloud storage solution.
- **Task 2**: Data Cleaning and Preprocessing
  - Implement a Luigi task to clean the ingested data, handling missing values,
    and transforming features as necessary for modeling.
- **Task 3**: Feature Engineering
  - Create a task that generates new features based on existing data (e.g.,
    combining or transforming features) to enhance model performance.
- **Task 4**: Model Training
  - Define a Luigi task to train a linear regression model using the processed
    dataset, applying techniques for hyperparameter tuning.
- **Task 5**: Model Evaluation
  - Implement a task to evaluate the trained model's performance using metrics
    like RMSE (Root Mean Square Error) and visualize the results.

## Bonus Ideas
- **Hyperparameter Optimization**: Explore advanced hyperparameter tuning
  techniques such as Grid Search or Random Search within the Luigi workflow.
- **Comparison with Other Models**: Implement additional models (e.g., Decision
  Trees, Random Forests) and compare their performance against the linear
  regression model.
- **Data Visualization**: Create a Luigi task that generates visualizations of
  model predictions versus actual values to interpret the model's performance
  better.
- **Deployment**: Extend the project by deploying the model as a REST API using
  Flask or FastAPI, orchestrated through Luigi.

## Useful Resources
- [Luigi Documentation](https://luigi.readthedocs.io/en/stable/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Geofabrik OpenStreetMap Data](https://download.geofabrik.de/)
- [Zillow API Documentation](https://www.zillow.com/howto/api/APIOverview.htm)
