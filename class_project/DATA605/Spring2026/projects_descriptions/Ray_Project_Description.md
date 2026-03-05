# Ray

## Description
- Ray is an open-source framework designed for distributed computing that
  simplifies the development of scalable and high-performance applications.
- It provides a unified API for building and executing parallel and distributed
  applications, making it easier to scale Python code across multiple cores or
  nodes.
- Key features include support for task parallelism, actor-based programming,
  and easy integration with existing Python libraries, enabling seamless scaling
  of machine learning workloads.
- Ray comes with built-in libraries such as Ray Tune for hyperparameter tuning,
  Ray Serve for model serving, and Ray Data for data processing, facilitating
  end-to-end machine learning workflows.
- The framework is designed to be user-friendly, allowing developers to focus on
  their algorithms rather than the complexities of distributed systems.

## Project Objective
The goal of this project is to build a scalable machine learning model that
predicts housing prices based on various features such as location, size, and
amenities. The project will optimize the model's accuracy through hyperparameter
tuning using Ray Tune.

## Dataset Suggestions
1. **Kaggle Housing Prices Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Housing Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Features of houses (e.g., square footage, number of
     bedrooms, location) and their corresponding sale prices.
   - **Access Requirements**: Free account on Kaggle for downloading datasets.

2. **California Housing Prices**
   - **Source**: California Housing Prices Dataset (from Scikit-learn)
   - **URL**:
     [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
   - **Data Contains**: Features such as median income, house age, and location,
     along with median house values.
   - **Access Requirements**: Available through Scikit-learn library with no
     authentication needed.

3. **Open Data Portal - NYC Housing Data**
   - **Source**: NYC Open Data
   - **URL**: [NYC Housing Data](https://opendata.cityofnewyork.us/)
   - **Data Contains**: Various features about housing in New York City,
     including rent prices, neighborhood demographics, and building types.
   - **Access Requirements**: Publicly accessible without authentication.

## Tasks
- **Data Collection**: Use Ray Data to load and preprocess the selected housing
  dataset, ensuring it is ready for modeling.
- **Exploratory Data Analysis**: Conduct EDA to understand feature distributions
  and relationships using libraries like Pandas and Matplotlib.
- **Model Development**: Implement a regression model (e.g., Random Forest or
  Gradient Boosting) using Ray's integration with popular ML libraries.
- **Hyperparameter Tuning**: Utilize Ray Tune to optimize model hyperparameters,
  improving prediction accuracy.
- **Model Evaluation**: Evaluate the model's performance using metrics such as
  RMSE and R², and visualize the results.
- **Deployment**: Use Ray Serve to deploy the trained model as a REST API for
  predictions.

## Bonus Ideas
- Experiment with different regression algorithms (e.g., XGBoost, LightGBM) and
  compare their performance.
- Implement feature engineering techniques to create new features that may
  improve model accuracy.
- Analyze the impact of using distributed computing on training time and model
  performance.
- Create a dashboard using Streamlit or Dash to visualize predictions and model
  performance metrics.

## Useful Resources
- [Ray Official Documentation](https://docs.ray.io/en/latest/)
- [Ray Tune Documentation](https://docs.ray.io/en/latest/tune/index.html)
- [Ray Serve Documentation](https://docs.ray.io/en/latest/serve/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
