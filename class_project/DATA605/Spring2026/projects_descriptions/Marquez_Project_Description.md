# Marquez

## Description
- Marquez is an open-source metadata service designed to help data teams manage
  and track the lifecycle of their data assets.
- It provides capabilities for data lineage tracking, allowing users to
  visualize how data flows through various pipelines and transformations.
- The tool supports integration with various data processing frameworks, making
  it easier to manage data across different environments.
- Marquez offers a user-friendly interface for browsing datasets, pipelines, and
  jobs, facilitating better collaboration among data scientists and engineers.
- It includes features for documenting datasets and pipelines, enhancing
  transparency and reproducibility in data projects.

## Project Objective
The goal of this project is to build a data pipeline that ingests, transforms,
and analyzes a public dataset, while utilizing Marquez to track the data lineage
and document the entire process. Students will optimize the pipeline for
performance and ensure that the metadata is correctly captured and displayed in
Marquez.

## Dataset Suggestions
1. **Kaggle Titanic Dataset**
   - **Source**: Kaggle
   - **URL**: [Titanic Dataset](https://www.kaggle.com/c/titanic/data)
   - **Data Contains**: Passenger information including age, gender, class, and
     survival status.
   - **Access Requirements**: Free to use with a Kaggle account (registration
     required).

2. **UCI Machine Learning Repository - Wine Quality**
   - **Source**: UCI Machine Learning Repository
   - **URL**:
     [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
   - **Data Contains**: Chemical properties of wines and their quality ratings.
   - **Access Requirements**: Publicly available without any registration.

3. **Open Government Data - NYC Airbnb Listings**
   - **Source**: NYC Open Data
   - **URL**:
     [NYC Airbnb Listings](https://data.cityofnewyork.us/Housing-Development/NYC-Airbnb-Listings/2yzn-sv3g)
   - **Data Contains**: Information about Airbnb listings in New York City
     including price, location, and availability.
   - **Access Requirements**: Publicly accessible without authentication.

4. **Hugging Face Datasets - IMDB Reviews**
   - **Source**: Hugging Face Datasets
   - **URL**: [IMDB Reviews Dataset](https://huggingface.co/datasets/imdb)
   - **Data Contains**: Movie reviews labeled for sentiment (positive/negative).
   - **Access Requirements**: Free to use without authentication.

## Tasks
- **Task 1: Dataset Ingestion**
  - Load the selected dataset into a data processing framework (e.g., Pandas or
    Spark) and prepare it for transformation.
- **Task 2: Data Transformation**
  - Perform necessary data cleaning and transformation steps, such as handling
    missing values and encoding categorical variables.

- **Task 3: Model Training**
  - Implement a machine learning model (e.g., logistic regression for
    classification or linear regression for prediction) using the transformed
    dataset.

- **Task 4: Metadata Documentation**
  - Use Marquez to document the dataset and data processing steps, ensuring that
    lineage is tracked throughout the pipeline.

- **Task 5: Visualization and Reporting**
  - Create visualizations to summarize the results of the model and the data
    pipeline, and present the findings in a report format.

## Bonus Ideas
- **Extension**: Implement additional machine learning models and compare their
  performance using Marquez to track model lineage.
- **Baseline Comparison**: Compare the performance of your model against a
  simple baseline model (e.g., predicting the mean or mode).
- **Challenge**: Explore the impact of different feature engineering techniques
  on model performance and document these changes in Marquez.

## Useful Resources
- [Marquez Documentation](https://marquez.readthedocs.io/en/latest/)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [NYC Open Data](https://opendata.cityofnewyork.us/)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
