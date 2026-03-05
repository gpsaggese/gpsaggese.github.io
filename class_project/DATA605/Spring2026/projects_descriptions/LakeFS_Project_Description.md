# LakeFS

## Description
- LakeFS is an open-source data versioning tool designed for data lakes,
  enabling users to manage data in a way similar to version control systems like
  Git.
- It allows data scientists and engineers to create branches of data, making it
  easier to experiment with new data transformations and models without
  affecting the main dataset.
- LakeFS supports data lineage tracking, enabling users to understand how data
  changes over time and the impact of those changes on analysis and machine
  learning models.
- The tool integrates seamlessly with existing data lake storage solutions such
  as Amazon S3, Google Cloud Storage, and Azure Blob Storage, making it
  versatile for various cloud environments.
- Users can perform data operations like merging, branching, and reverting
  changes, which enhances collaboration among team members working on
  data-driven projects.

## Project Objective
The goal of this project is to build a machine learning model that predicts
housing prices based on various features such as location, size, and amenities.
Students will utilize LakeFS to manage different versions of the dataset,
enabling them to explore various feature engineering strategies and model
training approaches while maintaining a clear history of their experiments.

## Dataset Suggestions
1. **Kaggle Housing Prices Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Housing Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Features of houses in Ames, Iowa, including sale price,
     square footage, number of bedrooms, and more.
   - **Access Requirements**: Free to use after creating a Kaggle account.

2. **California Housing Prices**
   - **Source**: California Housing Prices (Open Data)
   - **URL**:
     [California Housing Prices](https://www.datasets.org/datasets/california-housing-prices)
   - **Data Contains**: Information on housing prices in California, including
     median income, housing age, and location.
   - **Access Requirements**: Publicly available without authentication.

3. **Boston Housing Dataset**
   - **Source**: UCI Machine Learning Repository
   - **URL**:
     [Boston Housing Dataset](https://archive.ics.uci.edu/ml/datasets/Housing)
   - **Data Contains**: Various features affecting housing prices in Boston,
     including crime rate, number of rooms, and distance to employment centers.
   - **Access Requirements**: Publicly accessible without authentication.

## Tasks
- **Set Up LakeFS Environment**: Install and configure LakeFS to manage your
  datasets effectively.
- **Data Ingestion**: Load the chosen housing dataset into LakeFS and create an
  initial branch for your project.
- **Exploratory Data Analysis (EDA)**: Use LakeFS to version your EDA notebooks,
  documenting insights and data cleaning processes.
- **Feature Engineering**: Experiment with different feature sets in separate
  branches, allowing easy comparison of model performance.
- **Model Development**: Train multiple regression models (e.g., Linear
  Regression, Decision Trees) on different branches of the dataset and evaluate
  their performance.
- **Version Control and Collaboration**: Utilize LakeFS features to merge
  successful experiments and document the evolution of your model over time.

## Bonus Ideas
- Explore advanced regression techniques such as Random Forest or Gradient
  Boosting and compare their performance against simpler models.
- Implement a model interpretability approach (e.g., SHAP or LIME) to understand
  which features influence housing prices the most.
- Create a dashboard using Streamlit or Dash to visualize predictions and model
  performance metrics.
- Investigate the impact of adding new features (e.g., economic indicators) on
  model accuracy by creating separate branches for each experiment.

## Useful Resources
- [LakeFS Official Documentation](https://docs.lakefs.io/)
- [LakeFS GitHub Repository](https://github.com/treeverse/lakeFS)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Streamlit Documentation](https://docs.streamlit.io/) for building interactive
  dashboards.
