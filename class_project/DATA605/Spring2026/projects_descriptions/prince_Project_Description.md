# Prince

## Description
- **Prince** is a Python library designed for performing Principal Component
  Analysis (PCA) and other related multivariate techniques, such as Multiple
  Correspondence Analysis (MCA) and Factor Analysis.
- It provides an intuitive interface for dimensionality reduction, allowing
  users to visualize high-dimensional data in lower dimensions effectively.
- The library is particularly useful for exploratory data analysis, enabling
  users to identify patterns, trends, and relationships within datasets.
- Prince supports various data formats, including pandas DataFrames, making it
  easy to integrate into existing data workflows.
- It allows for comprehensive visualization of results, including biplots and
  correlation plots, to facilitate interpretation of the analysis.

## Project Objective
The goal of the project is to conduct a comprehensive exploratory data analysis
on a dataset of customer reviews, using PCA to identify the underlying factors
that influence customer satisfaction. Students will optimize for dimensionality
reduction while preserving as much variance as possible in the data.

## Dataset Suggestions
1. **Customer Reviews Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Customer Reviews Dataset](https://www.kaggle.com/datasets/saurabhshahane/customer-reviews-data)
   - **Data Contains**: Customer reviews, ratings, product categories, and text
     reviews.
   - **Access Requirements**: Free account on Kaggle required.

2. **Wine Quality Dataset**
   - **Source**: UCI Machine Learning Repository
   - **URL**:
     [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
   - **Data Contains**: Chemical properties of wine samples and quality ratings.
   - **Access Requirements**: No authentication needed; freely accessible.

3. **Online Retail Dataset**
   - **Source**: UCI Machine Learning Repository
   - **URL**:
     [Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail)
   - **Data Contains**: Transactions of an online retail store, including
     product details and customer IDs.
   - **Access Requirements**: No authentication needed; freely accessible.

## Tasks
- **Data Exploration**: Load the dataset and perform initial data exploration to
  understand its structure, missing values, and basic statistics.
- **Data Preprocessing**: Clean the data by handling missing values, encoding
  categorical variables, and normalizing numerical features as necessary.
- **Dimensionality Reduction**: Apply PCA using the Prince library to reduce the
  dataset's dimensions while retaining maximum variance.
- **Visualization**: Create biplots and correlation plots to visualize the
  results of PCA and interpret the relationships between variables.
- **Analysis of Results**: Analyze the PCA output to identify key factors
  influencing customer satisfaction or product quality.

## Bonus Ideas
- **Feature Selection**: Experiment with different feature selection techniques
  before PCA to see how it affects the results.
- **Clustering**: After PCA, apply clustering algorithms (e.g., K-means) to
  identify distinct groups within the data based on the reduced dimensions.
- **Comparative Analysis**: Compare PCA results with other dimensionality
  reduction techniques, such as t-SNE or UMAP, to evaluate their effectiveness.

## Useful Resources
- [Prince Documentation](https://github.com/MaxHalford/Prince)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Kaggle API Documentation](https://www.kaggle.com/docs/api)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Dimensionality Reduction Techniques](https://towardsdatascience.com/dimensionality-reduction-techniques-in-machine-learning-with-python-1a0c3b2f1f43)
