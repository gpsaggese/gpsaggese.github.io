# Petl

## Description
- **Data Extraction and Transformation**: Petl (Python ETL) is a lightweight
  library designed for extracting, transforming, and loading data, making it
  easy to manipulate tabular data from various sources.
- **Intuitive API**: It provides a simple and intuitive API that allows users to
  perform common data wrangling tasks, such as filtering, joining, and
  aggregating, with minimal code.
- **Support for Multiple Formats**: Petl can handle data from various formats,
  including CSV, JSON, XML, and databases, making it versatile for different
  data sources.
- **Integration with Pandas**: It can seamlessly integrate with Pandas, allowing
  users to convert data between Petl tables and Pandas DataFrames for more
  complex analysis and visualization.
- **Memory Efficiency**: Designed to work efficiently with large datasets, Petl
  operates in a manner that minimizes memory usage, which is particularly
  beneficial for data processing on standard laptops.

## Project Objective
The goal of the project is to perform data cleaning and transformation on a
selected dataset, ultimately training a machine learning model to predict a
target variable. Students will optimize the model for accuracy while ensuring
the data is well-prepared and relevant.

## Dataset Suggestions
1. **Kaggle's House Prices: Advanced Regression Techniques**
   - **Source**: Kaggle
   - **URL**:
     [House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
   - **Data Contains**: Features of houses in Ames, Iowa, including sale price,
     number of rooms, and neighborhood.
   - **Access Requirements**: Free to use with a Kaggle account.

2. **UCI Machine Learning Repository: Wine Quality**
   - **Source**: UCI
   - **URL**:
     [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
   - **Data Contains**: Chemical properties of red and white wine samples, along
     with quality ratings.
   - **Access Requirements**: Publicly available without authentication.

3. **Open Data Portal: NYC Taxi Trip Data**
   - **Source**: NYC Open Data
   - **URL**: [NYC Taxi Trip Data](https://opendata.cityofnewyork.us/)
   - **Data Contains**: Records of taxi trips in New York City, including pickup
     and drop-off locations, fare amounts, and timestamps.
   - **Access Requirements**: Publicly accessible without authentication.

4. **Hugging Face Datasets: IMDb Movie Reviews**
   - **Source**: Hugging Face
   - **URL**: [IMDb Dataset](https://huggingface.co/datasets/imdb)
   - **Data Contains**: Movie reviews with associated sentiment labels
     (positive/negative).
   - **Access Requirements**: Free to use via Hugging Face API.

## Tasks
- **Data Extraction**: Use Petl to extract data from the chosen dataset and load
  it into a Petl table.
- **Data Cleaning**: Perform data cleaning tasks such as removing duplicates,
  handling missing values, and correcting data types using Petl functions.
- **Data Transformation**: Apply transformations like filtering, aggregating,
  and creating new features to prepare the dataset for modeling.
- **Model Training**: Split the cleaned dataset into training and testing sets,
  then train a regression or classification model using a suitable library
  (e.g., Scikit-learn).
- **Model Evaluation**: Evaluate the model's performance using appropriate
  metrics (e.g., RMSE for regression or accuracy for classification) and
  fine-tune as necessary.

## Bonus Ideas
- Explore advanced feature engineering techniques to improve model performance.
- Compare the performance of different machine learning algorithms (e.g.,
  decision trees, random forests, and linear regression).
- Implement cross-validation for a more robust evaluation of model performance.
- Create visualizations to illustrate data distributions and model predictions.

## Useful Resources
- [Petl Documentation](https://petl.readthedocs.io/en/stable/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/index)
