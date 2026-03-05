# OpenRefine

## Description
- OpenRefine is a powerful open-source tool for working with messy data:
  cleaning it, transforming it, and exploring it.
- It allows users to perform complex data transformations using a simple
  interface and a powerful expression language called GREL (General Refine
  Expression Language).
- OpenRefine supports various data formats, including CSV, TSV, JSON, and XML,
  making it versatile for different types of datasets.
- The tool enables users to reconcile and link datasets with external data
  sources, enhancing the richness of the data.
- It provides robust functionalities for data clustering, allowing for the
  identification and merging of similar entries, which is crucial for data
  cleaning.
- OpenRefine also allows for easy visualization of data distributions and
  trends, aiding in exploratory data analysis.

## Project Objective
The goal of this project is to clean and prepare a messy dataset for analysis,
followed by building a classification model that predicts the category of items
based on their attributes. Students will optimize the model's accuracy using
various data cleaning techniques and transformations in OpenRefine.

## Dataset Suggestions
1. **Kaggle's "The Movies Dataset"**
   - **URL:**
     [Kaggle Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
   - **Data Contains:** Information about movies, including titles, genres,
     ratings, and release dates.
   - **Access Requirements:** Free access; requires a Kaggle account to
     download.

2. **UCI Machine Learning Repository - Wine Quality Dataset**
   - **URL:**
     [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
   - **Data Contains:** Red and white wine samples with attributes like acidity,
     sugar, and quality ratings.
   - **Access Requirements:** Publicly available without registration.

3. **Open Government Data - UK Crime Data**
   - **URL:** [UK Crime Data](https://data.police.uk/data/)
   - **Data Contains:** Monthly crime data reported by police forces across
     England and Wales, including types of crime and locations.
   - **Access Requirements:** Publicly accessible API with no authentication
     required.

4. **Hugging Face Datasets - Amazon Product Reviews**
   - **URL:**
     [Amazon Product Reviews](https://huggingface.co/datasets/amazon_polarity)
   - **Data Contains:** Reviews of Amazon products categorized as positive or
     negative.
   - **Access Requirements:** Available for free without authentication; can be
     directly accessed via the Hugging Face library.

## Tasks
- **Data Import:** Import the selected dataset into OpenRefine and familiarize
  yourself with its structure.
- **Data Cleaning:** Identify and correct inconsistencies in the dataset, such
  as duplicate entries, missing values, and formatting issues.
- **Data Transformation:** Use GREL to transform data types, create new
  attributes, and standardize categorical variables for better analysis.
- **Data Clustering:** Employ OpenRefine's clustering features to identify and
  merge similar entries, enhancing the quality of the dataset.
- **Export Cleaned Data:** Export the cleaned and transformed dataset for
  further analysis in a machine learning framework (e.g., scikit-learn).
- **Model Training:** Use a machine learning library to train a classification
  model on the cleaned dataset and evaluate its performance.

## Bonus Ideas
- Implement additional data transformation techniques, such as feature
  engineering, to improve model performance.
- Compare the performance of different classification algorithms (e.g., Decision
  Trees, Random Forest, SVM) on the cleaned dataset.
- Explore the use of external datasets to enrich the initial dataset and assess
  the impact on model accuracy.
- Create a visualization dashboard to present the cleaned data and model
  predictions.

## Useful Resources
- [OpenRefine Official Documentation](https://docs.openrefine.org/)
- [OpenRefine GitHub Repository](https://github.com/OpenRefine/OpenRefine)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets/)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
