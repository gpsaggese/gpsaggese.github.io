# Rich

## Description
- **Rich** is a Python library designed for creating visually appealing
  command-line interfaces (CLIs) with advanced formatting options.
- It supports rendering of rich text, tables, progress bars, and markdown,
  making it ideal for interactive data science applications.
- The library allows for easy integration of colorful output, which enhances the
  user experience when presenting data insights in terminal applications.
- Rich enables the display of complex data structures, like data frames, in a
  readable format, which is particularly useful for exploratory data analysis.
- It also provides features for logging with customizable styles, making it
  easier to track the progress and status of data processing tasks.

## Project Objective
The goal of this project is to create a command-line data analysis tool that
processes a dataset, performs exploratory data analysis (EDA), and presents the
results in an interactive and visually appealing format using the Rich library.
The project aims to optimize data presentation and user interaction in a
terminal environment.

## Dataset Suggestions
1. **Iris Dataset**
   - **Source**: UCI Machine Learning Repository
   - **URL**: [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
   - **Data Contains**: Measurements of iris flowers (sepal length, sepal width,
     petal length, petal width) and species classification.
   - **Access Requirements**: Publicly available without any authentication.

2. **Titanic Dataset**
   - **Source**: Kaggle
   - **URL**: [Titanic Dataset](https://www.kaggle.com/c/titanic/data)
   - **Data Contains**: Passenger information (age, sex, ticket class, etc.) and
     survival status.
   - **Access Requirements**: Free to access with a Kaggle account.

3. **Wine Quality Dataset**
   - **Source**: UCI Machine Learning Repository
   - **URL**:
     [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
   - **Data Contains**: Chemical properties of wine and quality ratings.
   - **Access Requirements**: Publicly available without any authentication.

4. **Global Temperature Dataset**
   - **Source**: Kaggle
   - **URL**:
     [Global Temperature Dataset](https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data)
   - **Data Contains**: Global surface temperature records over time.
   - **Access Requirements**: Free to access with a Kaggle account.

## Tasks
- **Data Loading**: Utilize pandas to load the dataset and display basic
  information using Rich for better readability.
- **Exploratory Data Analysis**: Generate summary statistics and visualizations
  (e.g., histograms, scatter plots) using Rich to enhance the output.
- **Data Cleaning**: Identify and handle missing values or outliers, displaying
  the process and results with Rich.
- **Modeling**: Implement a machine learning model (e.g., classification or
  regression) using scikit-learn, and present the results with Rich.
- **Interactive CLI**: Create a user-friendly command-line interface that allows
  users to navigate through different analysis options and view results in a
  structured format.

## Bonus Ideas
- Implement a feature to export results (e.g., visualizations or reports) to CSV
  or text files.
- Compare the performance of different machine learning models and present a
  summary table using Rich.
- Incorporate user inputs to filter the dataset dynamically and re-run analyses
  based on those inputs.
- Add a logging feature that tracks the analysis steps taken and any issues
  encountered during execution.

## Useful Resources
- [Rich Documentation](https://rich.readthedocs.io/en/stable/)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)
