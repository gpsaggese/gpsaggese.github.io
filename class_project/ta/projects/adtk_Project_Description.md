# ADTK

## Description

ADTK (Anomaly Detection ToolKit) is a Python library designed for detecting anomalies in time series data. It provides a range of methods for both unsupervised and supervised anomaly detection, making it suitable for various applications in finance, healthcare, and IoT. Students will leverage ADTK to analyze time series data, identify anomalies, and visualize the results, enhancing their understanding of anomaly detection techniques.

## Technologies Used

- **ADTK**
  - Offers a variety of anomaly detection algorithms, including:
    - Statistical methods (e.g., Z-Score, Median Absolute Deviation)
    - Machine learning models (e.g., Isolation Forest, One-Class SVM)
  - Provides tools for time series preprocessing and feature engineering.
  - Includes visualization capabilities to help interpret anomaly detection results.

- **Pandas**
  - Data manipulation and analysis library for handling time series data.
  - Facilitates data cleaning, transformation, and aggregation.

## Project Objective

- The goal of this project is to build a pipeline that detects anomalies in a publicly available time series dataset (e.g., stock prices, sensor readings) using ADTK. Students will optimize the detection process to identify significant deviations from normal behavior, providing insights into potential issues or events.

## Dataset Suggestions

- Explore time series datasets available on platforms like:
  - Kaggle (look for datasets related to stock prices, energy consumption, or sensor data)
  - Open government portals (check for public datasets on environmental metrics or transportation data)
  - UCI Machine Learning Repository (for various time series datasets)

## Tasks

- **Set Up Environment**
  - Install ADTK and other required packages:
    - `adtk`
    - `pandas`
    - `matplotlib`
  - Set up a Python environment (e.g., Google Colab or Jupyter Notebook) to run the project.

- **Data Ingestion**
  - Load the chosen time series dataset using Pandas.
  - Inspect the dataset for missing values and data types.
  - Clean and preprocess the data as necessary (e.g., handling missing values, converting timestamps).

- **Exploratory Data Analysis (EDA)**
  - Visualize the time series data to understand its structure and identify potential anomalies visually.
  - Generate summary statistics to characterize the dataset (mean, median, standard deviation).

- **Anomaly Detection Implementation**
  - Choose appropriate anomaly detection algorithms from ADTK based on the dataset characteristics.
  - Apply the selected models to detect anomalies in the time series data.
  - Configure parameters for the models to optimize detection performance.

- **Evaluation of Anomaly Detection**
  - Analyze the detected anomalies:
    - Count and visualize detected anomalies against the original time series.
    - Assess the impact of detected anomalies on the overall dataset.
  - If available, compare detected anomalies with known events or labels in the dataset (if applicable).

- **Visualization of Results**
  - Create visualizations to illustrate the original time series along with detected anomalies.
  - Use Matplotlib to create plots that clearly show the anomalies against the normal data points.

## Bonus Ideas (Optional)

- **Comparison of Algorithms**: Implement multiple anomaly detection algorithms from ADTK and compare their performance on the same dataset. Analyze which method provides better results in terms of accuracy and interpretability.
  
- **Feature Engineering**: Explore additional feature engineering techniques (e.g., rolling statistics) to enhance the anomaly detection performance and compare results with and without these features.

- **Real-Time Anomaly Detection**: Simulate a real-time anomaly detection system by periodically ingesting new data and applying the anomaly detection pipeline, allowing for continuous monitoring of the dataset.

## Useful Resources
- [ADTK Documentation](https://adtk.readthedocs.io/en/latest/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

## Cost
- ADTK: Open-source, free.
- Pandas: Open-source, free.
- Matplotlib: Open-source, free.

