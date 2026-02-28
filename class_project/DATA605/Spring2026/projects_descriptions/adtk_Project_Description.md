# ADTK

## Description
ADTK (Anomaly Detection ToolKit) is a Python library designed for detecting
anomalies in time series data. It provides a comprehensive set of tools that
simplify the implementation of various anomaly detection algorithms, including
statistical and machine learning approaches. The library is particularly useful
for preprocessing data, visualizing results, and evaluating the performance of
different anomaly detection methods.

## Technologies Used
- ADTK
  - Offers a variety of anomaly detection algorithms (e.g.,
    SeasonalDecomposition, Threshold, and Isolation Forest).
  - Provides utilities for time series preprocessing and feature extraction.
  - Includes visualization tools to examine the detected anomalies.

- Pandas
  - Essential for data manipulation and time series handling.
  - Facilitates data cleaning, filtering, and transformation.

## Project Objective
- The goal of this project is to detect anomalies in a public time series
  dataset related to air quality measurements (e.g., PM2.5 levels) over time.
  Students will optimize the model to accurately identify unusual spikes or
  drops in pollution levels, which can indicate potential environmental issues
  or data collection errors.

## Dataset Suggestions
- Students can find suitable datasets on platforms such as:
  - Kaggle (search for air quality or pollution datasets)
  - Government open data portals (e.g., EPA, WHO)
  - Open data repositories on GitHub

## Tasks
- **Set Up Environment**
  - Install required packages:
    - `adtk`
    - `pandas`
    - `matplotlib`
  - Create a new Python script or Jupyter notebook for the project.

- **Data Ingestion**
  - Download the air quality dataset from the selected source.
  - Load the dataset into a Pandas DataFrame and explore its structure.
  - Handle missing values and perform any necessary data cleaning.

- **Time Series Preprocessing**
  - Convert the date/time column to a Pandas datetime object.
  - Set the date/time column as the DataFrame index.
  - Resample the data if necessary (e.g., to daily averages).

- **Anomaly Detection Implementation**
  - Choose an appropriate anomaly detection algorithm from ADTK (e.g.,
    SeasonalDecomposition, Threshold).
  - Fit the model to the time series data and detect anomalies.
  - Visualize the detected anomalies alongside the original time series data
    using Matplotlib.

- **Evaluation of Results**
  - Analyze the detected anomalies:
    - Count the number of anomalies detected.
    - Investigate specific periods where anomalies occurred.
  - Assess the effectiveness of the model by comparing detected anomalies
    against known events (if available).

- **Visualization**
  - Create visualizations to show:
    - The original time series data with anomalies highlighted.
    - Time series decomposition (trend, seasonality, residuals).
    - Any relevant correlation analysis with external datasets (e.g., weather
      data).

## Bonus Ideas (Optional)
- **Compare Different Algorithms**
  - Implement multiple anomaly detection algorithms from ADTK and compare their
    performance based on precision and recall metrics.

- **Feature Engineering**
  - Create additional features such as moving averages or rolling statistics to
    improve anomaly detection accuracy.

- **Real-Time Anomaly Detection**
  - Simulate a real-time anomaly detection system by periodically updating the
    dataset and re-running the anomaly detection pipeline.

## Useful Resources
- [ADTK Documentation](https://adtk.readthedocs.io/en/latest/)
- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

## Cost
- ADTK: Open-source, free.
- Pandas: Open-source, free.
- Matplotlib: Open-source, free.
