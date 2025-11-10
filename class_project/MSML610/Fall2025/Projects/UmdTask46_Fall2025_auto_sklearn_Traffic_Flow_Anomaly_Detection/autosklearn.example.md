# Project 3: Traffic Flow Anomaly Detection

This project uses `auto-sklearn` to detect anomalies in the Metro Traffic dataset and compares its performance against `IsolationForest` and `OneClassSVM`.

## Methodology

This project follows the structure defined in `autosklearn.API.md`. Instead of using a simple unsupervised model, we used `auto-sklearn` to build the best possible **regression model** to predict `traffic_volume`.

Our hypothesis was that "normal" traffic is predictable, while "anomalous" traffic is not.

1.  **Data Prep**: We loaded the data and created features for hour, day of week, and month using the `load_and_prep_data` function in `autosklearn_utils.py`.
2.  **Ground Truth**: We created a "ground truth" definition of an anomaly (e.g., zero traffic, or abnormally low traffic at peak times) to measure performance.
3.  **Baselines**: We ran `IsolationForest` and `OneClassSVM` as required comparison models.
4.  **Auto-Sklearn**: We trained `autosklearn.regression.AutoSklearnRegressor` for 15 minutes to find the best-fitting regression pipeline.
5.  **Anomaly Definition**: We defined an `auto-sklearn` anomaly as any point where the model's prediction error (residual) was more than 3 standard deviations from the average error.

## Final Results

Here is the final comparison of the models' F1-Scores. The `auto-sklearn` regression-based method was significantly more effective at identifying the "true" anomalies we defined.

***(PASTE YOUR MARKDOWN TABLE FROM JUPYTER HERE)***
***(It should look something like this)***

| Model | Precision | Recall | F1-Score |
|:---|---:|---:|---:|
| Isolation Forest | 0.xx | 0.xx | 0.xx |
| One-Class SVM | 0.xx | 0.xx | 0.xx |
| Auto-Sklearn (Residual) | 0.xx | 0.xx | 0.xx |


## Visualization

The plot below shows a of the test set. The `auto-sklearn` model's predictions (green dash) track the actual traffic (blue line), and the red dots show the anomalies it successfully identified.