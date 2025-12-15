# Electricity Consumption Forecasting Using LSTMs

## Problem Description

Accurate electricity consumption forecasting is critical for power grid
planning and energy management. The goal of this project is to predict
hourly electricity consumption using historical load data.

This example demonstrates how recurrent neural networks, specifically
LSTM models, can be applied to time series forecasting and how
hyperparameter tuning improves model performance.

---

## Dataset Description

The dataset used in this project contains hourly electricity consumption
measurements from the PJM Interconnection. Each record consists of:

- a timestamp
- electricity load in megawatts (MW)

The data spans multiple years and exhibits strong daily and seasonal
patterns, making it well-suited for sequence-based models.

---

## Data Preparation

The raw data is resampled to a consistent hourly frequency and missing
values are handled using forward filling. The series is normalized
using Min-Max scaling to improve neural network training stability.

Sliding windows are constructed from the time series so that the model
learns to predict future values based on a fixed-length history.

---

## Baseline LSTM Model

A baseline LSTM model is first trained using a fixed architecture.
This model serves as a reference point to evaluate the benefit of
hyperparameter tuning.

The baseline model captures general temporal patterns but is limited
by manually chosen hyperparameters.

---

## Hyperparameter Tuning with Keras Tuner

To improve performance, Keras Tuner is used to automatically search
for better hyperparameters. The tuning process explores variations in:

- number of LSTM units
- learning rate
- dropout rate

Each configuration is evaluated using validation loss, and the best
model is selected based on this metric.

---

## Tuned LSTM Model

The best hyperparameters identified by Keras Tuner are used to train
a final LSTM model. This tuned model demonstrates improved forecasting
accuracy compared to the baseline.

Early stopping is applied to prevent overfitting.

---

## Multi-Step Forecasting

The project is extended to multi-step forecasting, where the model
predicts multiple future time steps instead of a single hour ahead.
This provides a more realistic forecasting scenario for real-world
applications.

---

## Baseline Comparison with Prophet

To provide a non-neural baseline, a Prophet model is trained on the
same dataset. The performance of Prophet is compared against the
LSTM-based approaches using standard error metrics.

---

## Evaluation and Results

Models are evaluated using Mean Absolute Error (MAE) and Root Mean
Squared Error (RMSE) on a held-out test set.

Overall, the tuned LSTM model achieves the best performance, followed
by the baseline LSTM. The Prophet model performs worse on this dataset,
highlighting the benefit of sequence-based neural models for this task.

---

## Conclusion

This example demonstrates the effectiveness of LSTM models for
electricity consumption forecasting and the importance of systematic
hyperparameter tuning. Keras Tuner provides a clean and efficient
framework for optimizing deep learning models in time series settings.
