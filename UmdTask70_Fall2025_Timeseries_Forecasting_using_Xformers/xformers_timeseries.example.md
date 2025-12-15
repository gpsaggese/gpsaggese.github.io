# Time-Series Forecasting with Xformers: Example

This example demonstrates how to use the Xformers-based API to forecast stock prices. We will use historical data for Apple Inc. (AAPL) to predict future closing prices.

## Workflow

1.  **Data Acquisition**: Fetch historical data for 'AAPL' from Yahoo Finance using `utils_data_io`.
2.  **Data Preprocessing**:
    -   Select the 'Close' price column.
    -   Normalize data using `MinMaxScaler`.
    -   Create sliding window sequences (e.g., use past 60 days to predict next day).
3.  **Model Initialization**: Create the `XformersTimeSeriesModel`.
4.  **Training**: Train the model for 20 epochs using Mean Squared Error (MSE) loss.
5.  **Evaluation**:
    -   Predict on the test set.
    -   Inverse transform validation data to get original price scale.
    -   Calculate MAE and RMSE.
6.  **Visualization**: Plot training loss and actual vs. predicted prices.

## Prerequisities

Ensure the Docker container is running as per the README instructions to satisfy all dependencies.
