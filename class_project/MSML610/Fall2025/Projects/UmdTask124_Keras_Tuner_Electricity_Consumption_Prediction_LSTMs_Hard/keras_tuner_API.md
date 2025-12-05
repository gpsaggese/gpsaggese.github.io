# electricity_forecast.API.md

This file documents the small helper layer used in the electricity
forecasting project. The goal of this layer is to keep the notebooks
clean and to separate basic data preparation and model setup from the
main analysis.

The functions below wrap common operations (resampling, scaling,
windowing, baseline LSTM model creation, inverse-scaling of outputs).
Each function is designed to behave predictably with simple inputs and
minimal configuration.

---

## load_first_csv_in_data(data_dir="data")
Searches the `data/` directory for the first CSV file and loads it
as a pandas DataFrame.

**Args**
- `data_dir` (str)

**Returns**
- `pd.DataFrame`

---

## prepare_data_from_df(df, timestamp_col, value_col, freq="H", window_size=24, horizon=1, val_fraction=0.2)

Converts a raw dataframe into supervised learning windows.
Performs sorting, resampling, forward-fill, scaling, then builds sliding
windows for LSTM-style models.

**Args**
- df : input DataFrame with timestamp and value
- timestamp_col (str)
- value_col (str)
- freq (str): resample frequency
- window_size (int)
- horizon (int)
- val_fraction (float)

**Returns**
- X_train, y_train, X_val, y_val, scaler

---

## build_lstm_model(input_shape, units=64, dropout=0.1, lr=1e-3)
Builds a simple LSTM model for 1-step forecasting.

---

## build_multistep_model(input_shape, units=64, dropout=0.1, lr=1e-3, horizon=24)
LSTM whose Dense layer outputs multiple future steps.

---

## train_model(...)
Thin wrapper around model.fit() with early stopping.

---

## invert_and_eval(...)
Inverse transform + compute MAE/RMSE.

---

## Notes
The wrapper is intentionally light. The notebook still exposes the
Keras and Keras-Tuner APIs. This file helps avoid repeated boilerplate
and keeps the example notebook cleaner.