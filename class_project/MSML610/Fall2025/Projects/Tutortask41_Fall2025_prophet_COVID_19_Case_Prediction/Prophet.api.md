
---

## **`API.md`**  

```markdown
# COVID-19 Forecasting API Documentation

This document describes the **native programming interface** for the COVID-19 prediction utilities.

---

## 1. Data Extraction

### `extract_region_daily(df, country, state=None) -> pd.DataFrame`

Extracts daily new cases for a specific country or state.

**Parameters:**
- `df` (`pd.DataFrame`): Raw COVID-19 cumulative dataset.
- `country` (`str`): Country name.
- `state` (`str`, optional): State/province name.

**Returns:**
- `pd.DataFrame` with columns:
  - `ds`: date
  - `y`: new daily cases
  - `cases_cum`: cumulative cases

---

## 2. External Regressors

### `build_binary_regressor(dates_index, event_dates) -> pd.Series`

Creates a binary series indicating the occurrence of events (e.g., lockdowns).

**Parameters:**
- `dates_index` (`pd.DatetimeIndex`): Date range for the series.
- `event_dates` (`List[str]`): List of event dates in `'YYYY-MM-DD'` format.

**Returns:**
- `pd.Series` of 0/1 values aligned with `dates_index`.

---

## 3. LSTM Utilities

### `create_sequences(values, seq_len=28) -> Tuple[np.ndarray, np.ndarray]`

Transforms a 1D array into overlapping input sequences for LSTM.

**Parameters:**
- `values` (`np.ndarray`): Input series (scaled).
- `seq_len` (`int`): Length of input sequences.

**Returns:**
- `X`: Input sequences (`n_samples, seq_len, 1`)
- `y`: Target values (`n_samples,`)

---

### `forecast_lstm_multi(model, last_window, steps, scaler) -> np.ndarray`

Performs iterative multi-step forecasting with a trained LSTM.

**Parameters:**
- `model` (`keras.Model`): Trained LSTM model.
- `last_window` (`np.ndarray`): Last observed input sequence.
- `steps` (`int`): Forecast horizon (number of steps).
- `scaler` (`MinMaxScaler`): Scaler used for input normalization.

**Returns:**
- Forecasted values in original scale (`np.ndarray`)

---

## 4. Plotting

### `plot_actual_vs_pred(dates, actual, preds, conf=None, interventions=None, title='')`

Visualizes actual vs predicted cases with optional confidence intervals and intervention markers.

**Parameters:**
- `dates` (`pd.Series` or `pd.DatetimeIndex`): Dates for x-axis.
- `actual` (`np.ndarray` or `pd.Series`): Actual daily cases.
- `preds` (`np.ndarray` or `pd.Series`): Predicted cases.
- `conf` (`np.ndarray`, optional): Confidence intervals `[lower, upper]`.
- `interventions` (`dict`, optional): `{name: [dates]}` to show intervention lines.
- `title` (`str`, optional): Plot title.

**Usage Example:**

```python
plot_actual_vs_pred(
    dates=df_region['ds'],
    actual=df_region['y'],
    preds=fcst['yhat'],
    conf=fcst[['yhat_lower','yhat_upper']].values,
    interventions={'lockdown': lockdowns, 'vax_start': vax_starts},
    title='Prophet Forecast vs Actual Cases'
)

