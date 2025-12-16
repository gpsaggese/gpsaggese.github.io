# Xformers Time Series Library API

## 📘 Overview

The **Xformers Time Series Library** is a modular, high-performance toolkit designed for sequence modeling and forecasting. It leverages the memory-efficient attention mechanisms provided by Meta's `xformers` library, allowing for faster training and lower memory usage on compatible hardware.

Key features include:
- **Resilient Architecture**: Automatically detects `xformers`. If unavailable, it seamlessly falls back to standard PyTorch implementation without breaking execution.
- **Modular Components**: Clean separation of dataset creation, preprocessing, and modeling.
- **Transformer Encoder**: A pure encoder-based architecture optimized for regression tasks.

---

## 🛠️ Installation & Dependencies

This library requires:
- `torch` (PyTorch 2.0+ recommended)
- `pandas` & `numpy` (for data manipulation)
- `scikit-learn` (for scaling)
- `xformers` (Optional, but recommended for speed)

```bash
pip install torch pandas numpy scikit-learn xformers
```

---

## 📚 API Reference

### 1. Data Processing

#### `class xformers_timeseries_utils.TimeSeriesDataset`

A PyTorch `Dataset` wrapper that converts linear time-series data into sliding window sequences for training.

```python
dataset = TimeSeriesDataset(data, seq_len=60, target_len=7)
```

**Arguments:**
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `data` | `np.ndarray` | The input time-series array of shape `(N, features)`. |
| `seq_len` | `int` | The lookback window length (input sequence length). |
| `target_len` | `int` | The prediction horizon length (target sequence length). Default: `1`. |

---

#### `class xformers_timeseries_utils.DataPreprocessor`

A utility class to handle common preprocessing tasks like MinMax scaling and train/test splitting.

```python
processor = DataPreprocessor(seq_len=60, train_split=0.8)
train_loader, test_loader, scaler = processor.fit_transform(df, feature_cols)
```

**Methods:**

**`fit_transform(df: pd.DataFrame, feature_cols: List[str]) -> (Dataset, Dataset, MinMaxScaler)`**
- Standardizes the data using `MinMaxScaler`.
- Splits data into training (80%) and testing (20%) sets.
- Returns `TimeSeriesDataset` objects. You should wrap these in `DataLoader` for batching.

---

### 2. Modeling

#### `class xformers_timeseries_utils.XformersTimeSeriesModel`

The core neural network model. It acts as a wrapper around a Transformer Encoder with a linear regression head.

**Architecture:**
Input Projection → [Transformer Encoder Layers (xformers/Attention)] → LayerNorm → Decoder Head

**Arguments:**
| Parameter | Type | Description |
| :--- | :--- | :--- |
| `input_dim` | `int` | Number of features per time step (e.g., 1 for univariate, 5 for OHLCV). |
| `d_model` | `int` | Dimension of the internal embeddings. |
| `nhead` | `int` | Number of attention heads. |
| `num_layers` | `int` | Number of stacked transformer encoder layers. |
| `dropout` | `float` | Dropout rate for regularization. Default: `0.1`. |

---

### 3. Training & Evaluation

#### `def train_model(...)`

Standard PyTorch training loop tailored for this model.

**Signature:**
```python
train_model(model, train_loader, criterion, optimizer, device, epochs=10)
```
- **Returns**: A list of average loss values per epoch for plotting learning curves.

#### `def evaluate_model(...)`

Generates predictions on the test set without updating gradients.

**Signature:**
```python
evaluate_model(model, test_loader, device)
```
- **Returns**: `(predictions, actuals)` tuple of numpy arrays.

---

## ⚙️ Configuration & Fallback Logic

The library includes a robust import block that attempts to load `xformers`.

1. **Primary**: Tries `xformers.components.MultiHeadDispatch`.
2. **Fallback**: If `xformers` is missing or incompatible, it defines a local `MultiHeadDispatch` that wraps `torch.nn.MultiheadAttention`.

**Verification:**
You can check which backend is active by inspecting:
```python
import xformers_timeseries_utils
print(xformers_timeseries_utils.MultiHeadDispatch)
```
If it points to `xformers.components...`, you have the optimized version. If it shows a local class, you are on the fallback.
