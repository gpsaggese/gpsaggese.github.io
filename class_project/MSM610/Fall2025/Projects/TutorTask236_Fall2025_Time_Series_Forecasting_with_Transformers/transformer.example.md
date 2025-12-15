# Transformer Time-Series Forecasting — Example Application

## Overview

This document presents a complete example application built using the
Transformer Time-Series Forecasting API and wrapper layer provided in
`transformer_utils.py`.

The application demonstrates how to:
- Prepare historical stock price data
- Train a Transformer model for multi-step forecasting
- Evaluate model performance
- Generate and visualize future price predictions

Amazon (AMZN) daily stock price data is used as the example dataset.

---

## Application Workflow

### 1. Data Preparation

The dataset is loaded from a CSV file containing historical OHLCV stock data.
The API handles:

- Date parsing and sorting
- Feature scaling
- Sliding window construction
- Train / validation / test splitting

```python
X_train, y_train, X_val, y_val, X_test, y_test, scaler_X, mean, std = prepare_dataset(
    ticker_path,
    seq_len=60,
    horizon=5
)
```
### 2. Model Training

A Transformer model is trained to predict the next 5 closing prices using
a 60-day historical window. Training uses:

- Mean Squared Error loss
- AdamW optimizer
- Learning rate warm-up and cosine decay

Automatic checkpointing of the best validation model

```python
state = train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    seq_len=60,
    out_len=5,
    epochs=80,
    batch_size=64,
    ckpt_dir=CKPT_DIR
)
```
### 3. Model Evaluation

After training, the best checkpoint is restored and evaluated on the test set.
Mean Absolute Error (MAE) is reported for each forecast horizon.

MAE_h1: 1.401
MAE_h2: 1.329
MAE_h3: 1.525
MAE_h4: 1.275
MAE_h5: 1.425

### 4. Forecast Visualization

The model’s predictions are visualized in both:

- Scaled feature space
- Real price space

This allows direct comparison between predicted and actual future prices.

### 5. Future Price Forecasting

Using the final observed window, the model predicts the next 5 trading days:

Day +1: $50.72
Day +2: $59.76
Day +3: $43.06
Day +4: $56.34
Day +5: $52.54

## Summary

This example demonstrates a full end-to-end forecasting pipeline built on top
of a clean Transformer API and wrapper layer. The separation between core logic
and notebooks ensures reusability, clarity, and maintainability.

See transformer.example.ipynb for the executable version of this application.