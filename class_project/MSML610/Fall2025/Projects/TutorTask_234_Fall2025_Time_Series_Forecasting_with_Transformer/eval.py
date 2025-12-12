"""
eval.py — Evaluation utilities for Transformer forecasting
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints
from model import TimeSeriesTransformer
import matplotlib.pyplot as plt


# ============================================================
# LOAD MODEL + CHECKPOINT
# ============================================================

def load_model(ckpt_dir, seq_len, out_len, num_features):
    """
    Loads model architecture and restores parameters.
    """

    print("🔹 Loading trained model...")

    model = TimeSeriesTransformer(
        seq_len=seq_len,
        out_len=out_len,
        num_features=num_features,
    )

    dummy = jnp.ones((1, seq_len, num_features))

    init_params = model.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
        dummy,
        train=False,
    )["params"]

    params = checkpoints.restore_checkpoint(ckpt_dir, target=init_params)

    print("✔ Checkpoint loaded.")
    return model, params


# ============================================================
# PREDICT (BATCH)
# ============================================================

def predict(model, params, X):
    """
    X: (batch, seq_len, num_features)
    Returns: (batch, out_len)
    """

    preds = model.apply(
        {"params": params},
        jnp.array(X),
        train=False,
        rngs={"dropout": jax.random.PRNGKey(0)},
    )

    return np.array(preds)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred):
    """
    Compute MAE and RMSE per horizon.
    """

    results = {}

    for h in range(y_true.shape[1]):
        mae = float(np.mean(np.abs(y_true[:, h] - y_pred[:, h])))
        rmse = float(np.sqrt(np.mean((y_true[:, h] - y_pred[:, h]) ** 2)))

        results[f"MAE_h{h+1}"] = mae
        results[f"RMSE_h{h+1}"] = rmse

    return results


# ============================================================
# Predict Next 5 Days (Real Values)
# ============================================================

def predict_next_5_days(model, params, last_window_scaled, scaler_y):
    """
    last_window_scaled: (seq_len, num_features)
    scaler_y: StandardScaler for Close price
    """

    window = jnp.array(last_window_scaled)[None, :, :]

    y_scaled = model.apply(
        {"params": params},
        window,
        train=False,
        rngs={"dropout": jax.random.PRNGKey(0)},
    )[0]

    y_real = scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

    return y_real, y_scaled


# ============================================================
# Plotting
# ============================================================

def plot_horizon(y_true, y_pred, horizon=1, num_points=200):

    h = horizon - 1

    plt.figure(figsize=(12, 5))
    plt.plot(y_true[:num_points, h], label="Actual", linewidth=2)
    plt.plot(y_pred[:num_points, h], label="Predicted", linewidth=2)
    plt.title(f"Forecast Horizon {horizon}")
    plt.xlabel("Test window index")
    plt.ylabel("Scaled Close price")
    plt.grid(True)
    plt.legend()
    plt.show()
