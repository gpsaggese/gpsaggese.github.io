import numpy as np
import matplotlib.pyplot as plt


def compute_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    return mae, rmse


def plot_predictions(y_true, y_pred, n=500, title="Actual vs Predicted Load"):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true[:n], label='Actual')
    plt.plot(y_pred[:n], label='Predicted', alpha=0.8)
    plt.title(title)
    plt.xlabel("Time Steps (Hours)")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.show()
