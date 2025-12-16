import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(loss_history, title="Training Loss"):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_predictions(actual, predicted, title="Stock Price Prediction", save_path=None):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label='Actual Price', color='blue')
    plt.plot(predicted, label='Predicted Price', color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    plt.show()
