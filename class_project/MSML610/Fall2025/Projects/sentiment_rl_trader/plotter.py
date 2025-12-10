# plotter.py
# This file creates the final plot that shows the stock price
# and the RL trading strategy's equity curve — and saves it for W&B.

import matplotlib.pyplot as plt
import os

def plot_performance(results_df, img_path=None):
    # Create figure
    fig, ax1 = plt.subplots(figsize=(10,5))

    
    #  Stock Price on left axis
    
    ax1.plot(results_df["Datetime"], results_df["Close"], label="Stock Price", color="blue")
    ax1.set_ylabel("Stock Price", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")

    
    #  Equity Curve on right axis
    
    ax2 = ax1.twinx()
    ax2.plot(results_df["Datetime"], results_df["capital"], label="Equity Curve", linestyle="--", color="orange")
    ax2.set_ylabel("Equity (start = 1.0)", color="orange")
    ax2.tick_params(axis='y', labelcolor="orange")

    plt.title("RL Trading Performance vs. Stock Price")
    fig.tight_layout()

    
    #  SAVE the plot (for W&B)
    
    if img_path is not None:
        directory = os.path.dirname(img_path)
        os.makedirs(directory, exist_ok=True)

        plt.savefig(img_path, dpi=300)
        print(f"[OK] Plot saved to: {img_path}")

    plt.close()



