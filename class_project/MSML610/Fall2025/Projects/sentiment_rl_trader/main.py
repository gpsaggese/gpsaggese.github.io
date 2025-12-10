# main.py
# This file runs the whole project: getting news + stock data,
# doing sentiment, training the LSTM (local only), training the RL agent,
# saving the final plot, and logging everything to Weights & Biases.

import os
import wandb

from data_fetcher import fetch_news, fetch_stock_history, align_news_with_stock
from sentiment_model import add_sentiment
from rl_trader import train_q_learning, run_strategy
from plotter import plot_performance
from config import LSTM_EPOCHS, RL_EPISODES

# Detect Docker environment — Docker has a special file "/.dockerenv"
IN_DOCKER = os.path.exists("/.dockerenv")


def run_pipeline(ticker="AAPL"):

    # W&B run start
    wandb.init(
        project="sentiment_rl_trader",
        name=f"{ticker}_run",
        config={
            "ticker": ticker,
            "lstm_epochs": LSTM_EPOCHS,
            "rl_episodes": RL_EPISODES
        }
    )

    print("Fetching news articles...")
    news_df = fetch_news(ticker)

    print("Getting stock data...")
    stock_df = fetch_stock_history(ticker)

    print("Adding sentiment scores...")
    news_df = add_sentiment(news_df)

    print("Matching news with stock prices...")
    merged_df = align_news_with_stock(news_df, stock_df)

    # -----------------------------------------
    # LOCAL-ONLY LSTM TRAINING
    # -----------------------------------------
    if not IN_DOCKER:
        print("Training LSTM model (LOCAL ONLY)...")

        # Import TensorFlow model ONLY when not in Docker
        from lstm_model import train_lstm

        lstm_model, scaler = train_lstm(merged_df)
    else:
        print("Skipping LSTM model inside Docker.")

    print("Training RL agent...")
    Q_table = train_q_learning(merged_df)

    print("Running the trading simulation...")
    results_df, perf_stats = run_strategy(merged_df, Q_table)

    # send all performance stats to wandb (final return, win rate etc)
    wandb.log(perf_stats)

    # -----------------------------------------
    # PATH FOR PLOT SAVING
    # -----------------------------------------
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(BASE_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    img_path = os.path.join(plot_dir, f"{ticker}_equity.png")

    print("Plotting graph and saving...")
    plot_performance(results_df, img_path)

    # upload the saved image to wandb
    wandb.log({"equity_curve_plot": wandb.Image(img_path)})

    print("Done with everything!")


if __name__ == "__main__":
    run_pipeline("AAPL")




