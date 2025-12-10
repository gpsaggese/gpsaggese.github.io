# sentiment_rl_trader_utils.py
# Utility API layer for notebooks and demonstrations

from data_fetcher import fetch_news, fetch_stock_history, align_news_with_stock
from sentiment_model import add_sentiment
from rl_trader import train_q_learning, run_strategy
from plotter import plot_performance


# -------------------------------------------------------------------
# 1. Load merged sentiment + stock dataset
# -------------------------------------------------------------------
def load_sentiment_stock_data(ticker: str = "AAPL"):
    """
    Fetch news, stock prices, compute sentiment, and return a merged DataFrame.
    Used by both API and Example notebooks.
    """
    news_df = fetch_news(ticker)
    stock_df = fetch_stock_history(ticker)

    news_df = add_sentiment(news_df)
    merged_df = align_news_with_stock(news_df, stock_df)

    return merged_df


# -------------------------------------------------------------------
# 2. Train RL (Q-learning) model
# -------------------------------------------------------------------
def train_rl_model(merged_df, episodes: int = 200):
    """
    Train the reinforcement learning agent and return the Q-table.
    """
    Q_table = train_q_learning(merged_df, episodes=episodes)
    return Q_table


# -------------------------------------------------------------------
# 3. Run RL trading simulation
# -------------------------------------------------------------------
def run_rl_simulation(merged_df, Q_table):
    """
    Runs the RL trading simulation using a trained Q-table.

    Returns:
        results_df (capital, positions, timestamps)
        stats (performance metrics)
    """
    results_df, stats = run_strategy(merged_df, Q_table)
    return results_df, stats


# -------------------------------------------------------------------
# 4. Create and save equity curve plot
# -------------------------------------------------------------------
# sentiment_rl_trader_utils.py

from plotter import plot_performance

def create_equity_plot(results_df, save_path="plots/equity_curve.png"):
    """
    Generate and save the equity curve plot using the shared plotting utility.
    Also displays the plot in Jupyter notebooks.
    """
    plot_performance(results_df, save_path)
    return save_path




