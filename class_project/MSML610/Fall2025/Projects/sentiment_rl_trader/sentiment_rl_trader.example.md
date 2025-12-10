Sentiment RL Trader — Example Walkthrough

This document provides a full walkthrough of the Sentiment RL Trader project using a real stock ticker (AAPL).
The goal is to demonstrate how sentiment analysis and reinforcement learning can be combined into a trading strategy.

The example pipeline includes:

    1. Fetching news articles

    2. Loading historical stock prices

    3. Computing sentiment scores

    4. Merging sentiment with the price timeline

    5. Training a Q-learning reinforcement learning agent

    6. Running a simulated trading strategy

    7. Inspecting performance metrics and charts

All logic is accessed through the functions defined in sentiment_rl_trader_utils.py.

## 1. Load Sentiment + Stock Data

We begin by fetching:

    1. News headlines for the selected ticker

    2. Historical price data from Yahoo Finance

    3. These are automatically merged into a single DataFrame using timestamp alignment.

## merged_df = load_sentiment_stock_data("AAPL")

The resulting dataset contains:

    1. News headlines

    2. Computed sentiment scores

    3. Stock prices

The aligned timeline used for modeling

## 2. Train the Reinforcement Learning Agent

We train a Q-learning agent that decides whether to:

    1. Go LONG

    2. Go SHORT

    3. Stay FLAT

The agent learns based on:

    1. Sentiment values

    2. Future returns

    3. Reward-based feedback

## Q_table = train_rl_model(merged_df)


The Q-table represents learned action values for each sentiment/position state.


## 3. Run the Trading Simulation

Using the trained Q-table, we simulate a trading strategy over the full timeline:

## results_df, stats = run_rl_simulation(merged_df, Q_table)


The output includes:

    1. Capital over time

    2. Positions taken

    3. Timestamps

    4. Full equity curve

All trades implicitly captured by position changes

## 4. Evaluate Performance

We review key performance metrics:

    1. Final Return — total capital growth

    2. Win Rate — profitable trades

    3. Sharpe Ratio — risk-adjusted return

    4. Max Drawdown — worst capital drop

    5. Total Trades — number of actions taken

## stats


These metrics help us evaluate how well the RL agent behaved in this sentiment-driven environment.

## 5. Visualization

We generate an equity curve showing the growth of capital over time:

## plot_performance(results_df, "equity_curve_example.png")


This plot clearly illustrates:

    1. When the model gained money

    2. When it lost money

    3. Whether it was stable or volatile

## 6. Summary

This example shows:

How sentiment extracted from news can serve as a trading signal

How reinforcement learning can adaptively learn trading actions

How the model performs over real market data

The system is modular and can be extended with:

    1. LSTM-based predictions

    2. More advanced RL algorithms

    3. Larger datasets

Alternative sentiment models

This example serves as a complete demonstration of the project’s functionality.