 1. Data API
### fetch_news(ticker: str) -> pd.DataFrame

Fetch recent news articles for the given stock ticker.

Returns a DataFrame with:

title

description

publishedAt

### fetch_stock_history(ticker: str) -> pd.DataFrame

Load historical stock data using Yahoo Finance.

Returns columns such as:

Open

High

Low

Close

Volume

Datetime

### align_news_with_stock(news_df, stock_df) -> pd.DataFrame

Align news timestamps with stock price timestamps using merge_asof.

Returns a merged DataFrame containing both sentiment and market data.


 2. Sentiment API
### add_sentiment(news_df) -> pd.DataFrame

Compute VADER sentiment scores for each news article.

Adds columns:

compound

sentiment (normalized)

3. Reinforcement Learning API
### train_q_learning(df, episodes=200, ...) -> Q_table

Train a Q-learning agent using sentiment and future returns.

Input DataFrame must include:

sentiment

Close prices

Returns:

A Q-table of shape (num_states × num_actions)

### run_strategy(df, Q_table) -> (results_df, perf_stats)

Run a trading simulation using the trained RL agent.

Returns:

results_df — positions, equity curve, timestamps

perf_stats — dictionary with:

Final_Return

Win_Rate

Sharpe_Ratio

Max_Drawdown

Total_Trades

4. Utility API (Notebook-Friendly)
### run_full_pipeline(ticker: str, skip_lstm=True)

High-level helper to run the entire workflow.

Pipeline steps:

Fetch news

Fetch stock prices

Compute sentiment

Merge datasets

Train RL agent

Simulate trades

Returns:

merged_df

results_df

stats

This wrapper is specifically used in API.ipynb and example.ipynb.

5. Plotting API
### plot_performance(results_df, save_path)

Generate a trading equity-curve visualization and save it to a file.

 
Notes

This API is intentionally clean and minimal, separating interface from implementation.

LSTM training runs only locally (TensorFlow excluded from Docker).

RL-based trading and sentiment analysis work fully in Docker.

The stable API allows future upgrades without breaking user code.