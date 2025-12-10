# rl_trader.py
# Reinforcement Learning trading using Q-learning with improved win-rate tracking.

import numpy as np
import pandas as pd
import wandb

# actions: -1 = short, 0 = flat, 1 = long
ACTIONS = [-1, 0, 1]
N_ACTIONS = 3

# number of sentiment bins (0 to 4)
N_SENTIMENT_BINS = 5

# positions: -1, 0, 1
N_POSITIONS = 3

# total states = sentiment bins × positions
N_STATES = N_SENTIMENT_BINS * N_POSITIONS


def discretize_sentiment(value):
    value = np.clip(value, -1, 1)
    bin_index = int((value + 1) / 2 * N_SENTIMENT_BINS)
    if bin_index == N_SENTIMENT_BINS:
        bin_index -= 1
    return bin_index


def get_state(sentiment, position):
    s_bin = discretize_sentiment(sentiment)
    pos_idx = position + 1  # -1→0, 0→1, 1→2
    return s_bin * N_POSITIONS + pos_idx


# Q-LEARNING TRAINING

def train_q_learning(df, episodes=200, alpha=0.1, gamma=0.95, epsilon=0.05):
    df = df.sort_values("Datetime").reset_index(drop=True)

    prices = df["Close"].values
    returns = pd.Series(prices).pct_change().fillna(0).values
    sentiments = df["sentiment"].values

    Q = np.zeros((N_STATES, N_ACTIONS))

    for ep in range(episodes):
        position = 0  # starting at 0

        for t in range(len(df) - 1):
            state = get_state(sentiments[t], position)

            # epsilon greedy exploration
            if np.random.rand() < epsilon:
                action_idx = np.random.randint(N_ACTIONS)
            else:
                action_idx = np.argmax(Q[state])

            new_pos = ACTIONS[action_idx]

            # Sentiment threshold -- prevents bad/noisy trades
            if abs(sentiments[t]) < 0.1:
                new_pos = 0

            # Reward = return * position
            reward = new_pos * returns[t + 1]

            # Small penalty for staying flat = 0 
            if new_pos == 0:
                reward -= 0.0001

            next_state = get_state(sentiments[t + 1], new_pos)
            best_next = np.max(Q[next_state])

            # Q-learning update
            Q[state, action_idx] += alpha * (reward + gamma * best_next - Q[state, action_idx])

            position = new_pos

    return Q



# RUNNING STRATEGY WITH CORRECT WIN RATE

def run_strategy(df, Q):
    df = df.sort_values("Datetime").reset_index(drop=True)

    prices = df["Close"].values
    returns = pd.Series(prices).pct_change().fillna(0).values
    sentiments = df["sentiment"].values

    capital = 1.0
    equity_curve = []
    positions = []

    # performance tracking
    trades = 0
    long_trades = 0
    short_trades = 0
    wins = 0
    losses = 0
    last_position = 0
    entry_price = None

    for t in range(len(df) - 1):
        state = get_state(sentiments[t], last_position)
        action_idx = np.argmax(Q[state])
        position = ACTIONS[action_idx]

        # Sentiment threshold during simulation
        if abs(sentiments[t]) < 0.1:
            position = 0
      
        # CORRECT TRADE OPEN/CLOSE LOGIC        
        if position != last_position:

            # Closing previous trade
            if last_position != 0 and entry_price is not None:
                exit_price = prices[t]
                pnl = (exit_price - entry_price) * last_position

                if pnl > 0:
                    wins += 1
                else:
                    losses += 1

            # Opening a new trade
            if position == 1:
                long_trades += 1
                entry_price = prices[t]
            elif position == -1:
                short_trades += 1
                entry_price = prices[t]
            else:
                entry_price = None  # going flat

            trades += 1

        last_position = position

        # Update capital
        capital *= (1 + position * returns[t + 1])

        equity_curve.append(capital)
        positions.append(position)

        
        # W&B LOGGING FOR VISUALIZATIONS (Sentiment, Price...)
        
        wandb.log({
            "Datetime": df.loc[t, "Datetime"].timestamp(), # convert to string
            "Close_Price": prices[t],
            "Sentiment": sentiments[t],
            "Capital": capital,
            "Position": position
        })


    
    # PERFORMANCE METRICS
    
    final_return = (capital - 1) * 100

    equity_arr = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_arr)
    drawdown = (equity_arr - running_max) / running_max
    max_dd = drawdown.min() * 100

    daily_returns = pd.Series(equity_arr).pct_change().dropna()
    sharpe = (
        (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        if not daily_returns.empty else 0
    )

    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    
    # PRINT PERFORMANCE SUMMARY
    
    print("\n==============================")
    print("      RL PERFORMANCE SUMMARY")
    print("==============================")
    print(f"Total Trades:           {trades}")
    print(f"Long Trades:            {long_trades}")
    print(f"Short Trades:           {short_trades}")
    print(f"Win Rate:               {win_rate:.2f}%")
    print(f"Final Return:           {final_return:.2f}%")
    print(f"Max Drawdown:           {max_dd:.2f}%")
    print(f"Sharpe Ratio:           {sharpe:.3f}")
    print(f"Start Equity:           1.0000")
    print(f"End Equity:             {capital:.4f}")
    print("==============================\n")

    # Build output df
    out = df.iloc[1:].copy().reset_index(drop=True)
    out["capital"] = equity_curve
    out["position"] = positions

    # Stats dictionary (used for dashboard charts)
    perf_stats = {
        "Final_Return": final_return,
        "Win_Rate": win_rate,
        "Sharpe_Ratio": sharpe,
        "Max_Drawdown": max_dd,
        "Total_Trades": trades,
        "Long_Trades": long_trades,
        "Short_Trades": short_trades
    }

    return out, perf_stats




