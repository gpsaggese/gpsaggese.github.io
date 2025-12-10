# config.py
# This file stores the main settings for the sentiment + LSTM + RL trading project.

# News API key 
NEWS_API_KEY = "04e89446dfe146528bff504599f66b00"

# LSTM settings
WINDOW_SIZE = 15     # how many past sentiment values we use
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 8

# Reinforcement learning settings
RL_EPISODES = 500    # how many times the RL agent trains
RL_ALPHA = 0.05      # learning rate
RL_GAMMA = 0.95      # discount factor
RL_EPSILON = 0.1     # exploration rate

# Data fetch settings
NEWS_ARTICLE_COUNT = 100
STOCK_PERIOD = "1mo"
STOCK_INTERVAL = "1h"
