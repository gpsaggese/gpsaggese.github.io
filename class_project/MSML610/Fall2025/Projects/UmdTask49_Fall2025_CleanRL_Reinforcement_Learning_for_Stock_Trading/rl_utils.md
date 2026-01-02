# RL Utils Breakdown

This folder contains utility scripts for data fetching, processing, and feature engineering, originally adapted from a quantitative strategy project.

## File Descriptions

### `rl_utils/data.py`

- **Purpose**: Orchestrates the data preparation process.
- **Functionality**:
  - Fetches historical stock data using `AlpacaDataHandler`.
  - Fetches news data using `NewsHandler`.
  - Calculates technical indicators using `calculate_indicators`.
  - Merges all data into a single DataFrame for training.
- **Key Function**: `prepare_data_features(ticker, start_date, end_date, timeframe)`

### `rl_utils/data_handler.py`

- **Purpose**: Manages interactions with the Alpaca API for historical market data.
- **Functionality**:
  - Authenticates using API keys from `.env`.
  - Fetches OHLCV (Open, High, Low, Close, Volume) data.
  - Implements local caching to reduce API calls and speed up subsequent runs.
  - Handles timeframe conversions.

### `rl_utils/news_handler.py`

- **Purpose**: Manages interactions with the Polygon.io API for financial news.
- **Functionality**:
  - Authenticates using API keys from `.env`.
  - Fetches news articles related to specific tickers.
  - Implements caching for news articles and embeddings.
  - Processes text data (though advanced processing happens in the News Interpreter).

### `rl_utils/indicators.py`

- **Purpose**: Calculates technical indicators for market data.
- **Functionality**:
  - Uses `numba` for optimized, high-performance calculations.
  - Computes indicators like Moving Averages, RSI, MACD, Bollinger Bands, ATR, etc.
  - Provides a suite of functions to generate features for the LSTM and RL models.

## Important Configuration: `.env` and API Keys

**Crucial Note**: This project relies on external APIs (Alpaca and Polygon.io) which require valid API keys.

1.  **`.env` File**: You must have a `.env` file in the project root (or parent directory depending on setup) containing:
    ```
    ALPACA_API_KEY=your_alpaca_key
    ALPACA_API_SECRET=your_alpaca_secret
    POLYGON_API_KEY=your_polygon_key
    ```
2.  **API Key Expiration**:
    - If you are using free or trial keys, they may expire or have rate limits.
    - **Existing keys in the provided environment may be invalidated after 1 month/30 days.**
    - For long-term usage, you will need to generate your own API keys from [Alpaca](https://alpaca.markets/) and [Polygon.io](https://polygon.io/).
