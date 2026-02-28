# TextBlob

## Description

- TextBlob is a Python library for processing textual data with a simple,
  intuitive API built on top of NLTK and the Pattern library
- Provides out-of-the-box sentiment analysis returning polarity (−1.0 to 1.0)
  and subjectivity (0.0 to 1.0) scores per text blob
- Supports part-of-speech tagging, noun phrase extraction, spelling correction,
  translation, and word inflection
- Integrates easily with Pandas DataFrames for batch processing of large text
  corpora
- NewsAPI (free tier, requires email registration) gives access to articles from
  over 30,000 sources, filterable by keyword, language, and date

## Project Objective

- Create a pipeline to:
  - Ingest Bitcoin-related news articles in batch using NewsAPI
  - Analyze sentiment with TextBlob
  - Integrate sentiment scores with Bitcoin price data for predictive
    time-series analysis

## Dataset Suggestions

- **News articles**: NewsAPI free tier — query with keywords such as
  *Bitcoin, BTC, crypto market*; returns title, description, source, and
  publish date
- **Bitcoin price history**: CoinGecko public REST API
  (`/coins/bitcoin/market_chart`) — free, no API key required, returns OHLC and
  volume at hourly or daily granularity
- **Alternative price dataset**: Kaggle —
  [Bitcoin Historical Data](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)
  (minute-level OHLCV, 2012–present)
- **Alternative news corpus**: GDELT Project (`gdeltproject.org`) — open,
  no authentication, global news event dataset with sentiment scores for
  baseline comparison

## Tasks

- **Set Up Environment**: Install `textblob`, `newsapi-python`, `pandas`, and
  `requests`; obtain a free NewsAPI key; verify connectivity with a sample
  headline query
- **Ingest News Data**: Query NewsAPI for Bitcoin-related articles, normalize
  text fields, and persist results to a Pandas DataFrame (CSV or Parquet)
- **Perform Sentiment Analysis**: Preprocess article text (lowercase, remove
  punctuation, strip stopwords); compute per-article polarity and subjectivity;
  aggregate into hourly and daily averages
- **Integrate with Bitcoin Price Data**: Fetch historical OHLCV data from
  CoinGecko; align timestamps with the sentiment dataset; merge into a single
  time-indexed DataFrame
- **Time-Series Analysis**: Engineer lag features; fit ARIMA/SARIMA models;
  compare price-only vs. sentiment-enhanced forecasts using RMSE and MAE
- **Visualization**: Plot sentiment trends overlaid on Bitcoin price movements;
  produce rolling-correlation charts highlighting periods where sentiment leads
  price

## Bonus Ideas

- Replace TextBlob with a pre-trained transformer (e.g.,
  `ProsusAI/finbert` on HuggingFace) and compare sentiment scores and
  downstream forecast accuracy
- Add anomaly detection (e.g., Isolation Forest) on the combined
  price-sentiment time series to flag unusual market events
- Backtest a simple trading signal derived from sentiment polarity spikes and
  evaluate it against a buy-and-hold baseline
- Extend the pipeline to a second cryptocurrency (e.g., Ethereum) and compare
  cross-asset sentiment correlation

## Useful Resources

- [TextBlob Documentation](https://textblob.readthedocs.io/en/dev/)
- [NewsAPI Python Client Library](https://github.com/mattlisiv/newsapi-python)
- [CoinGecko API Reference](https://www.coingecko.com/api/documentation)
- [GDELT Project](https://www.gdeltproject.org/)
- TextBlob is open-source and free; NewsAPI free tier is available for
  educational use (registration required, usage limits apply)
