# data_fetcher.py
# This file handles getting news data and stock data, 
# and then matching them together based on time.

import pandas as pd
import yfinance as yf
from newsapi import NewsApiClient
from config import (
    NEWS_API_KEY,
    NEWS_ARTICLE_COUNT,
    STOCK_PERIOD,
    STOCK_INTERVAL
)


# fetch news using NewsAPI
def fetch_news(query="AAPL"):
    api = NewsApiClient(api_key=NEWS_API_KEY)

    data = api.get_everything(
        q=query,
        language="en",
        sort_by="publishedAt",
        page_size=NEWS_ARTICLE_COUNT
    )

    articles = []
    for a in data["articles"]:
        articles.append({
            "title": a["title"],
            "description": a["description"],
            "publishedAt": a["publishedAt"]
        })

    return pd.DataFrame(articles)


# fetch stock data using yfinance
def fetch_stock_history(ticker="AAPL"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=STOCK_PERIOD, interval=STOCK_INTERVAL)
    df = df.reset_index()

    # converting to UTC to match news time
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)

    return df


# align news timestamps with stock timestamps
def align_news_with_stock(news_df, stock_df):
    news_df["publishedAt"] = pd.to_datetime(news_df["publishedAt"], utc=True)

    # merge_asof matches each news article to the nearest stock price time
    merged = pd.merge_asof(
        news_df.sort_values("publishedAt"),
        stock_df.sort_values("Datetime"),
        left_on="publishedAt",
        right_on="Datetime",
        direction="nearest"
    )

    return merged
