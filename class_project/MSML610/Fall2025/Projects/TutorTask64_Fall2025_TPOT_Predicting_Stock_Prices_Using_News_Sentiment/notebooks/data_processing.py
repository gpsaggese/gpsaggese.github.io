"""
data_processing.py
Code to build out the necessary data for TPOT

Author: Bradley Scott
Date: October - December 2025
"""

'''
[BS11052025] dp_610_000001
[BS11052025] import all necessary modules
'''
import pandas as pd
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import yfinance as yf #yfinance < 2.60 will cause errors if using docker(cause bot detection)
from datetime import datetime
import time
import polars as pl
import gc

'''
[BS11102025] dp_610_000005 
[BS11102025] set file paths
'''
# data directory
BASE = Path("/workspace/data") 

# local file paths
file_daily_news = BASE / "daily_news_sentiment.parquet"
file_val_tick = BASE / "valid_tickers.csv"
file_prices = BASE / "ticker_price.parquet"
file_model = BASE / "model_data.parquet"

'''
[BS11052025] dp_610_000015
[BS11052025] Build out functions to use VADER for sentiment analysis
    NB: Will use textrank_summary if available, otherwise will use title
'''
tqdm.pandas()

analyzer = SentimentIntensityAnalyzer()

def pick_text(row):
    # prefer Textrank_summary; fallback to title; finally to Article
    for col in ["Textrank_summary", "Article_title", "Article"]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col])
    return ""

def vader_score(text: str) -> float:
    if not text:
        return 0.0
    return analyzer.polarity_scores(text)["compound"]

'''
[BS11052025] dp_610_000020
[BS11052025] Do the sentiment analysis and build out daily_news_sentiment.parquet
    Streams directly from Hugging Face (file is 3 GB so can't store in the repo)
    NB: I had to offload the build of this to google colab since my computer didn't have the RAM to run it
'''
chunksize = 200_000 

# Only create the file if it doesn't already exist
if not file_daily_news.exists():
    print("daily_news_sentiment.parquet does NOT exist — creating it...")
    print("Streaming from Hugging Face...")
    
    # Direct URL to CSV on Hugging Face
    huggingface_url = "https://huggingface.co/datasets/Zihan1004/FNSPID/resolve/main/Stock_news/nasdaq_exteral_data.csv"
    
    summary_cols = ["Textrank_summary"]   
    base_cols = ["Date", "Article_title", "Stock_symbol", "Article"]
    usecols = list(dict.fromkeys(base_cols + summary_cols))
    
    writer = None
    
    try:
        # Stream in chunks from Hugging Face
        for i, chunk in enumerate(pd.read_csv(
                huggingface_url,          
                chunksize=chunksize,
                usecols=lambda c: True,
                encoding_errors="ignore",
                low_memory=False          
            )):
            
            cols_present = [c for c in usecols if c in chunk.columns]
            df = chunk[cols_present].copy()
            
            # Clean date
            date_col = "Date"
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date
            df = df.dropna(subset=[date_col])
            
            # Expand tickers
            df["Stock_symbol"] = df["Stock_symbol"].astype(str).str.replace(" ", "")
            df = df[df["Stock_symbol"].str.len() > 0]
            df = df.assign(Stock_symbol=df["Stock_symbol"].str.split(",")).explode("Stock_symbol")
            df["Stock_symbol"] = df["Stock_symbol"].str.upper().str.replace(r"[^A-Z\.]", "", regex=True)
            df = df[df["Stock_symbol"].str.len() > 0]
            
            # Sentiment
            df["_text"] = df.apply(pick_text, axis=1)
            df["_sent"] = df["_text"].map(vader_score).astype("float32")
            
            # Daily aggregation
            daily = (
                df.groupby(["Stock_symbol", date_col], as_index=False)
                  .agg(avg_sentiment=("_sent", "mean"),
                       pos_count=("_sent", lambda s: np.sum(s > 0.05)),
                       neg_count=("_sent", lambda s: np.sum(s < -0.05)),
                       news_count=("_text", "count"))
            )
            daily = daily.rename(columns={date_col: "date", "Stock_symbol": "ticker"})
            
            # Append to parquet
            table = pa.Table.from_pandas(daily, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(file_daily_news, table.schema)
            writer.write_table(table)
            
            print(f"Processed chunk {i}: wrote {len(daily)} daily rows")
            
            # Free memory after each chunk
            del df, daily, table, chunk
            gc.collect()
        
        if writer is not None:
            writer.close()
        
        print("File created:", file_daily_news)
        print(f"Streaming complete - processed from Hugging Face")
        
    except Exception as e:
        print(f"Error processing from Hugging Face: {e}")
        if writer is not None:
            writer.close()
        raise

else:
    print("daily_news_sentiment.parquet already exists — skipping creation")
    print("   (Original data from Hugging Face)")

'''
[BS11072025] dp_610_000025
[BS11052025] Load the sentiment analysis for any tickers with data for more than 200 days
'''
# Load daily sentiment 
news = pd.read_parquet(file_daily_news)

# Format date
news['date'] = pd.to_datetime(news['date']).dt.date

# Filter to past 10 years 
news = news[(news['date'] >= datetime(2014,1,1).date())]

# Clean ticker column: drop NaN and "NAN", "", etc.
news = news[ news['ticker'].notna() ]               # removes actual NaN
news = news[ news['ticker'].str.upper() != "NAN" ]  # removes literal "NAN"
news = news[ news['ticker'].str.strip() != "" ]     # removes blanks

# Remove invalid tickers (must be A–Z or dot)
is_valid = news['ticker'].str.match(r'^[A-Z\.]+$')
news = news[is_valid.fillna(False)]

# Start with tickers that have enough coverage for ML dev
min_days = 200
ticker_counts = news['ticker'].value_counts()
tickers = ticker_counts.loc[ticker_counts >= min_days].index.tolist()

len(tickers), tickers[:10]

'''
[BS11072025] dp_610_000030
[BS11072025] yFinance does not contain info for most delisted stocks. 
             Of the stocks we have enough data from in the past 10 years, we need to remove
             all the delisted stocks. This will ping yFinance to check if the ticker exists
             and create a list of delisted stocks that we need to remove from our data.
             This took 18 minutes to run so save off the file if it doesn't already exist, otherwise skip.
'''
GLOBAL_START = "2014-01-01"
GLOBAL_END   = "2024-01-09"

def has_prices_in_window(ticker, start=GLOBAL_START, end=GLOBAL_END, max_retries=2):
    """Return True if YF returns any rows for (ticker, start..end)."""
    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                threads=False,
                progress=False,
            )
            return not df.empty
        except Exception:
            time.sleep(2 * attempt)  
    return False

# if the file exists load and skip processing

if os.path.exists(file_val_tick):
    print(f" Found existing valid ticker file: {file_val_tick}")
    valid_tickers = pd.read_csv(file_val_tick)["ticker"].tolist()
    print(f"Loaded {len(valid_tickers)} valid tickers.")
    
else:
    # file does not exist, create file and save results
    print("No valid ticker file found. Running yfinance validation step...")
    
    valid_tickers = []
    invalid_tickers = []

    for t in tickers:      # 'tickers' is the filtered high-coverage list
        ok = has_prices_in_window(t)
        (valid_tickers if ok else invalid_tickers).append(t)

    print(f"Valid: {len(valid_tickers)}  Invalid: {len(invalid_tickers)}")

    # Save the valid ones for next time
    pd.Series(valid_tickers, name="ticker").to_csv(file_val_tick, index=False)
    print(f"Saved valid tickers to {file_val_tick}")

'''
[BS11072025] dp_610_000035
[BS11072025] Load valid tickers and narrow sentiment down to just valid tickers
'''
# Load validated tickers  
valid_tickers = pd.read_csv(file_val_tick)["ticker"].tolist()

# Keep only sentiment rows with validated tickers
news = news[news['ticker'].isin(valid_tickers)]

'''
[BS11072025] dp_610_000040
[BS11072025] Get the earliest and latest sentiment date per ticker
    NB: I pad the start and end by 30 days for computing rolling volatility and lagged returns
'''
win = (
    news.groupby("ticker")["date"]
        .agg(min_date='min', max_date='max')
        .reset_index()
)

pad = pd.Timedelta(days=30)
win["start"] = pd.to_datetime(win["min_date"]) - pad
win["end"]   = pd.to_datetime(win["max_date"]) + pad

'''
[BS11072025] dp_610_000045
[BS11072025] If the file_prices data has not been saved off
             1) Pull the price data per valid ticker for the times done in step fp_610_000040
             2) put it all together and save it off to file_prices
             NB: Downloading in batches so as not to overload system and get timedout
             NB: 18 tickers failed because they've been delisted. Once delisted yfinance removes the data.
'''
# if file exists → load and skip downloading

if os.path.exists(file_prices):
    print(f"Found existing prices file: {file_prices}")
    prices = pd.read_parquet(file_prices)
    print(f"Loaded prices: {prices['ticker'].nunique()} tickers, {len(prices):,} rows")
    
else:
    # file does not exist so run full donwload
    print("No saved prices found — downloading from yfinance...")

    BATCH_SIZE = 25
    MAX_RETRIES = 4
    BASE_SLEEP = 8

    def chunked(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    frames = []

    for batch_idx, batch in enumerate(chunked(valid_tickers, BATCH_SIZE), start=1):

        # Per-batch date ranges
        sub = win[win["ticker"].isin(batch)]
        start = sub["start"].min().strftime("%Y-%m-%d")
        end   = sub["end"].max().strftime("%Y-%m-%d")

        tries = 0
        while True:
            try:
                print(f"[Batch {batch_idx}] Downloading {len(batch)} tickers...")
                px = yf.download(
                    batch,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    group_by="ticker",
                    progress=False,
                    threads=False
                )

                # Determine which tickers we got data for
                if isinstance(px.columns, pd.MultiIndex):
                    got = set(px.columns.get_level_values(0))
                else:
                    got = set(batch) if not px.empty else set()

                # For each ticker successfully returned, flatten & save
                if not px.empty and isinstance(px.columns, pd.MultiIndex):
                    for t in got:
                        p = px[t].reset_index().rename(columns={"Date": "date"})
                        p["ticker"] = t
                        p["date"] = pd.to_datetime(p["date"]).dt.date
                        frames.append(p)
                        downloaded_ok.add(t)

                # Missing tickers
                missing = [t for t in batch if t not in got]
                for t in missing:
                    failed_hard.append((t, "no_price_data_in_window"))

                break  # batch done

            except Exception as e:
                tries += 1
                if tries <= MAX_RETRIES:
                    sleep_s = BASE_SLEEP * (2 ** (tries - 1))
                    print(f"   Error: {e} — retrying in {sleep_s:.0f}s...")
                    time.sleep(sleep_s)
                else:
                    break

    print("Finished downloading price data")

    # create prices df
    prices = (
        pd.concat(frames, ignore_index=True)
          .rename(columns={
              "Open":"open","High":"high","Low":"low",
              "Close":"close","Volume":"volume"
          })
    )

    # Remove Adj Close if Yahoo inserted it
    # NB: adjusted close is the same as close or blank because we did audo_adj = True
    if "Adj Close" in prices.columns:
        prices = prices.drop(columns=["Adj Close"])

    # save for future use
    prices.to_parquet(file_prices, index=False)
    print(f"Saved prices to {file_prices}")

'''
[BS11072025] dp_610_000050
[BS11072025] Trim the price data to each ticker's window
'''
# merge windows
temp = win[["ticker", "min_date", "max_date"]].copy()
temp["min_date"] = pd.to_datetime(temp["min_date"]).dt.date
temp["max_date"] = pd.to_datetime(temp["max_date"]).dt.date

prices = prices.merge(temp, on="ticker", how="inner")
prices = prices[
    (prices["date"] >= prices["min_date"]) &
    (prices["date"] <= prices["max_date"])
]
prices = prices.drop(columns=["min_date","max_date"])

'''
[BS11072025] dp_610_000055
[BS11072025] Merge sentiment data and price data, fill 0 for non news days 
'''
df = prices.merge(
    news,
    on=["ticker", "date"],
    how="left"
).sort_values(["ticker", "date"])

# fill missing sentiment days with 0 indicating it is a news-free day
sent_cols = ["avg_sentiment","pos_count","neg_count","news_count"]
for c in sent_cols:
    df[c] = df[c].fillna(0.0)
'''
[BS11072025] fp_610_000060
[BS11072025] Feature engineer additional flags 
'''
if file_model.exists():
    print("Loading engineered model...")
    df_model = pl.read_parquet(file_model).to_pandas()
else:
    print("Running Polars optimized feature engineering...")
    # Ensure sorted data
    pl_df = pl.from_pandas(df).sort(["ticker", "date"])
    # Rolling helpers
    def roll(col, window, minp=1):
        return col.shift(1).rolling_mean(window_size=window, min_periods=minp)
    def roll_std(col, window, minp=1):
        return col.shift(1).rolling_std(window_size=window, min_periods=minp)
    
    out = (
        pl_df
        .with_columns([
    
            # Target: Tomorrow's return 
            (pl.col("close").pct_change().over("ticker").shift(-1)).alias("ret_1d"),
    
            # Sentiment features
            pl.col("avg_sentiment").shift(1).alias("sent_lag1"),
            roll(pl.col("avg_sentiment"), 5).alias("sent_roll5"),
            roll(pl.col("avg_sentiment"), 10).alias("sent_roll10"),
            
            pl.col("news_count")
                .shift(1)
                .rolling_sum(window_size=5, min_periods=3)
                .alias("news_count_roll5"),
    
            # Lagged returns 
            pl.col("close").pct_change().over("ticker").shift(1).alias("ret_1d_past"),
            pl.col("close").pct_change(5).over("ticker").shift(1).alias("ret_5d_past"),
            pl.col("close").pct_change(10).over("ticker").shift(1).alias("ret_10d_past"),
            pl.col("close").pct_change(20).over("ticker").shift(1).alias("ret_20d_past"),
    
            #  Volatility
            roll_std(pl.col("close").pct_change().over("ticker"), 10, minp=5).alias("price_vol10"),
            roll_std(pl.col("close").pct_change().over("ticker"), 20, minp=10).alias("price_vol20"),
    
            # - Binary sentiment flags 
            (pl.col("avg_sentiment").shift(1) > 0).cast(pl.Int8).alias("sent_pos"),
            (pl.col("avg_sentiment").shift(1) < 0).cast(pl.Int8).alias("sent_neg"),
    
            # Price features use YESTERDAY'S data
            (
                (pl.col("high").shift(1) - pl.col("low").shift(1)) / 
                pl.col("close").shift(1)
            ).alias("high_low_range"),
            
            (
                pl.col("close").shift(1) / pl.col("open").shift(1) - 1
            ).alias("close_open_ratio"),
    
            # Momentum uses past data only 
            pl.col("close").pct_change(5).over("ticker").shift(1).alias("momentum_5d"),
            pl.col("close").pct_change(10).over("ticker").shift(1).alias("momentum_10d"),
        ])
        .lazy()
        .collect()
    )
    
    #  Filter on ret_1d 
    df_model = out.filter(pl.col("ret_1d").is_not_null()).to_pandas()
    
    df_model.to_parquet(file_model)
    print("Saved engineered model.")

print(f"\nLoaded {len(df_model):,} observations")
print(f"   Tickers: {df_model['ticker'].nunique()}")
print(f"   Date range: {df_model['date'].min()} to {df_model['date'].max()}")
print(f"   Features: {len(df_model.columns)}")