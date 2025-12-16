import yfinance as yf
import pandas as pd
import os

def fetch_stock_data(ticker: str, start_date: str, end_date: str, cache_dir: str = "data") -> pd.DataFrame:
    """
    Fetches historical stock data from Yahoo Finance.
    Caches the data to a CSV file to avoid repeated API calls.
    
    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        cache_dir (str): Directory to save/load cached data.
        
    Returns:
        pd.DataFrame: The stock data.
    """
    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, f"{ticker}_{start_date}_{end_date}.csv")
    
    if os.path.exists(file_path):
        print(f"Loading data from cache: {file_path}")
        # Explicitly ignore headers that might cause MultiIndex issues if reading raw
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # CLEANUP: If the CSV was saved with MultiIndex headers (Price, Ticker), it might look like rows 0 and 1 are headers.
        # We check if the index is object or if first few rows are strings.
        # A robust way is to just force numeric conversion on all columns.
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows that became NaNs (likely the header rows from the cache)
        df.dropna(inplace=True)

    else:
        print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
        df = yf.download(ticker, start=start_date, end=end_date)
        
        # FLATTEN: yfinance often returns MultiIndex columns (Price, Ticker). We just want the 'Price' level.
        if isinstance(df.columns, pd.MultiIndex):
            # If 'Close' is in the columns, we try to extract it.
            # Usually level 0 is 'Price' (Close, Open...) and level 1 is 'Ticker'
            try:
                df = df.xs(ticker, axis=1, level=1, drop_level=True)
            except KeyError:
                # Sometimes it is just level 0
                pass
        
        # DOUBLE CHECK: Ensure we only have numeric data (no 'Ticker' columns if flatten failed)
        for col in df.columns:
             df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(inplace=True)
        df.to_csv(file_path)
        print(f"Data saved to {file_path}")
        
    return df
