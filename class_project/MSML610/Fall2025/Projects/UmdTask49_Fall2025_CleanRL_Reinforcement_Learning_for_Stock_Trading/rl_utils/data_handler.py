# Alpaca-py imports
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

import pandas as pd
import os
from dotenv import load_dotenv


# Load environment variables from .env file located in the project root
# Assuming this config.py is in options-strategy, and .env is one level up.
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path)
else:
    print(f"Warning: .env file not found at {dotenv_path}. Using default or environment-set credentials.")

# =============================================================================
# Alpaca Data Handler
# =============================================================================
import hashlib
from datetime import datetime
import pickle
from pathlib import Path

class AlpacaDataHandler:
    """
    Fetches underlying historical data from Alpaca with local caching support.
    """
    def __init__(self, cache_dir='data_cache'):
        self.trading_client = TradingClient(
            api_key=os.getenv("ALPACA_API_KEY", ""),
            secret_key=os.getenv("ALPACA_API_SECRET", ""),
            paper=True
        )
        self.stock_client = StockHistoricalDataClient(
            api_key=os.getenv("ALPACA_API_KEY", ""),
            secret_key=os.getenv("ALPACA_API_SECRET", ""),
        )
        
        # Create cache directory if it doesn't exist
        self.cache_dir = Path(os.path.join(os.path.dirname(__file__), '.', cache_dir))
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Cache expiry in days (set to 1 day by default)
        self.cache_expiry_days = 1

    def _map_interval_to_timeframe(self, interval: str) -> TimeFrame:
        interval = interval.lower().replace("min", "m").replace("minute", "m")
        if interval in ["1m", "1"]:
            return TimeFrame.Minute
        elif interval in ["5m", "5"]:
            return TimeFrame(5, TimeFrameUnit.Minute)
        elif interval in ["15m", "15"]:
            return TimeFrame(15, TimeFrameUnit.Minute)
        elif interval in ["1h", "60"]:
            return TimeFrame.Hour
        elif interval in ["3h", "120"]:
            return TimeFrame(3, TimeFrameUnit.Hour)
        elif interval in ["1d", "d"]:
            return TimeFrame.Day
        else:
            return TimeFrame(5, TimeFrameUnit.Minute)

    def _generate_cache_key(self, ticker, start, end, interval):
        """Generate a unique cache key based on request parameters"""
        key_string = f"{ticker}_{start}_{end}_{interval}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_path(self, cache_key):
        """Get the full path to the cache file"""
        return self.cache_dir / f"{cache_key}.pkl"

    def _is_cache_valid(self, cache_path):
        """Check if the cache file exists and is not expired"""
        if not cache_path.exists():
            return False
            
        # Check if cache is expired
        cache_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        current_time = datetime.now()
        diff_days = (current_time - cache_time).days
        
        return diff_days < self.cache_expiry_days

    def get_historical_data(self, ticker: str, start: str, end: str, interval: str, use_cache=True) -> pd.DataFrame:
        """
        Get historical data with caching support.
        
        Args:
            ticker: Symbol to fetch data for
            start: Start date/time
            end: End date/time
            interval: Data interval (1m, 5m, 15m, 1h, 1d)
            use_cache: Whether to use cached data (default: True)
            
        Returns:
            DataFrame with historical price data
        """
        # Generate cache key and path
        cache_key = self._generate_cache_key(ticker, start, end, interval)
        cache_path = self._get_cache_path(cache_key)
        
        # Try to load from cache if enabled
        if use_cache and self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except (IOError, pickle.PickleError):
                # If there's an issue with the cache file, fetch fresh data
                pass
                
        # Fetch fresh data from API
        timeframe = self._map_interval_to_timeframe(interval)
        
        req = StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=timeframe,
            start=start,
            end=end
        )
        bars_dict = self.stock_client.get_stock_bars(req)
        if ticker != bars_dict[ticker][0].symbol or len(bars_dict[ticker]) == 0:
            raise ValueError(f"No daily bars returned for {ticker}")

        def mapBarToDict(x):
            return {
                'symbol': x.symbol,
                'timestamp': x.timestamp,
                'open': x.open,
                'high': x.high,
                'low': x.low,
                'close': x.close,
                'volume': x.volume,
            }
        df = pd.DataFrame(map(mapBarToDict, bars_dict[ticker]))
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        
        # Save to cache
        if use_cache:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(df, f)
            except (IOError, pickle.PickleError) as e:
                print(f"Warning: Failed to cache data: {e}")
                
        return df

    def clear_cache(self, older_than_days=None):
        """
        Clear the cached data.
        
        Args:
            older_than_days: If provided, only clear cache files older than this many days
        """
        for cache_file in self.cache_dir.glob("*.pkl"):
            if older_than_days is not None:
                cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
                current_time = datetime.now()
                diff_days = (current_time - cache_time).days
                
                if diff_days >= older_than_days:
                    os.remove(cache_file)
            else:
                os.remove(cache_file)