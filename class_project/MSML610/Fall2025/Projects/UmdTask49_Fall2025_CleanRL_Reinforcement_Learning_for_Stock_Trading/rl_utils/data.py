from typing import Tuple, Dict
import numpy as np
import pandas as pd
from rl_utils.data_handler import AlpacaDataHandler
from rl_utils.news_handler import NewsHandler
from rl_utils.indicators import calculate_indicators

# ==================== DATA PREPARATION UTILITIES ====================
def prepare_data_features(
    ticker: str, 
    start_date: str, 
    end_date: str, 
    timeframe: str,
) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """
    Fetches historical stock data, news embeddings, and news documents,
    calculates technical indicators, and prepares data for RL training.
    
    Args:
        ticker (str): Stock ticker symbol.
        start_date (str): Start date for historical data in 'YYYY-MM-DD'
        end_date (str): End date for historical data in 'YYYY-MM-DD'
        timeframe (str): Timeframe for historical data (e.g., '1D')
    
    Returns:
        Tuple containing:
            - DataFrame with OHLCV, technical indicators, returns, and news embeddings
            - Dict mapping index -> concatenated news text for that day
    """    
    data_handler = AlpacaDataHandler()
    df = data_handler.get_historical_data(ticker, start_date, end_date, timeframe)

    indicators_to_calculate = {
            'MOVING_AVERAGES': True,
            'BOLLINGER': False,
            'RSI': False,
            'MACD': False,
            'STOCHASTIC': False,
            'ADX': False,
            'DMI': True,
            'ATR': True,
            'VOLUME': True,
            'MOMENTUM': True,
            'OSCILLATORS': True,
            'RANGE': True,
            'STATISTICAL': False,
            'PERCENTILE': False,
            'PATTERNS': False,
        }
    
    df = calculate_indicators(df, indicators_to_calculate=indicators_to_calculate)
    df['return_1d'] = df['Close'].pct_change(1)
    df['return_3d'] = df['Close'].pct_change(3)
    df['return_5d'] = df['Close'].pct_change(5)
    df.dropna(inplace=True)  # Remove NaN rows from indicators/returns

    news_handler = NewsHandler(verbose=False)
    news_documents = {}
    
    # For each row in the dataframe, fetch previous 1 month's news articles and embeddings
    for idx, current_date in enumerate(df.index):
        # Calculate 1 month back from current date
        month_start = current_date - pd.Timedelta(days=30)
        month_end = current_date
        
        # Fetch news for the rolling 1-month window
        articles, embeddings = news_handler.fetch_news(
            ticker, 
            month_start, 
            month_end, 
            limit=50, 
            generate_embeddings=False
        )
        
        # Store news documents (for LDA training)
        if articles is not None and len(articles) > 0:
            # Concatenate all news text for this day
            news_text = " ".join([f"{article.get('headline', '')} {article.get('description', '')}" for article in articles])
            news_documents[idx] = news_text        

    return df, news_documents

