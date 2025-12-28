import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, List, Optional, Dict


def load_stock_data(csv_path: str, date_column: str = 'Date') -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=[date_column])
    return df


def parse_and_sort_dates(df: pd.DataFrame, date_column: str = 'Date') -> pd.DataFrame:
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values(date_column).reset_index(drop=True)
    return df


def handle_missing_values(df: pd.DataFrame, method: str = 'forward_fill',
                         columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Handle missing values using forward fill or interpolation.

    Args:
        df: Input DataFrame
        method: 'forward_fill' or 'interpolate'
        columns: Columns to process (None for all numeric columns)

    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if method == 'forward_fill':
        df[columns] = df[columns].fillna(method='ffill')
        df[columns] = df[columns].fillna(method='bfill')
    elif method == 'interpolate':
        df[columns] = df[columns].interpolate(method='linear')
        df[columns] = df[columns].fillna(method='bfill')

    return df


def detect_and_handle_outliers(df: pd.DataFrame, columns: List[str],
                               method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
    """
    Detect and handle outliers using IQR or z-score method.

    Args:
        df: Input DataFrame
        columns: Columns to check for outliers
        method: 'iqr' or 'zscore'
        threshold: Threshold for outlier detection (1.5 for IQR, 3 for z-score)

    Returns:
        DataFrame with outliers capped
    """
    df = df.copy()

    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
        elif method == 'zscore':
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std

        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df


def split_data_chronological(df: pd.DataFrame, train_ratio: float = 0.7,
                            val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically into train, validation, and test sets.

    Args:
        df: Input DataFrame
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def normalize_data(df: pd.DataFrame, columns: List[str],
                  scaler_type: str = 'minmax') -> Tuple[pd.DataFrame, object]:
    """
    Normalize data using MinMaxScaler or StandardScaler.

    Args:
        df: Input DataFrame
        columns: Columns to normalize
        scaler_type: 'minmax' or 'standard'

    Returns:
        Tuple of (normalized_df, scaler)
    """
    df = df.copy()

    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")

    df[columns] = scaler.fit_transform(df[columns])

    return df, scaler


def calculate_moving_averages(df: pd.DataFrame, price_column: str = 'Close',
                             windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
    """
    Calculate Simple Moving Average (SMA) and Exponential Moving Average (EMA).

    Args:
        df: Input DataFrame
        price_column: Column to calculate averages on
        windows: List of window sizes

    Returns:
        DataFrame with SMA and EMA columns added
    """
    df = df.copy()

    for window in windows:
        df[f'SMA_{window}'] = df[price_column].rolling(window=window).mean()
        df[f'EMA_{window}'] = df[price_column].ewm(span=window, adjust=False).mean()

    return df


def calculate_bollinger_bands(df: pd.DataFrame, price_column: str = 'Close',
                              window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Calculate Bollinger Bands.

    Args:
        df: Input DataFrame
        price_column: Column to calculate bands on
        window: Rolling window size
        num_std: Number of standard deviations

    Returns:
        DataFrame with Bollinger Bands columns added
    """
    df = df.copy()

    rolling_mean = df[price_column].rolling(window=window).mean()
    rolling_std = df[price_column].rolling(window=window).std()

    df['BB_Middle'] = rolling_mean
    df['BB_Upper'] = rolling_mean + (rolling_std * num_std)
    df['BB_Lower'] = rolling_mean - (rolling_std * num_std)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']

    return df


def calculate_atr(df: pd.DataFrame, high_col: str = 'High', low_col: str = 'Low',
                 close_col: str = 'Close', window: int = 14) -> pd.DataFrame:
    """
    Calculate Average True Range (ATR) volatility indicator.

    Args:
        df: Input DataFrame
        high_col: High price column
        low_col: Low price column
        close_col: Close price column
        window: Rolling window size

    Returns:
        DataFrame with ATR column added
    """
    df = df.copy()

    high_low = df[high_col] - df[low_col]
    high_close = np.abs(df[high_col] - df[close_col].shift())
    low_close = np.abs(df[low_col] - df[close_col].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=window).mean()

    return df


def calculate_rsi(df: pd.DataFrame, price_column: str = 'Close',
                 window: int = 14) -> pd.DataFrame:
    """
    Calculate Relative Strength Index (RSI) momentum indicator.

    Args:
        df: Input DataFrame
        price_column: Column to calculate RSI on
        window: Rolling window size

    Returns:
        DataFrame with RSI column added
    """
    df = df.copy()

    delta = df[price_column].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    return df


def calculate_macd(df: pd.DataFrame, price_column: str = 'Close',
                  fast_period: int = 12, slow_period: int = 26,
                  signal_period: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence) momentum indicator.

    Args:
        df: Input DataFrame
        price_column: Column to calculate MACD on
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period

    Returns:
        DataFrame with MACD columns added
    """
    df = df.copy()

    ema_fast = df[price_column].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df[price_column].ewm(span=slow_period, adjust=False).mean()

    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    return df


def calculate_volume_indicators(df: pd.DataFrame, volume_column: str = 'Volume',
                                price_column: str = 'Close') -> pd.DataFrame:
    """
    Calculate volume-based indicators.

    Args:
        df: Input DataFrame
        volume_column: Volume column name
        price_column: Price column name

    Returns:
        DataFrame with volume indicator columns added
    """
    df = df.copy()

    df['Volume_SMA_20'] = df[volume_column].rolling(window=20).mean()
    df['Volume_Ratio'] = df[volume_column] / df['Volume_SMA_20']

    df['OBV'] = (np.sign(df[price_column].diff()) * df[volume_column]).fillna(0).cumsum()

    return df


def create_lagged_features(df: pd.DataFrame, columns: List[str],
                          lags: List[int]) -> pd.DataFrame:
    """
    Create lagged features for specified columns.

    Args:
        df: Input DataFrame
        columns: Columns to create lags for
        lags: List of lag periods

    Returns:
        DataFrame with lagged feature columns added
    """
    df = df.copy()

    for col in columns:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    return df


def create_rolling_windows(data: np.ndarray, window_size: int,
                          step_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create rolling windows for time series sequences (X, y pairs).

    Args:
        data: Input array (n_samples, n_features)
        window_size: Size of input window
        step_size: Step size for rolling window

    Returns:
        Tuple of (X, y) where X has shape (n_sequences, window_size, n_features)
        and y has shape (n_sequences, n_features)
    """
    X, y = [], []

    for i in range(0, len(data) - window_size, step_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])

    return np.array(X), np.array(y)


def calculate_returns(df: pd.DataFrame, price_column: str = 'Close') -> pd.DataFrame:
    """
    Calculate percentage returns (percentage change from previous close).

    Args:
        df: Input DataFrame
        price_column: Column to calculate returns on

    Returns:
        DataFrame with Returns column added
    """
    df = df.copy()
    df['Returns'] = df[price_column].pct_change()
    return df


def apply_all_features(df: pd.DataFrame,
                      price_cols: Dict[str, str] = None) -> pd.DataFrame:
    """
    Apply all feature engineering functions to the DataFrame.

    Args:
        df: Input DataFrame
        price_cols: Dictionary mapping column types to names
                   {'close': 'Close', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}

    Returns:
        DataFrame with all features added
    """
    if price_cols is None:
        price_cols = {
            'close': 'Close',
            'high': 'High',
            'low': 'Low',
            'volume': 'Volume'
        }

    df = df.copy()

    df = calculate_moving_averages(df, price_column=price_cols['close'])
    df = calculate_bollinger_bands(df, price_column=price_cols['close'])
    df = calculate_atr(df, high_col=price_cols['high'],
                      low_col=price_cols['low'],
                      close_col=price_cols['close'])
    df = calculate_rsi(df, price_column=price_cols['close'])
    df = calculate_macd(df, price_column=price_cols['close'])
    df = calculate_volume_indicators(df, volume_column=price_cols['volume'],
                                     price_column=price_cols['close'])

    return df


def load_and_stack_mag7_stocks(
    stock_paths: Dict[str, str],
    date_column: str = 'Date',
    apply_features: bool = True
) -> pd.DataFrame:
    """
    Load all MAG 7 stocks and stack vertically (concatenate rows).

    This creates a single dataset with all stocks combined, where each row
    maintains a 'stock' label for later per-stock evaluation.

    Args:
        stock_paths: Dictionary mapping ticker symbols to file paths
                    e.g., {'GOOG': 'data/Stocks/goog.us.txt', ...}
        date_column: Name of the date column
        apply_features: Whether to apply feature engineering

    Returns:
        DataFrame with all stocks stacked, sorted by date
    """
    dfs = []

    for ticker, path in stock_paths.items():
        print(f"Loading {ticker} from {path}...")
        df = load_stock_data(path, date_column=date_column)
        df = parse_and_sort_dates(df, date_column=date_column)
        df = handle_missing_values(df, method='forward_fill')
        if apply_features:
            df = apply_all_features(df)
        df['stock'] = ticker
        df = df.dropna()

        dfs.append(df)

        print(f"  Loaded {len(df)} rows for {ticker}")
    stacked = pd.concat(dfs, axis=0, ignore_index=False)

    stacked = stacked.sort_values(date_column).reset_index(drop=True)

    print(f"\nTotal stacked dataset: {len(stacked)} rows across {len(stock_paths)} stocks")

    return stacked


def prepare_mag7_features(
    stacked_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare feature matrix for MAG 7 stacked dataset.

    Args:
        stacked_df: Stacked DataFrame with all stocks
        feature_cols: List of feature column names (if None, auto-detect)

    Returns:
        Tuple of (feature_df, feature_column_names)
    """
    if feature_cols is None:
        feature_cols = [
            'Close', 'Open', 'High', 'Low', 'Volume',
            'SMA_20', 'EMA_20', 'RSI', 'MACD', 'MACD_Signal',
            'BB_Width', 'ATR', 'Volume_Ratio'
        ]

    missing_cols = [col for col in feature_cols if col not in stacked_df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

    feature_df = stacked_df[feature_cols].copy()

    return feature_df, feature_cols
