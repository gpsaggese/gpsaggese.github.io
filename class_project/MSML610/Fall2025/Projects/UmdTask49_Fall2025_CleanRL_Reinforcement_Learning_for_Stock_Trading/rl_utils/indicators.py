from typing import Dict, Optional
import pandas as pd
import talib as ta
import numpy as np
from numba import njit
import warnings

# Suppress Numba warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning)


# ============================================================================
# NUMBA-OPTIMIZED CALCULATION FUNCTIONS
# ============================================================================

@njit(cache=True, fastmath=True)
def _rolling_mean_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling mean using Numba."""
    n = len(arr)
    result = np.empty(n)
    result[:window-1] = np.nan
    
    for i in range(window-1, n):
        result[i] = np.mean(arr[i-window+1:i+1])
    return result


@njit(cache=True, fastmath=True)
def _rolling_max_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling max using Numba."""
    n = len(arr)
    result = np.empty(n)
    result[:window-1] = np.nan
    
    for i in range(window-1, n):
        result[i] = np.max(arr[i-window+1:i+1])
    return result


@njit(cache=True, fastmath=True)
def _rolling_min_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling min using Numba."""
    n = len(arr)
    result = np.empty(n)
    result[:window-1] = np.nan
    
    for i in range(window-1, n):
        result[i] = np.min(arr[i-window+1:i+1])
    return result


@njit(cache=True, fastmath=True)
def _ema_numba(arr: np.ndarray, span: int) -> np.ndarray:
    """Fast exponential moving average using Numba."""
    n = len(arr)
    result = np.empty(n)
    alpha = 2.0 / (span + 1.0)
    
    # Initialize with first valid value
    result[0] = arr[0]
    
    for i in range(1, n):
        if np.isnan(arr[i]):
            result[i] = result[i-1]
        else:
            result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
    
    return result


@njit(cache=True, fastmath=True)
def _rsi_numba(close: np.ndarray, period: int) -> np.ndarray:
    """
    Fast RSI calculation using Numba with Wilder's smoothing.
    This matches the standard RSI calculation used by most platforms.
    """
    n = len(close)
    result = np.empty(n)
    result[:period] = np.nan
    
    # Calculate price changes
    deltas = np.diff(close)
    
    # Initialize gains and losses for first RSI value
    initial_gains = 0.0
    initial_losses = 0.0
    
    for j in range(period):
        if deltas[j] > 0:
            initial_gains += deltas[j]
        else:
            initial_losses -= deltas[j]
    
    # First average gain/loss (SMA)
    avg_gain = initial_gains / period
    avg_loss = initial_losses / period
    
    # First RSI value
    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))
    
    # Subsequent RSI values using Wilder's smoothing
    # Smoothed average = (previous_avg * (period - 1) + current_value) / period
    for i in range(period + 1, n):
        gain = deltas[i-1] if deltas[i-1] > 0 else 0.0
        loss = -deltas[i-1] if deltas[i-1] < 0 else 0.0
        
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        
        if avg_loss == 0:
            result[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return result


@njit(cache=True, fastmath=True)
def _atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Fast ATR calculation using Numba."""
    n = len(close)
    tr = np.empty(n)
    atr = np.empty(n)
    
    # First TR
    tr[0] = high[0] - low[0]
    
    # Calculate True Range
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    # Calculate ATR using smoothed average
    atr[:period] = np.nan
    atr[period] = np.mean(tr[:period+1])
    
    for i in range(period+1, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr


@njit(cache=True, fastmath=True)
def _percentile_rank_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling percentile rank (0-1) of current value in window."""
    n = len(arr)
    result = np.empty(n)
    result[:window-1] = np.nan
    
    for i in range(window-1, n):
        window_data = arr[i-window+1:i+1]
        current_val = arr[i]
        
        # Count values less than or equal to current
        count = 0
        for val in window_data:
            if val <= current_val:
                count += 1
        
        result[i] = count / window
    
    return result


@njit(cache=True, fastmath=True)
def _linear_regression_slope_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling linear regression slope."""
    n = len(arr)
    result = np.empty(n)
    result[:window-1] = np.nan
    
    for i in range(window-1, n):
        y = arr[i-window+1:i+1]
        x = np.arange(window, dtype=np.float64)
        
        # Calculate slope using least squares
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = 0.0
        denominator = 0.0
        
        for j in range(window):
            numerator += (x[j] - x_mean) * (y[j] - y_mean)
            denominator += (x[j] - x_mean) ** 2
        
        if denominator != 0:
            result[i] = numerator / denominator
        else:
            result[i] = 0.0
    
    return result


@njit(cache=True, fastmath=True)
def _kurtosis_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling kurtosis (measure of tail heaviness)."""
    n = len(arr)
    result = np.empty(n)
    result[:window-1] = np.nan
    
    for i in range(window-1, n):
        window_data = arr[i-window+1:i+1]
        mean = np.mean(window_data)
        std = np.std(window_data)
        
        if std > 0:
            normalized = (window_data - mean) / std
            result[i] = np.mean(normalized ** 4) - 3.0  # Excess kurtosis
        else:
            result[i] = 0.0
    
    return result


@njit(cache=True, fastmath=True)
def _skewness_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """Calculate rolling skewness (measure of asymmetry)."""
    n = len(arr)
    result = np.empty(n)
    result[:window-1] = np.nan
    
    for i in range(window-1, n):
        window_data = arr[i-window+1:i+1]
        mean = np.mean(window_data)
        std = np.std(window_data)
        
        if std > 0:
            normalized = (window_data - mean) / std
            result[i] = np.mean(normalized ** 3)
        else:
            result[i] = 0.0
    
    return result


@njit(cache=True, fastmath=True)
def _autocorrelation_numba(arr: np.ndarray, window: int, lag: int) -> np.ndarray:
    """Calculate rolling autocorrelation at given lag."""
    n = len(arr)
    result = np.empty(n)
    result[:window-1] = np.nan
    
    for i in range(window-1, n):
        if i >= window + lag - 1:
            window_data = arr[i-window+1:i+1]
            lagged_data = arr[i-window+1-lag:i+1-lag]
            
            mean_current = np.mean(window_data)
            mean_lagged = np.mean(lagged_data)
            
            numerator = 0.0
            denom_current = 0.0
            denom_lagged = 0.0
            
            for j in range(window):
                dev_current = window_data[j] - mean_current
                dev_lagged = lagged_data[j] - mean_lagged
                numerator += dev_current * dev_lagged
                denom_current += dev_current ** 2
                denom_lagged += dev_lagged ** 2
            
            denominator = np.sqrt(denom_current * denom_lagged)
            
            if denominator > 0:
                result[i] = numerator / denominator
            else:
                result[i] = 0.0
        else:
            result[i] = np.nan
    
    return result


@njit(cache=True, fastmath=True)
def _hurst_exponent_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate rolling Hurst exponent (measure of long-term memory).
    H < 0.5: mean-reverting, H = 0.5: random walk, H > 0.5: trending
    """
    n = len(arr)
    result = np.empty(n)
    result[:window-1] = np.nan
    
    lags = np.array([2, 4, 8, 16], dtype=np.int64)
    
    for i in range(window-1, n):
        window_data = arr[i-window+1:i+1]
        
        tau = np.empty(len(lags))
        
        for lag_idx in range(len(lags)):
            lag = lags[lag_idx]
            
            if lag < window // 2:
                # Calculate rescaled range
                ts = window_data
                mean = np.mean(ts)
                
                # Mean-adjusted series
                y = ts - mean
                
                # Cumulative deviate series
                z = np.cumsum(y)
                
                # Range
                r = np.max(z) - np.min(z)
                
                # Standard deviation
                s = np.std(ts)
                
                if s > 0:
                    tau[lag_idx] = r / s
                else:
                    tau[lag_idx] = 0.0
            else:
                tau[lag_idx] = np.nan
        
        # Calculate Hurst exponent from log-log slope
        valid_mask = ~np.isnan(tau)
        if np.sum(valid_mask) >= 2:
            log_lags = np.log(lags[valid_mask].astype(np.float64))
            log_tau = np.log(tau[valid_mask])
            
            # Linear regression
            x_mean = np.mean(log_lags)
            y_mean = np.mean(log_tau)
            
            numerator = 0.0
            denominator = 0.0
            
            for j in range(len(log_lags)):
                numerator += (log_lags[j] - x_mean) * (log_tau[j] - y_mean)
                denominator += (log_lags[j] - x_mean) ** 2
            
            if denominator > 0:
                result[i] = numerator / denominator
            else:
                result[i] = 0.5
        else:
            result[i] = 0.5
    
    return result


@njit(cache=True, fastmath=True)
def _fractal_dimension_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate rolling fractal dimension using box-counting method.
    FD ~ 1: smooth trend, FD ~ 2: very rough/random
    """
    n = len(arr)
    result = np.empty(n)
    result[:window-1] = np.nan
    
    for i in range(window-1, n):
        window_data = arr[i-window+1:i+1]
        
        # Normalize to [0, 1]
        min_val = np.min(window_data)
        max_val = np.max(window_data)
        
        if max_val > min_val:
            normalized = (window_data - min_val) / (max_val - min_val)
            
            # Simple fractal dimension estimate
            # Count direction changes as proxy for complexity
            changes = 0
            for j in range(1, len(normalized)):
                if (normalized[j] - normalized[j-1]) * (normalized[j-1] - normalized[j-2] if j > 1 else 0) < 0:
                    changes += 1
            
            # Map changes to fractal dimension estimate (1-2)
            result[i] = 1.0 + (changes / window)
        else:
            result[i] = 1.0
    
    return result


@njit(cache=True, fastmath=True)
def _efficiency_ratio_numba(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate Kaufman's Efficiency Ratio.
    ER = Net change / Sum of absolute changes
    ER close to 1: trending, ER close to 0: choppy/ranging
    """
    n = len(arr)
    result = np.empty(n)
    result[:window] = np.nan
    
    for i in range(window, n):
        net_change = abs(arr[i] - arr[i-window])
        
        sum_changes = 0.0
        for j in range(i-window+1, i+1):
            sum_changes += abs(arr[j] - arr[j-1])
        
        if sum_changes > 0:
            result[i] = net_change / sum_changes
        else:
            result[i] = 0.0
    
    return result


@njit(cache=True, fastmath=True)
def _choppiness_index_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate Choppiness Index.
    CI > 61.8: choppy/sideways, CI < 38.2: trending
    """
    n = len(close)
    result = np.empty(n)
    result[:window-1] = np.nan
    
    for i in range(window-1, n):
        high_window = high[i-window+1:i+1]
        low_window = low[i-window+1:i+1]
        close_window = close[i-window+1:i+1]
        
        # Calculate ATR sum
        atr_sum = 0.0
        for j in range(1, window):
            hl = high_window[j] - low_window[j]
            hc = abs(high_window[j] - close_window[j-1])
            lc = abs(low_window[j] - close_window[j-1])
            atr_sum += max(hl, hc, lc)
        
        # Calculate range
        max_high = np.max(high_window)
        min_low = np.min(low_window)
        range_val = max_high - min_low
        
        if range_val > 0 and atr_sum > 0:
            result[i] = 100.0 * np.log10(atr_sum / range_val) / np.log10(window)
        else:
            result[i] = 50.0
    
    return result


@njit(cache=True, fastmath=True)
def _vwap_deviation_numba(close: np.ndarray, volume: np.ndarray, window: int) -> np.ndarray:
    """Calculate deviation from Volume-Weighted Average Price."""
    n = len(close)
    result = np.empty(n)
    result[:window-1] = np.nan
    
    for i in range(window-1, n):
        close_window = close[i-window+1:i+1]
        volume_window = volume[i-window+1:i+1]
        
        total_vol = np.sum(volume_window)
        
        if total_vol > 0:
            vwap = np.sum(close_window * volume_window) / total_vol
            result[i] = (close[i] - vwap) / vwap
        else:
            result[i] = 0.0
    
    return result


@njit(cache=True, fastmath=True)
def _money_flow_index_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                             volume: np.ndarray, period: int) -> np.ndarray:
    """Calculate Money Flow Index (volume-weighted RSI)."""
    n = len(close)
    result = np.empty(n)
    result[:period] = np.nan
    
    # Calculate typical price
    typical_price = (high + low + close) / 3.0
    
    # Calculate money flow
    money_flow = typical_price * volume
    
    for i in range(period, n):
        positive_flow = 0.0
        negative_flow = 0.0
        
        for j in range(i-period+1, i+1):
            if typical_price[j] > typical_price[j-1]:
                positive_flow += money_flow[j]
            elif typical_price[j] < typical_price[j-1]:
                negative_flow += money_flow[j]
        
        if negative_flow == 0:
            result[i] = 100.0
        else:
            money_ratio = positive_flow / negative_flow
            result[i] = 100.0 - (100.0 / (1.0 + money_ratio))
    
    return result


@njit(cache=True, fastmath=True)
def _rate_of_change_numba(arr: np.ndarray, period: int) -> np.ndarray:
    """Calculate Rate of Change (momentum)."""
    n = len(arr)
    result = np.empty(n)
    result[:period] = np.nan
    
    for i in range(period, n):
        if arr[i-period] != 0:
            result[i] = ((arr[i] - arr[i-period]) / arr[i-period]) * 100.0
        else:
            result[i] = 0.0
    
    return result


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

class IndicatorConfig:
    """
    Extensible configuration for all technical indicators.
    All parameters can be overridden by subclasses for strategy-specific tuning.
    """
    
    # ========== MOVING AVERAGES ==========
    EMA_FAST = 10          # Fast EMA for trend following
    EMA_MEDIUM = 21        # Medium EMA
    EMA_SLOW = 50          # Slow EMA
    SMA_SHORT = 15         # Short SMA for trend confirmation
    SMA_MEDIUM = 30        # Medium SMA for confidence
    SMA_LONG = 60          # Long SMA for major trend
    
    # ========== BOLLINGER BANDS ==========
    BOLL_PERIOD = 20       # Standard is 20
    BOLL_STDDEV = 2.0      # Standard is 2.0
    BB_DISTANCE_FACTOR = 0.3  # Sensitivity factor
    
    # ========== RSI (Relative Strength Index) ==========
    RSI_PERIOD = 14        # Standard RSI period
    FAST_RSI_PERIOD = 7    # Short-term RSI
    RSI_OVERBOUGHT = 70    # Overbought threshold
    RSI_OVERSOLD = 30      # Oversold threshold
    
    # ========== MACD (Moving Average Convergence Divergence) ==========
    MACD_FASTPERIOD = 12   # Standard fast period
    MACD_SLOWPERIOD = 26   # Standard slow period
    MACD_SIGNALPERIOD = 9  # Standard signal period
    MACD_THRESHOLD = 0.5   # Threshold for signal strength
    
    # ========== STOCHASTIC OSCILLATOR ==========
    STOCH_K_PERIOD = 14    # %K period
    STOCH_D_PERIOD = 3     # %D smoothing period
    STOCH_OVERBOUGHT = 80  # Overbought level
    STOCH_OVERSOLD = 20    # Oversold level
    
    # ========== DMI/ADX (Directional Movement Index) ==========
    DMI_PERIOD = 14        # Standard DMI period
    ADX_PERIOD = 14        # Standard ADX period
    ADX_STRONG_TREND = 25  # Strong trend threshold
    MAX_ADX = 30           # Maximum ADX for filtering
    
    # ========== VOLATILITY INDICATORS ==========
    ATR_PERIOD = 14        # Average True Range period
    ATR_PERCENTILE_WINDOW = 40  # Window for ATR percentile calculation
    VOL_MA_PERIOD = 20     # Volume moving average
    VOL_EXPAND_MULT = 1.05 # Volume expansion multiplier
    
    # ========== MOMENTUM INDICATORS ==========
    ROC_PERIOD = 12        # Rate of Change period
    MFI_PERIOD = 14        # Money Flow Index period
    CCI_PERIOD = 20        # Commodity Channel Index period
    MOMENTUM_PERIOD = 10   # Generic momentum period
    
    # ========== OSCILLATORS ==========
    ULTOSC_PERIOD1 = 7     # Ultimate Oscillator short-term
    ULTOSC_PERIOD2 = 14    # Ultimate Oscillator medium-term
    ULTOSC_PERIOD3 = 28    # Ultimate Oscillator long-term
    WILLR_PERIOD = 14      # Williams %R period
    
    # ========== RANGE & BREAKOUT ==========
    BREAKOUT_LOOKBACK = 15      # Lookback for range highs/lows
    PULLBACK_LOOKBACK = 8       # Lookback for pullback reference
    RANGE_TOLERANCE_PCT = 0.01  # Tolerance for breakout detection
    
    # ========== STATISTICAL INDICATORS ==========
    KURTOSIS_WINDOW = 20        # Rolling kurtosis window
    SKEWNESS_WINDOW = 20        # Rolling skewness window
    HURST_WINDOW = 50           # Hurst exponent window
    FRACTAL_WINDOW = 30         # Fractal dimension window
    EFFICIENCY_RATIO_WINDOW = 10  # Kaufman efficiency ratio
    CHOPPINESS_WINDOW = 14      # Choppiness index window
    AUTOCORR_WINDOW = 20        # Autocorrelation window
    AUTOCORR_LAG = 1            # Autocorrelation lag
    LINEAR_REG_WINDOW = 14      # Linear regression slope window
    VWAP_WINDOW = 20            # VWAP deviation window
    
    # ========== ON-BALANCE VOLUME ==========
    OBV_MA_PERIOD = 20     # OBV moving average for smoothing


# ============================================================================
# MAIN INDICATOR CALCULATION FUNCTION
# ============================================================================

def calculate_indicators(df: pd.DataFrame, 
                        config = IndicatorConfig, 
                        indicators_to_calculate: Optional[Dict[str, bool]] = None) -> pd.DataFrame:
    """
    Calculate technical indicators based on configuration using Numba-optimized functions.
    
    Args:
        df: DataFrame with OHLCV data (columns: Open, High, Low, Close, Volume)
        config: Configuration object with indicator parameters (class or instance)
        indicators_to_calculate: Optional dictionary specifying which indicators to calculate
                                {indicator_name: True/False}
    
    Returns:
        DataFrame with added indicators
    
    Available indicator groups:
        - MOVING_AVERAGES: EMA_FAST, EMA_MEDIUM, EMA_SLOW, SMA_SHORT, SMA_MEDIUM, SMA_LONG
        - BOLLINGER: BB_UPPER, BB_MID, BB_LOWER, BB_WIDTH, BB_POSITION
        - RSI: RSI, FAST_RSI
        - MACD: MACD, MACD_SIGNAL, MACD_HIST
        - STOCHASTIC: STOCH_K, STOCH_D
        - ADX: ADX
        - DMI: +DI, -DI, ATR (also calculates TR components)
        - ATR: ATR
        - VOLUME: Vol_MA, RVOL (relative volume), OBV, OBV_MA, VOL_EXPAND
        - MOMENTUM: ROC, MFI, CCI
        - OSCILLATORS: ULTOSC, WILLR
        - RANGE: RANGE_HIGH, RANGE_LOW, PULLBACK_REF_HIGH, PULLBACK_REF_LOW
        - STATISTICAL: KURTOSIS, SKEWNESS, HURST, FRACTAL_DIM, EFFICIENCY_RATIO, 
                      CHOPPINESS, AUTOCORR, LINEAR_SLOPE, VWAP_DEV
        - PERCENTILE: ATR_PCT, ATR_PCTL (ATR as percentage and percentile)
        - PATTERNS: PCT_FROM_HIGH, PCT_FROM_LOW, PCT_FROM_PB_HIGH, PCT_FROM_PB_LOW,
                   BREAKOUT_CAND, BREAKDOWN_CAND
    """
    df = df.copy()
    
    # Handle config as class or instance
    if not hasattr(config, '__dict__'):
        config = config()
    
    # Default to calculating all indicators if not specified
    if indicators_to_calculate is None:
        indicators_to_calculate = {
            'MOVING_AVERAGES': True,
            'BOLLINGER': True,
            'RSI': True,
            'MACD': True,
            'STOCHASTIC': True,
            'ADX': True,
            'DMI': True,
            'ATR': True,
            'VOLUME': True,
            'MOMENTUM': True,
            'OSCILLATORS': True,
            'RANGE': True,
            'STATISTICAL': True,
            'PERCENTILE': True,
            'PATTERNS': True,
        }
    
    # Extract numpy arrays for Numba functions
    close_arr = df['Close'].values
    high_arr = df['High'].values
    low_arr = df['Low'].values
    volume_arr = df['Volume'].values
    
    # ========== MOVING AVERAGES ==========
    if indicators_to_calculate.get('MOVING_AVERAGES', False):
        df['EMA_FAST'] = _ema_numba(close_arr, config.EMA_FAST)
        df['EMA_MEDIUM'] = _ema_numba(close_arr, config.EMA_MEDIUM)
        df['EMA_SLOW'] = _ema_numba(close_arr, config.EMA_SLOW)
        df['SMA_SHORT'] = _rolling_mean_numba(close_arr, config.SMA_SHORT)
        df['SMA_MEDIUM'] = _rolling_mean_numba(close_arr, config.SMA_MEDIUM)
        df['SMA_LONG'] = _rolling_mean_numba(close_arr, config.SMA_LONG)
    
    # ========== BOLLINGER BANDS ==========
    if indicators_to_calculate.get('BOLLINGER', False):
        # Use TALib for Bollinger Bands (optimized C implementation)
        df['BB_UPPER'], df['BB_MID'], df['BB_LOWER'] = ta.BBANDS(
            df['Close'], 
            timeperiod=config.BOLL_PERIOD, 
            nbdevup=config.BOLL_STDDEV, 
            nbdevdn=config.BOLL_STDDEV
        )
        df['BB_WIDTH'] = (df['BB_UPPER'] - df['BB_LOWER']) / df['BB_MID']
        df['BB_POSITION'] = (df['Close'] - df['BB_LOWER']) / (df['BB_UPPER'] - df['BB_LOWER'])
    
    # ========== RSI ==========
    if indicators_to_calculate.get('RSI', False):
        df['RSI'] = _rsi_numba(close_arr, config.RSI_PERIOD)
        df['FAST_RSI'] = _rsi_numba(close_arr, config.FAST_RSI_PERIOD)
    
    # ========== MACD ==========
    if indicators_to_calculate.get('MACD', False):
        # Use TALib for MACD (optimized C implementation)
        df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = ta.MACD(
            df['Close'], 
            fastperiod=config.MACD_FASTPERIOD, 
            slowperiod=config.MACD_SLOWPERIOD, 
            signalperiod=config.MACD_SIGNALPERIOD
        )
    
    # ========== STOCHASTIC ==========
    if indicators_to_calculate.get('STOCHASTIC', False):
        # Use TALib for Stochastic (optimized C implementation)
        df['STOCH_K'], df['STOCH_D'] = ta.STOCH(
            df['High'], 
            df['Low'], 
            df['Close'],
            fastk_period=config.STOCH_K_PERIOD,
            slowk_period=config.STOCH_D_PERIOD,
            slowd_period=config.STOCH_D_PERIOD
        )
    
    # ========== ADX ==========
    if indicators_to_calculate.get('ADX', False):
        # Use TALib for ADX (optimized C implementation)
        df['ADX'] = ta.ADX(df['High'], df['Low'], df['Close'], timeperiod=config.ADX_PERIOD)
    
    # ========== DMI ==========
    if indicators_to_calculate.get('DMI', False) or indicators_to_calculate.get('ATR_DMI', False):
        # Calculate True Range components
        df['H-L'] = df['High'] - df['Low']
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        
        # Calculate ATR using Numba
        df['ATR'] = _atr_numba(high_arr, low_arr, close_arr, config.ATR_PERIOD)
        
        # Calculate DMI
        dmi_period = config.DMI_PERIOD
        df['+DM'] = np.where(
            (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
            np.maximum(df['High'] - df['High'].shift(1), 0),
            0
        )
        df['-DM'] = np.where(
            (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
            np.maximum(df['Low'].shift(1) - df['Low'], 0),
            0
        )
        df['TR_DMI'] = df['TR'].rolling(window=dmi_period).sum()
        df['+DM_DMI'] = pd.Series(df['+DM']).rolling(window=dmi_period).sum()
        df['-DM_DMI'] = pd.Series(df['-DM']).rolling(window=dmi_period).sum()
        df['+DI'] = 100 * (df['+DM_DMI'] / df['TR_DMI'])
        df['-DI'] = 100 * (df['-DM_DMI'] / df['TR_DMI'])
    
    # ========== ATR (if not already calculated) ==========
    if indicators_to_calculate.get('ATR', False) and 'ATR' not in df.columns:
        df['ATR'] = _atr_numba(high_arr, low_arr, close_arr, config.ATR_PERIOD)
    
    # ========== VOLUME ==========
    if indicators_to_calculate.get('VOLUME', False):
        df['Vol_MA'] = _rolling_mean_numba(volume_arr, config.VOL_MA_PERIOD)
        df['RVOL'] = df['Volume'] / df['Vol_MA']
        df['VOL_EXPAND'] = (df['Volume'] > config.VOL_EXPAND_MULT * df['Vol_MA']).astype(int)
        
        # On-Balance Volume
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV_MA'] = _rolling_mean_numba(df['OBV'].values, config.OBV_MA_PERIOD)
    
    # ========== MOMENTUM ==========
    if indicators_to_calculate.get('MOMENTUM', False):
        df['ROC'] = _rate_of_change_numba(close_arr, config.ROC_PERIOD)
        df['MFI'] = _money_flow_index_numba(high_arr, low_arr, close_arr, volume_arr, config.MFI_PERIOD)
        # Use TALib for CCI (optimized C implementation)
        df['CCI'] = ta.CCI(df['High'], df['Low'], df['Close'], timeperiod=config.CCI_PERIOD)
    
    # ========== OSCILLATORS ==========
    if indicators_to_calculate.get('OSCILLATORS', False):
        # Use TALib for oscillators (optimized C implementation)
        df['ULTOSC'] = ta.ULTOSC(
            df['High'], 
            df['Low'], 
            df['Close'], 
            timeperiod1=config.ULTOSC_PERIOD1,
            timeperiod2=config.ULTOSC_PERIOD2,
            timeperiod3=config.ULTOSC_PERIOD3
        )
        df['WILLR'] = ta.WILLR(
            df['High'], 
            df['Low'], 
            df['Close'], 
            timeperiod=config.WILLR_PERIOD
        )
    
    # ========== RANGE & BREAKOUT ==========
    if indicators_to_calculate.get('RANGE', False):
        df['RANGE_HIGH'] = _rolling_max_numba(close_arr, config.BREAKOUT_LOOKBACK)
        df['RANGE_LOW'] = _rolling_min_numba(close_arr, config.BREAKOUT_LOOKBACK)
        df['PULLBACK_REF_HIGH'] = _rolling_max_numba(close_arr, config.PULLBACK_LOOKBACK)
        df['PULLBACK_REF_LOW'] = _rolling_min_numba(close_arr, config.PULLBACK_LOOKBACK)
    
    # ========== PATTERNS ==========
    if indicators_to_calculate.get('PATTERNS', False):
        # Requires RANGE indicators
        if 'RANGE_HIGH' not in df.columns:
            df['RANGE_HIGH'] = _rolling_max_numba(close_arr, config.BREAKOUT_LOOKBACK)
            df['RANGE_LOW'] = _rolling_min_numba(close_arr, config.BREAKOUT_LOOKBACK)
            df['PULLBACK_REF_HIGH'] = _rolling_max_numba(close_arr, config.PULLBACK_LOOKBACK)
            df['PULLBACK_REF_LOW'] = _rolling_min_numba(close_arr, config.PULLBACK_LOOKBACK)
        
        df['PCT_FROM_HIGH'] = df['Close'] / df['RANGE_HIGH'] - 1
        df['PCT_FROM_LOW'] = df['Close'] / df['RANGE_LOW'] - 1
        df['PCT_FROM_PB_HIGH'] = df['Close'] / df['PULLBACK_REF_HIGH'] - 1
        df['PCT_FROM_PB_LOW'] = df['Close'] / df['PULLBACK_REF_LOW'] - 1
        
        # Breakout/breakdown candidates
        df['BREAKOUT_CAND'] = (df['Close'] >= (1 - config.RANGE_TOLERANCE_PCT) * df['RANGE_HIGH']).astype(int)
        df['BREAKDOWN_CAND'] = (df['Close'] <= (1 + config.RANGE_TOLERANCE_PCT) * df['RANGE_LOW']).astype(int)
    
    # ========== PERCENTILE (ATR-based) ==========
    if indicators_to_calculate.get('PERCENTILE', False):
        # Requires ATR
        if 'ATR' not in df.columns:
            df['ATR'] = _atr_numba(high_arr, low_arr, close_arr, config.ATR_PERIOD)
        
        # ATR as percentage of price
        atr_pct = (df['ATR'] / df['Close']).replace(0, np.nan)
        df['ATR_PCT'] = atr_pct
        
        # ATR percentile (rolling rank) using Numba
        df['ATR_PCTL'] = _percentile_rank_numba(atr_pct.fillna(0).values, config.ATR_PERCENTILE_WINDOW)
        
        # Fill remaining NaNs with neutral value
        df['ATR_PCTL'] = df['ATR_PCTL'].fillna(0.5)
    
    # ========== STATISTICAL INDICATORS ==========
    if indicators_to_calculate.get('STATISTICAL', False):
        # All statistical indicators use Numba-optimized functions
        df['KURTOSIS'] = _kurtosis_numba(close_arr, config.KURTOSIS_WINDOW)
        df['SKEWNESS'] = _skewness_numba(close_arr, config.SKEWNESS_WINDOW)
        df['HURST'] = _hurst_exponent_numba(close_arr, config.HURST_WINDOW)
        df['FRACTAL_DIM'] = _fractal_dimension_numba(close_arr, config.FRACTAL_WINDOW)
        df['EFFICIENCY_RATIO'] = _efficiency_ratio_numba(close_arr, config.EFFICIENCY_RATIO_WINDOW)
        df['CHOPPINESS'] = _choppiness_index_numba(high_arr, low_arr, close_arr, config.CHOPPINESS_WINDOW)
        df['AUTOCORR'] = _autocorrelation_numba(close_arr, config.AUTOCORR_WINDOW, config.AUTOCORR_LAG)
        df['LINEAR_SLOPE'] = _linear_regression_slope_numba(close_arr, config.LINEAR_REG_WINDOW)
        df['VWAP_DEV'] = _vwap_deviation_numba(close_arr, volume_arr, config.VWAP_WINDOW)
    
    return df.dropna()