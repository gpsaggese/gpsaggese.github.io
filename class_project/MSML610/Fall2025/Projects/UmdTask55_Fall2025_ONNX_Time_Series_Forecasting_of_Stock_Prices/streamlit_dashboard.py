import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
import os
import time

from onnx_forecasting_utils import (
    ONNXInferenceSession,
    flatten_sequences_for_xgboost,
    create_ensemble_predictions,
    apply_all_features
)

MAG7_TICKERS = {
    'GOOG': 'Alphabet (Google)',
    'AAPL': 'Apple',
    'AMZN': 'Amazon',
    'META': 'Meta (Facebook)',
    'NVDA': 'Nvidia',
    'TSLA': 'Tesla',
    'MSFT': 'Microsoft'
}

SEQUENCE_LENGTH = 15

@st.cache_data(ttl=3600)
def fetch_mag7_data(days=30):
    """Fetch last N days of OHLC data for MAG 7 stocks using yfinance."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 100)

    data_dict = {}
    for ticker in MAG7_TICKERS.keys():
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            data_dict[ticker] = df
        except Exception as e:
            st.error(f"Error fetching {ticker}: {e}")
            data_dict[ticker] = None

    return data_dict

def create_candlestick_chart(df, ticker, days=30):
    """Create plotly candlestick chart for stock OHLC data."""
    display_df = df.tail(days)

    fig = go.Figure(data=[go.Candlestick(
        x=display_df.index,
        open=display_df['Open'],
        high=display_df['High'],
        low=display_df['Low'],
        close=display_df['Close'],
        name=ticker
    )])

    fig.update_layout(
        title=f"{ticker} - {MAG7_TICKERS[ticker]}",
        yaxis_title="Price ($)",
        xaxis_title="Date",
        height=400,
        xaxis_rangeslider_visible=False
    )

    return fig

@st.cache_resource
def load_models():
    """Load ONNX models and scaler."""
    try:
        lstm_session = ONNXInferenceSession('models/lstm_mag7.onnx')
        xgb_session = ONNXInferenceSession('models/xgboost_mag7.onnx')

        tcn_session = None
        if os.path.exists('models/tcn_mag7.onnx'):
            try:
                tcn_session = ONNXInferenceSession('models/tcn_mag7.onnx')
            except:
                pass

        with open('models/mag7_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        return lstm_session, tcn_session, xgb_session, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

def prepare_input_data(df, scaler):
    """Prepare stock data for model inference."""
    temp_df = df.copy()
    temp_df = temp_df.reset_index()

    if 'index' in temp_df.columns:
        temp_df = temp_df.rename(columns={'index': 'Date'})

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in temp_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available: {temp_df.columns.tolist()}")

    temp_df = temp_df[['Date'] + required_cols].copy()

    temp_df = apply_all_features(temp_df)

    feature_cols = [
        'Close', 'Open', 'High', 'Low', 'Volume',
        'SMA_20', 'EMA_20', 'RSI', 'MACD', 'MACD_Signal',
        'BB_Width', 'ATR', 'Volume_Ratio'
    ]

    temp_df = temp_df.dropna()

    if len(temp_df) < SEQUENCE_LENGTH:
        raise ValueError(f"Not enough data after feature calculation. Need {SEQUENCE_LENGTH}, got {len(temp_df)}")

    features = temp_df[feature_cols].values[-SEQUENCE_LENGTH:]
    features_normalized = scaler.transform(features)

    X = features_normalized.reshape(1, SEQUENCE_LENGTH, len(feature_cols))
    X_flat = features_normalized.reshape(1, -1)

    return X, X_flat, features_normalized[-1, 0]

def predict_next_day(lstm_session, tcn_session, xgb_session, X, X_flat, scaler):
    """Generate ensemble prediction for next day's closing price."""
    predictions = []

    lstm_pred = lstm_session.predict(X.astype(np.float32))[0, 0]
    predictions.append(lstm_pred)

    if tcn_session is not None:
        try:
            tcn_pred = tcn_session.predict(X.astype(np.float32))[0, 0]
            predictions.append(tcn_pred)
        except:
            pass

    xgb_pred = xgb_session.predict(X_flat.astype(np.float32))[0, 0]
    predictions.append(xgb_pred)

    ensemble_pred = np.mean(predictions)

    return ensemble_pred, lstm_pred, xgb_pred

def run_inference_speed_comparison(lstm_session, tcn_session, xgb_session, X, X_flat):
    """Run inference speed comparison across models."""
    num_iterations = 100

    start = time.time()
    for _ in range(num_iterations):
        _ = lstm_session.predict(X.astype(np.float32))
    lstm_time = time.time() - start

    tcn_time = None
    if tcn_session is not None:
        try:
            start = time.time()
            for _ in range(num_iterations):
                _ = tcn_session.predict(X.astype(np.float32))
            tcn_time = time.time() - start
        except:
            pass

    start = time.time()
    for _ in range(num_iterations):
        _ = xgb_session.predict(X_flat.astype(np.float32))
    xgb_time = time.time() - start

    results = {
        'Model': [],
        'Time (s)': [],
        'Samples/sec': [],
        'Avg Latency (ms)': []
    }

    results['Model'].append('LSTM (ONNX)')
    results['Time (s)'].append(f"{lstm_time:.4f}")
    results['Samples/sec'].append(f"{num_iterations/lstm_time:.1f}")
    results['Avg Latency (ms)'].append(f"{(lstm_time/num_iterations)*1000:.2f}")

    if tcn_time is not None:
        results['Model'].append('TCN (ONNX)')
        results['Time (s)'].append(f"{tcn_time:.4f}")
        results['Samples/sec'].append(f"{num_iterations/tcn_time:.1f}")
        results['Avg Latency (ms)'].append(f"{(tcn_time/num_iterations)*1000:.2f}")

    results['Model'].append('XGBoost (ONNX)')
    results['Time (s)'].append(f"{xgb_time:.4f}")
    results['Samples/sec'].append(f"{num_iterations/xgb_time:.1f}")
    results['Avg Latency (ms)'].append(f"{(xgb_time/num_iterations)*1000:.2f}")

    results['Model'].append('Ensemble (Avg)')
    avg_time = np.mean([t for t in [lstm_time, tcn_time, xgb_time] if t is not None])
    results['Time (s)'].append(f"{avg_time:.4f}")
    results['Samples/sec'].append(f"{num_iterations/avg_time:.1f}")
    results['Avg Latency (ms)'].append(f"{(avg_time/num_iterations)*1000:.2f}")

    return pd.DataFrame(results)

def get_model_sizes():
    """Get model file sizes for comparison."""
    model_paths = {
        'LSTM (ONNX)': 'models/lstm_mag7.onnx',
        'TCN (ONNX)': 'models/tcn_mag7.onnx',
        'XGBoost (ONNX)': 'models/xgboost_mag7.onnx',
    }

    sizes = {
        'Model': [],
        'Size (MB)': []
    }

    for name, path in model_paths.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            sizes['Model'].append(name)
            sizes['Size (MB)'].append(f"{size_mb:.2f}")

    return pd.DataFrame(sizes)

def main():
    st.set_page_config(
        page_title="ONNX Forecasting - MAG 7",
        page_icon="📈",
        layout="wide"
    )

    st.title("📈 ONNX Forecasting for Magnificent 7")

    if 'stock_data' not in st.session_state:
        with st.spinner("Loading MAG 7 stock data..."):
            st.session_state.stock_data = fetch_mag7_data(days=30)
            st.session_state.models_loaded = False

    if not st.session_state.models_loaded:
        with st.spinner("Loading ONNX models..."):
            lstm_session, tcn_session, xgb_session, scaler = load_models()
            if lstm_session is not None:
                st.session_state.lstm_session = lstm_session
                st.session_state.tcn_session = tcn_session
                st.session_state.xgb_session = xgb_session
                st.session_state.scaler = scaler
                st.session_state.models_loaded = True

    if 'selected_stock' not in st.session_state:
        st.session_state.selected_stock = None

    if st.session_state.selected_stock is None:
        st.markdown("### Last 30 Days - Candlestick Charts")

        cols = st.columns(2)

        for idx, (ticker, name) in enumerate(MAG7_TICKERS.items()):
            df = st.session_state.stock_data.get(ticker)

            if df is not None and not df.empty:
                col = cols[idx % 2]

                with col:
                    fig = create_candlestick_chart(df, ticker)
                    st.plotly_chart(fig, use_container_width=True)

                    if st.button(f"➡️ Forecast", key=f"forecast_{ticker}"):
                        st.session_state.selected_stock = ticker
                        st.rerun()
            else:
                st.error(f"No data available for {ticker}")

    else:
        ticker = st.session_state.selected_stock
        df = st.session_state.stock_data.get(ticker)

        if st.button("⬅️ Back to All Stocks"):
            st.session_state.selected_stock = None
            st.rerun()

        st.markdown(f"## {ticker} - {MAG7_TICKERS[ticker]}")

        fig = create_candlestick_chart(df, ticker)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"### Tomorrow's Forecast - {(datetime.now() + timedelta(days=1)).strftime('%A, %Y-%m-%d')}")

        if st.session_state.models_loaded:
            try:
                X, X_flat, current_close = prepare_input_data(
                    df,
                    st.session_state.scaler
                )

                ensemble_pred, lstm_pred, xgb_pred = predict_next_day(
                    st.session_state.lstm_session,
                    st.session_state.tcn_session,
                    st.session_state.xgb_session,
                    X, X_flat,
                    st.session_state.scaler
                )

                dummy_features = np.zeros((1, 13))
                dummy_features[0, 0] = ensemble_pred
                ensemble_pred_actual = st.session_state.scaler.inverse_transform(dummy_features)[0, 0]

                dummy_features[0, 0] = current_close
                current_close_actual = st.session_state.scaler.inverse_transform(dummy_features)[0, 0]

                percent_change = ((ensemble_pred_actual - current_close_actual) / current_close_actual) * 100

                direction = "UP" if percent_change > 0 else "DOWN"
                print(f"[PREDICTION] {ticker}: {percent_change:+.2f}% {direction} | Current: ${current_close_actual:.2f} -> Predicted: ${ensemble_pred_actual:.2f}")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Current Close",
                        f"${current_close_actual:.2f}"
                    )

                with col2:
                    st.metric(
                        "Predicted Close (Next Day)",
                        f"${ensemble_pred_actual:.2f}",
                        f"{percent_change:+.2f}%"
                    )

                with col3:
                    direction = "📈 UP" if percent_change > 0 else "📉 DOWN"
                    st.metric(
                        "Prediction",
                        direction
                    )

                st.markdown("### ⚡ Inference Speed Comparison")

                with st.spinner("Running inference speed benchmark (100 iterations)..."):
                    speed_df = run_inference_speed_comparison(
                        st.session_state.lstm_session,
                        st.session_state.tcn_session,
                        st.session_state.xgb_session,
                        X, X_flat
                    )

                st.dataframe(speed_df, use_container_width=True, hide_index=True)

                st.markdown("### 💾 Model Size Comparison")
                size_df = get_model_sizes()
                st.dataframe(size_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"Error generating forecast: {e}")
                st.exception(e)
        else:
            st.error("Models could not be loaded. Please ensure all ONNX models are available in the models/ directory.")

if __name__ == "__main__":
    main()
