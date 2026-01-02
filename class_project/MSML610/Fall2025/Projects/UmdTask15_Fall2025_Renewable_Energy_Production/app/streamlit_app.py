#!/usr/bin/env python3
"""
Streamlit app for the Renewable Energy Forecasting project.

Features:
- Loads the best RandomForest model from MLflow.
- Interactive predictor: user provides weather + time + lag features -> predicts energy.
- Simple dashboard: last 7 days of actual energy + RF feature importances.
"""

from pathlib import Path
import pandas as pd
import streamlit as st
import mlflow
import mlflow.sklearn

# ---------------------------------------------------------------------
# Project + MLflow setup
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_PATH = DATA_DIR / "processed" / "train.csv"

EXPERIMENT_NAME = "solar_energy_forecasting"


def init_mlflow():
    """Set MLflow tracking URI + get experiment."""
    tracking_uri = f"file:{PROJECT_ROOT / 'mlruns'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    return exp


def load_best_rf_model(exp):
    """
    Load the best RandomForest run from MLflow based on R^2.

    We filter by param 'model_type' == 'RandomForestRegressor'
    (as logged in train.py) and pick the highest r2.
    """
    if exp is None:
        st.error("MLflow experiment not found. Have you run scripts/train.py?")
        st.stop()

    runs_df = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="params.model_type = 'RandomForestRegressor'",
        order_by=["metrics.r2 DESC"],
    )

    if runs_df.empty:
        st.error("No RandomForest runs found in MLflow. Run train.py first.")
        st.stop()

    best_run_id = runs_df.iloc[0]["run_id"]
    st.sidebar.success(f"Loaded best RF run_id: {best_run_id}")

    model_uri = f"runs:/{best_run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    return model


def load_processed_data():
    """Load processed features for dashboard visuals."""
    if not PROCESSED_PATH.exists():
        st.warning("Processed data not found. Run scripts/make_features.py first.")
        return None

    df = pd.read_csv(PROCESSED_PATH, parse_dates=[0], index_col=0)

    return df


# ---------------------------------------------------------------------
# Streamlit UI helpers
# ---------------------------------------------------------------------
def sidebar_inputs():
    """
    Sidebar controls for interactive prediction.

    Expected features (must match RF training):
        temp_c, cloud_cover, solar_radiation, wind_speed,
        hour, dayofweek, month,
        energy_mwh_lag_1, energy_mwh_lag_2, energy_mwh_lag_24,
        energy_mwh_rollmean_3, energy_mwh_rollmean_24
    """
    st.sidebar.header("Input Features")

    st.sidebar.subheader("Weather variables")
    temp_c = st.sidebar.number_input("Temperature (°C)", -20.0, 50.0, 25.0, 0.5)
    cloud_cover = st.sidebar.slider("Cloud cover (%)", 0, 100, 30, 1)
    solar_radiation = st.sidebar.number_input("Solar radiation (W/m²)", 0.0, 1500.0, 600.0, 10.0)
    wind_speed = st.sidebar.number_input("Wind speed (m/s)", 0.0, 40.0, 3.0, 0.5)

    st.sidebar.subheader("Time features")
    hour = st.sidebar.slider("Hour of day", 0, 23, 12, 1)
    dayofweek = st.sidebar.selectbox("Day of week (0=Mon)", options=list(range(7)), index=2)
    month = st.sidebar.slider("Month", 1, 12, 6, 1)

    st.sidebar.subheader("Lag / rolling features")
    energy_mwh_lag_1 = st.sidebar.number_input("Energy lag 1h (MWh)", 0.0, 200.0, 50.0, 1.0)
    energy_mwh_lag_2 = st.sidebar.number_input("Energy lag 2h (MWh)", 0.0, 200.0, 48.0, 1.0)
    energy_mwh_lag_24 = st.sidebar.number_input("Energy lag 24h (MWh)", 0.0, 200.0, 60.0, 1.0)
    energy_mwh_rollmean_3 = st.sidebar.number_input("Energy rolling mean 3h (MWh)", 0.0, 200.0, 49.0, 1.0)
    energy_mwh_rollmean_24 = st.sidebar.number_input("Energy rolling mean 24h (MWh)", 0.0, 200.0, 55.0, 1.0)

    feature_row = {
        "temp_c": temp_c,
        "cloud_cover": cloud_cover,
        "solar_radiation": solar_radiation,
        "wind_speed": wind_speed,
        "hour": hour,
        "dayofweek": dayofweek,
        "month": month,
        "energy_mwh_lag_1": energy_mwh_lag_1,
        "energy_mwh_lag_2": energy_mwh_lag_2,
        "energy_mwh_lag_24": energy_mwh_lag_24,
        "energy_mwh_rollmean_3": energy_mwh_rollmean_3,
        "energy_mwh_rollmean_24": energy_mwh_rollmean_24,
    }

    return pd.DataFrame([feature_row])


def main():
    st.set_page_config(page_title="Solar Energy Forecasting", layout="wide")

    st.title( "Solar Energy Production Forecasting")
    st.markdown(
        """
        This app uses a **RandomForest time-series model** trained on hourly solar data
        (weather + time + lag features) to forecast energy production.

        - Use the left sidebar to set input features.
        - See the model prediction and simple dashboard on the right.
        """
    )

    # Initialize MLflow + load model
    exp = init_mlflow()
    model = load_best_rf_model(exp)

    # Sidebar inputs
    input_df = sidebar_inputs()

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("Model Prediction")
        st.write("Input features used for prediction:")
        st.dataframe(input_df)

        if st.button("Predict energy (MWh)"):
            try:
                y_pred = model.predict(input_df)[0]
                st.success(f"Predicted energy production: **{y_pred:.2f} MWh**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with col2:
        st.subheader("Simple Dashboard")

        df = load_processed_data()
        if df is not None:
            if "energy_mwh" in df.columns:
                df_last = df.tail(24 * 7)  # last 7 days (assuming hourly)
                st.markdown("**Last 7 days: actual energy production**")
                st.line_chart(df_last["energy_mwh"])
            else:
                st.info("Column 'energy_mwh' not found in processed data.")

            if hasattr(model, "feature_importances_"):
                fi = pd.Series(model.feature_importances_, index=input_df.columns).sort_values(ascending=False)
                st.markdown("**RandomForest feature importances**")
                st.bar_chart(fi)
            else:
                st.info("Feature importances not available for this model.")
        else:
            st.info("Processed data not found, run `scripts/make_features.py` first for dashboard visuals.")


if __name__ == "__main__":
    main()
