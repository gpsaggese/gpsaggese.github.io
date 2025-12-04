# app/streamlit_app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import joblib

# -----------------------------
# Paths
# -----------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "hourly_congestion.parquet"
MODEL_PATH = ROOT / "data" / "models" / "xgb_congestion_model.joblib"


# -----------------------------
# Cached Functions
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    return df


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    return model


# -----------------------------
# Load Data and Model
# -----------------------------
df = load_data()
model = load_model()


# -----------------------------
# UI Layout
# -----------------------------
st.set_page_config(page_title="Airport Congestion Dashboard", layout="wide")

st.title("✈️ Airport Hourly Congestion Dashboard")
st.markdown("""
This interactive dashboard allows users to explore **hourly airport congestion**  
and view **model predictions** powered by an XGBoost classifier.
""")

# -----------------------------
# Sidebar Controls
# -----------------------------
airports = sorted(df["AIRPORT"].unique())
selected_airport = st.sidebar.selectbox("Select Airport", airports)

airport_df = df[df["AIRPORT"] == selected_airport].copy()

dates = sorted(airport_df["date"].unique())
selected_date = st.sidebar.selectbox("Select Date", dates)

day_df = airport_df[airport_df["date"] == selected_date].copy()

if day_df.empty:
    st.warning("No data available for this airport and date.")
    st.stop()


# -----------------------------
# Predict Congestion
# -----------------------------
feature_cols = [
    "total_flights",
    "dep_delay_mean",
    "arr_delay_mean",
    "taxi_out_mean",
    "taxi_in_mean",
    "day_of_week",
    "month",
    "HOUR",
    "AIRPORT"
]

X_day = day_df[feature_cols]
preds = model.predict(X_day)
day_df["predicted_congestion"] = preds


# -----------------------------
# Peak Hour
# -----------------------------
cong_map = {"Low": 1, "Medium": 2, "High": 3}
peak_row = day_df.loc[day_df["predicted_congestion"].map(cong_map).idxmax()]

st.subheader("🔥 Predicted Peak Congestion")
st.metric(
    "Peak Hour",
    f"{int(peak_row['hour']):02d}:00",
    help=f"Predicted level: {peak_row['predicted_congestion']}"
)


# -----------------------------
# Bar Chart – Hourly Congestion
# -----------------------------
st.subheader(f"Hourly Congestion for {selected_airport} on {selected_date}")

fig_bar = px.bar(
    day_df,
    x="hour",
    y="total_flights",
    color="predicted_congestion",
    category_orders={"predicted_congestion": ["Low", "Medium", "High"]},
    labels={"hour": "Hour of Day", "total_flights": "Total Flights"},
    title="Predicted Congestion by Hour"
)
fig_bar.update_layout(xaxis=dict(dtick=1))

st.plotly_chart(fig_bar, use_container_width=True)


# -----------------------------
# Heatmap – Historical Pattern
# -----------------------------
st.markdown("---")
st.subheader(f"📅 Historical Congestion Heatmap – {selected_airport}")

fig_heat = px.density_heatmap(
    airport_df,
    x="hour",
    y="date",
    z="total_flights",
    color_continuous_scale="RdYlGn_r",
    title="Flights per Hour (Historical Heatmap)"
)
fig_heat.update_layout(yaxis=dict(autorange="reversed"))

st.plotly_chart(fig_heat, use_container_width=True)


# -----------------------------
# Raw Table
# -----------------------------
st.markdown("---")
st.subheader("Detailed Hourly Data")

st.dataframe(
    day_df[
        ["timestamp", "AIRPORT", "AIRPORT_NAME", "total_flights",
         "congestion_level", "predicted_congestion"]
    ].sort_values("timestamp")
)