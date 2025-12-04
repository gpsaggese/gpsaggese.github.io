import streamlit as st
st.set_page_config(page_title="Airport Congestion Dashboard", layout="wide")

import pandas as pd
import plotly.express as px
from pathlib import Path
import joblib

# --------------------------------
# Paths
# --------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "data" / "models"

# --------------------------------
# Load data & model
# --------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA / "hourly_congestion.csv")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["date_only"] = df["DATE"].dt.date
    df["AIRPORT"] = df["AIRPORT"].astype(str)
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODELS / "model.pkl")

df = load_data()
model = load_model()

# Label decoding
rev_label_map = {0: "Low", 1: "Medium", 2: "High"}

# --------------------------------
# Streamlit UI
# --------------------------------
st.title("✈️ Airport Hourly Congestion Dashboard")
st.markdown("Explore predicted congestion levels using an XGBoost model.")

# Sidebar
airport_list = sorted(df["AIRPORT"].unique())
selected_airport = st.sidebar.selectbox("Select Airport", airport_list)

airport_df = df[df["AIRPORT"] == selected_airport]

date_list = sorted(airport_df["date_only"].unique())
selected_date = st.sidebar.selectbox("Select Date", date_list)

day_df = airport_df[airport_df["date_only"] == selected_date].copy()

# --------------------------------
# SAFETY CHECK 1 — No data for this airport/date
# --------------------------------
if day_df.empty:
    st.warning("⚠️ No flights found for this airport on the selected date.")
    st.stop()

# --------------------------------
# Predict congestion
# --------------------------------
feature_cols = ["departures", "arrivals", "total_flights", "HOUR", "AIRPORT"]
day_df["AIRPORT"] = day_df["AIRPORT"].astype(str)

try:
    preds = model.predict(day_df[feature_cols])
    day_df["predicted"] = [rev_label_map[p] for p in preds]
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# --------------------------------
# SAFETY CHECK 2 — total_flights missing or all zero
# --------------------------------
if "total_flights" not in day_df.columns or day_df["total_flights"].isna().all():
    st.warning("⚠️ No total flight data available for this day.")
    st.stop()

if day_df["total_flights"].max() == 0:
    st.warning("⚠️ No flights scheduled for this airport on this date.")
    st.stop()

# --------------------------------
# Peak Hour Section
# --------------------------------
peak_index = day_df["total_flights"].idxmax()

if pd.isna(peak_index):
    st.warning("⚠️ Peak hour cannot be determined.")
else:
    peak = day_df.loc[peak_index]
    st.subheader("🔥 Peak Congestion Hour")
    st.metric(
        "Peak Hour",
        f"{int(peak['HOUR']):02d}:00",
        f"{peak['total_flights']} flights"
    )

# --------------------------------
# Plot Congestion Bar Chart
# --------------------------------
st.subheader("Hourly Congestion Prediction")

fig = px.bar(
    day_df,
    x="HOUR",
    y="total_flights",
    color="predicted",
    text="predicted",
    title=f"{selected_airport} — {selected_date}",
    color_discrete_map={"Low": "green", "Medium": "orange", "High": "red"}
)

fig.update_traces(textposition="outside")

st.plotly_chart(fig, use_container_width=True)

# --------------------------------
# Data Table
# --------------------------------
st.subheader("Hourly Details")
st.dataframe(
    day_df[["DATE", "HOUR", "departures", "arrivals", "total_flights", "predicted"]],
    use_container_width=True
)