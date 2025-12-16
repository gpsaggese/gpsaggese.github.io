import streamlit as st
st.set_page_config(
    page_title="Airport Congestion Dashboard",
    layout="wide"
)

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
    df = pd.read_csv("hourly_congestion.csv")

    df["DATE"] = pd.to_datetime(df["DATE"])
    df["date_only"] = df["DATE"].dt.date
    df["AIRPORT"] = df["AIRPORT"].astype(str)

    return df


@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


df = load_data()
model = load_model()

# Label decoding
rev_label_map = {
    0: "Low",
    1: "Medium",
    2: "High"
}

# --------------------------------
# UI Header
# --------------------------------
st.title("✈️ Airport Hourly Congestion Dashboard")
st.markdown(
    "This dashboard shows **predicted airport congestion by hour**, "
    "helping travelers identify the **best and busiest times** to fly."
)

# --------------------------------
# Sidebar Controls
# --------------------------------
airport_list = sorted(df["AIRPORT"].unique())
selected_airport = st.sidebar.selectbox(
    "Select Airport",
    airport_list
)

airport_df = df[df["AIRPORT"] == selected_airport]

date_list = sorted(airport_df["date_only"].unique())
selected_date = st.sidebar.selectbox(
    "Select Date",
    date_list
)

st.caption(f"📅 Selected Date: **{selected_date}**")

day_df = airport_df[airport_df["date_only"] == selected_date].copy()

# --------------------------------
# Safety Check — No data
# --------------------------------
if day_df.empty:
    st.warning("⚠️ No flight data available for this airport on the selected date.")
    st.stop()

# --------------------------------
# Predict congestion
# --------------------------------
feature_cols = [
    "departures",
    "arrivals",
    "total_flights",
    "HOUR",
    "AIRPORT"
]

day_df["AIRPORT"] = day_df["AIRPORT"].astype(str)

try:
    preds = model.predict(day_df[feature_cols])
    day_df["Congestion Level"] = [rev_label_map[p] for p in preds]
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# --------------------------------
# Safety Check — No flights
# --------------------------------
if day_df["total_flights"].max() == 0:
    st.warning("⚠️ No flights scheduled for this airport on this date.")
    st.stop()

# --------------------------------
# Create user-friendly time column
# --------------------------------
day_df["Time"] = day_df["HOUR"].apply(lambda h: f"{int(h):02d}:00")

# --------------------------------
# DAILY SUMMARY (Most important for users)
# --------------------------------
st.subheader("📊 Daily Congestion Summary")

busy_hours = day_df[day_df["Congestion Level"] != "Low"].shape[0]

if busy_hours == 0:
    st.success(
        "🟢 **Low congestion day** — flights are evenly distributed "
        "with no significant peak hours."
    )
else:
    st.warning(
        f"⚠️ **{busy_hours} hour(s)** show moderate or high congestion. "
        "Consider avoiding those times."
    )

# --------------------------------
# Peak Hour
# --------------------------------
peak_index = day_df["total_flights"].idxmax()
peak = day_df.loc[peak_index]

st.subheader("🔥 Busiest Hour of the Day")

st.metric(
    label="Peak Time",
    value=peak["Time"],
    delta=f"{int(peak['total_flights'])} flights"
)

# --------------------------------
# Congestion Chart (Simplified)
# --------------------------------
st.subheader("⏰ Flights by Time of Day")

fig = px.bar(
    day_df,
    x="Time",
    y="total_flights",
    color="Congestion Level",
    title="How busy the airport is at different times",
    labels={
        "total_flights": "Number of Flights",
        "Time": "Time of Day"
    },
    color_discrete_map={
        "Low": "#2ecc71",
        "Medium": "#f39c12",
        "High": "#e74c3c"
    }
)

fig.update_layout(
    legend_title_text="Congestion Level",
    xaxis_tickangle=0
)

st.plotly_chart(fig, use_container_width=True)

# --------------------------------
# Simple, User-Friendly Table
# --------------------------------
st.subheader("📋 Hourly Flight Summary")

display_df = day_df.rename(columns={
    "total_flights": "Total Flights"
})[["Time", "Total Flights", "Congestion Level"]]

# Remove index so no empty top-left cell
display_df = display_df.reset_index(drop=True)

styled_df = (
    display_df.style
    .set_properties(**{"text-align": "center"})
    .set_table_styles([
        {"selector": "th", "props": [("text-align", "center")]}
    ])
)

st.dataframe(styled_df, use_container_width=True)

# --------------------------------
# Explanation (helps non-technical users & TAs)
# --------------------------------
st.info(
    "ℹ️ **How congestion is predicted:**\n\n"
    "Predictions are generated using an XGBoost machine learning model "
    "trained on historical flight data, considering hourly arrivals, "
    "departures, and total flight volume."
)
