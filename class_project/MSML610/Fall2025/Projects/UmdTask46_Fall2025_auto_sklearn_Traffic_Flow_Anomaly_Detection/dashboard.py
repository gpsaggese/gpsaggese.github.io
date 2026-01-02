import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import os
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Traffic Anomaly Detection", layout="wide")
st.title(" Metro Interstate Traffic Anomaly Detection")
st.markdown("""
This dashboard visualizes traffic patterns, detects anomalies using unsupervised logic, 
and simulates real-time traffic monitoring.
""")

# --- DATA LOADING ---
@st.cache_data
def get_data():
    if os.path.exists('Metro_Interstate_Traffic_Volume.csv'):
        return pd.read_csv('Metro_Interstate_Traffic_Volume.csv')
    return None

df = get_data()

if df is not None:
    # --- PREPROCESSING ---
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values('date_time')
    
    # Create derived features
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    
    # --- SIDEBAR FILTERS ---
    st.sidebar.header("Control Panel")
    years = df['date_time'].dt.year.unique()
    selected_year = st.sidebar.selectbox("Select Year for Historical Analysis", years, index=len(years)-1)
    
    # Filter data based on year
    filtered_df = df[df['date_time'].dt.year == selected_year]
    
    # Define Anomaly Logic (Same as your model logic)
    # Anomaly = Weekday (0-4) AND Morning Rush (6-9) AND Low Traffic (< 500)
    busy_time = (filtered_df['day_of_week'] < 5) & (filtered_df['hour'].between(6, 9))
    low_traffic = filtered_df['traffic_volume'] < 500
    anomalies = filtered_df[busy_time & low_traffic]

    # --- SECTION 1: MAIN TRAFFIC OVERVIEW ---
    st.subheader(f"1. Traffic Volume Overview ({selected_year})")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Main Time Series Plot
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(filtered_df['date_time'], filtered_df['traffic_volume'], label='Traffic Volume', color='#1f77b4', alpha=0.6)
        ax.scatter(anomalies['date_time'], anomalies['traffic_volume'], color='red', label='Anomaly', s=50, zorder=5)
        ax.set_xlabel("Date Time")
        ax.set_ylabel("Traffic Volume")
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Statistics**")
        st.metric("Total Records", len(filtered_df))
        st.metric("Detected Anomalies", len(anomalies))
        st.metric("Avg Traffic Volume", int(filtered_df['traffic_volume'].mean()))

    st.markdown("---")

    # --- SECTION 2: HEATMAP (LEVEL 1) ---
    st.subheader("2. Traffic Patterns Heatmap (Day vs. Hour)")
    st.markdown("Brighter colors indicate heavy traffic. Anomalies often appear as 'dark spots' during usually bright hours.")
    
    # Create pivot table for heatmap
    heatmap_data = filtered_df.groupby(['day_of_week', 'hour'])['traffic_volume'].mean().reset_index()
    
    # Map 0-6 to Mon-Sun for better readability
    days = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
    heatmap_data['day_label'] = heatmap_data['day_of_week'].map(days)
    
    fig_heat = px.density_heatmap(
        heatmap_data, 
        x='hour', 
        y='day_label', 
        z='traffic_volume', 
        title='Average Traffic Intensity',
        color_continuous_scale='Viridis',
        labels={'day_label': 'Day of Week', 'hour': 'Hour of Day', 'traffic_volume': 'Volume'},
        category_orders={"day_label": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]}
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # --- SECTION 3: WEATHER CORRELATION (LEVEL 2) ---
    st.subheader("3. 🌧️ Impact of Weather on Traffic")
    
    # Create a scatter plot: X=Temp, Y=Traffic, Color=Rain
    fig_weather = px.scatter(
        filtered_df, 
        x='temp', 
        y='traffic_volume', 
        color='rain_1h',
        size_max=10,
        title='Correlation: Temperature vs. Traffic Volume (Colored by Rain Intensity)',
        hover_data=['weather_description'],
        color_continuous_scale='Bluered'
    )
    
    # Overlay anomalies as red 'X' markers
    fig_weather.add_scatter(
        x=anomalies['temp'], 
        y=anomalies['traffic_volume'], 
        mode='markers', 
        marker=dict(color='red', symbol='x', size=10),
        name='Anomalies'
    )
    
    st.plotly_chart(fig_weather, use_container_width=True)

    # --- SECTION 4: LIVE SIMULATION (LEVEL 3) ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔴 Real-Time Simulation")
    
    if st.sidebar.button("Start Live Feed"):
        st.divider()
        st.subheader("🔴 Live Traffic Monitor Simulation")
        
        # Create placeholders for live updates
        metric_placeholder = st.empty()
        chart_placeholder = st.empty()
        
        # Get the last 100 data points to simulate "recent" history
        sim_data = df.iloc[-100:].reset_index(drop=True)
        
        # Loop to simulate streaming data
        for i in range(10, len(sim_data)):
            # Slice data up to current point
            current_slice = sim_data.iloc[:i]
            last_row = current_slice.iloc[-1]
            prev_row = current_slice.iloc[-2]
            
            # Calculate delta
            diff = int(last_row['traffic_volume'] - prev_row['traffic_volume'])
            
            with metric_placeholder.container():
                cols = st.columns(3)
                cols[0].metric("Current Traffic", int(last_row['traffic_volume']), delta=diff)
                cols[1].metric("Current Temp (K)", f"{last_row['temp']:.1f}")
                cols[2].metric("Weather", last_row['weather_main'])

            # Live Line Chart
            fig_live = px.line(
                current_slice, 
                x='date_time', 
                y='traffic_volume', 
                title="Live Incoming Traffic Data",
                markers=True
            )
            chart_placeholder.plotly_chart(fig_live, use_container_width=True)
            
            # Speed of simulation (0.1s per data point)
            time.sleep(0.1)

else:
    st.error("Dataset not found. Please ensure 'Metro_Interstate_Traffic_Volume.csv' is in the project folder.")