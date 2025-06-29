import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap  # Import HeatMap from folium.plugins
from streamlit_folium import folium_static
import plotly.express as px
import os

# Check if predictions.csv exists
if not os.path.exists('predictions.csv'):
    st.error("predictions.csv not found. Please run main.py to generate it.")
    st.stop()

# Load predictions
try:
    data = pd.read_csv('predictions.csv')
    # Ensure required columns exist
    required_columns = ['lat', 'lon', 'time', 'ground_pm25', 'predicted_pm25']
    if not all(col in data.columns for col in required_columns):
        st.error("predictions.csv is missing required columns: lat, lon, time, ground_pm25, predicted_pm25")
        st.stop()
except Exception as e:
    st.error(f"Error loading predictions.csv: {str(e)}")
    st.stop()

# Convert time to datetime
data['time'] = pd.to_datetime(data['time'])

# Streamlit app
st.title("Satellite-based Air Pollution Monitoring - Delhi")
st.markdown("Visualizing PM2.5 predictions using satellite, ground, and reanalysis data.")

# Map
st.subheader("PM2.5 Concentration Map")
try:
    m = folium.Map(location=[28.6, 77.2], zoom_start=10)  # Center on Delhi
    # Prepare data for heatmap (lat, lon, intensity)
    heat_data = [[row['lat'], row['lon'], row['predicted_pm25']] for _, row in data.iterrows()]
    HeatMap(heat_data, radius=15, blur=20).add_to(m)
    folium_static(m)
except Exception as e:
    st.error(f"Error creating heatmap: {str(e)}")

# Time-series plot
st.subheader("PM2.5 Trends (Jan 2025)")
try:
    # Aggregate by date for smoother plot
    data_agg = data.groupby(data['time'].dt.date).mean(numeric_only=True).reset_index()
    fig = px.line(data_agg, x='time', y=['ground_pm25', 'predicted_pm25'],
                  title="Ground vs Predicted PM2.5")
    st.plotly_chart(fig)
except Exception as e:
    st.error(f"Error creating time-series plot: {str(e)}")

# Data table
st.subheader("Sample Data")
st.dataframe(data.head())