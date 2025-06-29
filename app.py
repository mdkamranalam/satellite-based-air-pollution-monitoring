import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
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
st.title("Air Pollution Monitoring - Central Delhi")
st.markdown("Visualizing PM2.5 predictions using satellite, ground, and reanalysis data.")

# Model Performance
st.subheader("Model Performance")
try:
    with open('model_metrics.txt', 'r') as f:
        rmse_text = f.read()
    st.write(rmse_text)
except:
    st.write("Model metrics not available.")

# Feature Importance
st.subheader("What Drives PM2.5 Predictions?")
try:
    feature_importance = pd.read_csv('feature_importance.csv')
    fig = px.bar(feature_importance, x='feature', y='importance',
                 title="Feature Importance for PM2.5 Prediction",
                 labels={'importance': 'Importance Score', 'feature': 'Feature'})
    st.plotly_chart(fig)
except Exception as e:
    st.error(f"Error loading feature importance: {str(e)}")

# Date filter
st.subheader("Select Date for Analysis")
dates = data['time'].dt.date.unique()
selected_date = st.selectbox("Choose a date:", dates)
filtered_data = data[data['time'].dt.date == selected_date]

# High-risk areas (PM2.5 > 50 µg/m³)
st.subheader(f"High-Risk Areas on {selected_date} (PM2.5 > 50 µg/m³)")
high_risk = filtered_data[filtered_data['predicted_pm25'] > 50][['lat', 'lon', 'predicted_pm25']]
if high_risk.empty:
    st.write("No high-risk areas detected.")
else:
    st.dataframe(high_risk)

# Map with heatmap and high-risk markers
st.subheader(f"PM2.5 Concentration Map - Central Delhi ({selected_date})")
try:
    m = folium.Map(location=[28.6, 77.2], zoom_start=12)  # Center on Central Delhi
    heat_data = [[row['lat'], row['lon'], row['predicted_pm25']] for _, row in filtered_data.iterrows()]
    HeatMap(heat_data, radius=15, blur=20, gradient={0.4: 'blue', 0.65: 'yellow', 1: 'red'}).add_to(m)
    # Add markers for high-risk areas
    for _, row in high_risk.iterrows():
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=f"PM2.5: {row['predicted_pm25']:.2f} µg/m³",
            icon=folium.Icon(color='red')
        ).add_to(m)
    folium_static(m)
except Exception as e:
    st.error(f"Error creating heatmap: {str(e)}")

# Time-series plot
st.subheader(f"PM2.5 Trends (Jan 2025)")
try:
    data_agg = data.groupby(data['time'].dt.date).mean(numeric_only=True).reset_index()
    fig = px.line(data_agg, x='time', y=['ground_pm25', 'predicted_pm25'],
                  title="Ground vs Predicted PM2.5 in Central Delhi")
    st.plotly_chart(fig)
except Exception as e:
    st.error(f"Error creating time-series plot: {str(e)}")

# Data table
st.subheader("Sample Data")
st.dataframe(filtered_data.head())
