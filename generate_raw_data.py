import pandas as pd
import numpy as np

# Generate simulated raw data for Central Delhi
np.random.seed(42)  # For reproducibility
n_samples = 1000

# Central Delhi coordinates: lat 28.5–28.7, lon 77.1–77.3
data = {
    'lat': np.random.uniform(28.5, 28.7, n_samples),
    'lon': np.random.uniform(77.1, 77.3, n_samples),
    'time': pd.date_range('2025-01-01', periods=n_samples, freq='6H'),
    'aod': np.random.uniform(0.3, 1.0, n_samples),  # Higher AOD for urban Delhi
    'ground_pm25': np.random.uniform(30, 150, n_samples),  # Realistic PM2.5 for Delhi
    'reanalysis_pm25': np.random.uniform(25, 120, n_samples),  # CAMS PM2.5
    'wind_speed': np.random.uniform(0, 10, n_samples)  # Wind speed in m/s
}

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv('raw_data.csv', index=False)
print("Generated raw_data.csv for Central Delhi")