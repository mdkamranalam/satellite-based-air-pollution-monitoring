import pandas as pd
import numpy as np

# Generate simulated raw data
np.random.seed(42)
n_samples = 100

data = {
    'lat': np.random.uniform(28.4, 28.8, n_samples),
    'lon': np.random.uniform(76.8, 77.4, n_samples),
    'time': pd.date_range('2025-01-01', periods=n_samples, freq='6H'),
    'aod': np.random.uniform(0.1, 1.0, n_samples),
    'ground_pm25': np.random.uniform(20, 150, n_samples),
    'reanalysis_pm25': np.random.uniform(15, 120, n_samples),
    'wind_speed': np.random.uniform(0, 10, n_samples)
}

df = pd.DataFrame(data)
df.to_csv('raw_data.csv', index=False)
print("Generated raw_data.csv")