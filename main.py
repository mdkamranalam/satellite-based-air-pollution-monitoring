import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load data from raw_data.csv (provided earlier)
def load_data():
    # Load the raw CSV provided
    data = pd.read_csv('raw_data.csv')
    
    # For compatibility with original merge logic, split into satellite, ground, and reanalysis
    # In raw_data.csv, all data is in one file, so we simulate splitting
    satellite_data = data[['lat', 'lon', 'time', 'aod', 'wind_speed']].copy()
    ground_data = data[['lat', 'lon', 'time', 'ground_pm25']].copy()
    reanalysis_data = data[['lat', 'lon', 'time', 'reanalysis_pm25']].copy()
    
    return satellite_data, ground_data, reanalysis_data

# Merge datasets with more flexible matching
def merge_data(satellite_data, ground_data, reanalysis_data):
    # Convert time to datetime
    satellite_data['time'] = pd.to_datetime(satellite_data['time'])
    ground_data['time'] = pd.to_datetime(ground_data['time'])
    reanalysis_data['time'] = pd.to_datetime(reanalysis_data['time'])
    
    # Round lat/lon to 2 decimals for merging
    satellite_data['lat_lon'] = satellite_data.apply(lambda x: f"{round(x['lat'], 2)}_{round(x['lon'], 2)}", axis=1)
    ground_data['lat_lon'] = ground_data.apply(lambda x: f"{round(x['lat'], 2)}_{round(x['lon'], 2)}", axis=1)
    reanalysis_data['lat_lon'] = reanalysis_data.apply(lambda x: f"{round(x['lat'], 2)}_{round(x['lon'], 2)}", axis=1)
    
    # Merge on lat_lon and date (ignoring exact time for simplicity)
    satellite_data['date'] = satellite_data['time'].dt.date
    ground_data['date'] = ground_data['time'].dt.date
    reanalysis_data['date'] = reanalysis_data['time'].dt.date
    
    # Merge datasets
    merged = satellite_data.merge(ground_data, on=['lat_lon', 'date'], suffixes=('_sat', '_ground'), how='inner')
    merged = merged.merge(reanalysis_data, on=['lat_lon', 'date'], suffixes=('', '_reanalysis'), how='inner')
    
    # Select relevant columns
    merged = merged[['lat_sat', 'lon_sat', 'time_sat', 'aod', 'wind_speed', 'reanalysis_pm25', 'ground_pm25']]
    merged.columns = ['lat', 'lon', 'time', 'aod', 'wind_speed', 'reanalysis_pm25', 'ground_pm25']
    
    # Check if merged data is empty
    if merged.empty:
        raise ValueError("Merged dataset is empty. Check if lat/lon/time values align across datasets.")
    
    return merged

# Train Random Forest model
def train_model(data):
    if len(data) < 5:  # Minimum threshold for train-test split
        raise ValueError(f"Dataset too small ({len(data)} rows). Need at least 5 rows for train-test split.")
    
    X = data[['aod', 'wind_speed', 'reanalysis_pm25']]
    y = data['ground_pm25']
    
    # Adjust test_size to ensure non-empty train set
    test_size = min(0.2, (len(data) - 1) / len(data))  # Ensure at least 1 sample in test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse}")
    
    return model, X, y

# Save predictions for visualization
def save_predictions(model, data):
    X = data[['aod', 'wind_speed', 'reanalysis_pm25']]
    data['predicted_pm25'] = model.predict(X)
    data[['lat', 'lon', 'time', 'ground_pm25', 'predicted_pm25']].to_csv('predictions.csv', index=False)

def main():
    try:
        # Load and merge data
        satellite_data, ground_data, reanalysis_data = load_data()
        merged_data = merge_data(satellite_data, ground_data, reanalysis_data)
        print(f"Merged dataset size: {len(merged_data)} rows")
        
        # Train model
        model, X, y = train_model(merged_data)
        
        # Save predictions
        save_predictions(model, merged_data)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()