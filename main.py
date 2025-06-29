import pandas as pd
import numpy as np
from xgboost import XGBRegressor  # Changed to XGBoost
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import os


# Load data from raw_data.csv
def load_data():
    if not os.path.exists('raw_data.csv'):
        raise FileNotFoundError("raw_data.csv not found. Please generate it using generate_raw_data.py.")

    data = pd.read_csv('raw_data.csv')

    # Ensure time is datetime
    data['time'] = pd.to_datetime(data['time'])

    # Handle missing values
    data = data.dropna(subset=['aod', 'ground_pm25', 'reanalysis_pm25', 'wind_speed'])

    # Feature engineering: Add day of week, hour, and distance to Connaught Place
    data['day_of_week'] = data['time'].dt.dayofweek
    data['hour'] = data['time'].dt.hour
    data['dist_to_connaught'] = np.sqrt((data['lat'] - 28.6315) ** 2 + (data['lon'] - 77.2167) ** 2)

    return data


# Prepare features and target
def prepare_data(data):
    X = data[['aod', 'reanalysis_pm25', 'wind_speed', 'day_of_week', 'hour', 'dist_to_connaught']]
    y = data['ground_pm25']

    return X, y


# Train and evaluate XGBoost model
def train_model(X, y):
    if len(X) < 5:
        raise ValueError(f"Dataset too small ({len(X)} rows). Need at least 5 rows for training.")

    # Split data
    test_size = min(0.2, (len(X) - 1) / len(X))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train model
    model = XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE: {rmse:.2f}")

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    print(f"5-Fold CV RMSE: {cv_rmse:.2f}")

    # Save feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    feature_importance.to_csv('feature_importance.csv', index=False)
    print("\nFeature Importance:")
    print(feature_importance)

    # Save RMSE
    with open('model_metrics.txt', 'w') as f:
        f.write(f"Test RMSE: {rmse:.2f}")

    return model, X, y


# Save predictions for visualization
def save_predictions(model, data, X):
    data['predicted_pm25'] = model.predict(X)
    data[['lat', 'lon', 'time', 'ground_pm25', 'predicted_pm25']].to_csv('predictions.csv', index=False)
    print("Saved predictions to predictions.csv")


def main():
    try:
        # Load and prepare data
        data = load_data()
        print(f"Loaded dataset size: {len(data)} rows")

        X, y = prepare_data(data)

        # Train model
        model, X, y = train_model(X, y)

        # Save predictions
        save_predictions(model, data, X)
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
