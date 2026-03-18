# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the weather dataset using pandas.
2.Preprocess the data by handling missing values and sorting by time.
3.Select features and create lag variables for temperature and PM2.5.
4.Train Random Forest models to predict temperature and PM2.5 and save the models.


## Program:
```
/*
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==============================================================================
# 1. LOAD & CLEAN DATA
# ==============================================================================
# Read raw CSV and fix column whitespace
df = pd.read_csv("weather-station-eee-block_2024_07_13.csv")
df.columns = df.columns.str.strip()

# 1.1 Chronological Sorting: Ensure time flows correctly
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)

# 1.2 Interpolation: Fill gaps (up to 10 rows) to keep the timeline continuous
cols_to_fill = ['tem', 'pm2_5', 'tsr', 'hum', 'pressure', 'wind_speed', 'illumination', 'co2']
for col in cols_to_fill:
    if col in df.columns:
        df[col] = df[col].interpolate(method='linear', limit=10)

# ==============================================================================
# 2. FEATURE ENGINEERING
# ==============================================================================
# 2.1 Cyclical Time Features: Convert hour into circle coordinates (Sin/Cos)
df['hour'] = df['time'].dt.hour
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# 2.2 Lag Features: Give the model 'Memory' of what happened 1 and 2 steps ago
targets = ['tem', 'pm2_5', 'tsr']
for t in targets:
    df[f'{t}_lag1'] = df[t].shift(1)
    df[f'{t}_lag2'] = df[t].shift(2)

# 2.3 Cleanup: Drop rows where lags are NaN and save processed data
processed_df = df.dropna(subset=['tem_lag2', 'pm2_5_lag2', 'tsr_lag2', 'hum', 'pressure']).reset_index(drop=True)
processed_df.to_csv("combined_processed_weather_data.csv", index=False)

# Define the final high-performance feature set
features = [
    'hum', 'pressure', 'wind_speed', 'illumination', 'co2',
    'hour_sin', 'hour_cos', 'tem_lag1', 'pm2_5_lag1', 'tsr_lag1'
]
# Print summary of feature engineering
print("--- Feature Engineering Summary ---")
print(f"Original rows: {len(df)}")
print(f"Processed rows (after lags/cleaning): {len(processed_df)}")
print(f"Final high-performance feature set:",features)
# ==============================================================================
# 3. TRAIN-TEST SPLIT (Chronological)
# ==============================================================================
# Take the first 80% for training and the final 20% for testing (no shuffling)
split_idx = int(len(processed_df) * 0.8)
train, test = processed_df.iloc[:split_idx], processed_df.iloc[split_idx:]
X_train, X_test = train[features], test[features]

models = {}
results = {}

# ==============================================================================
# 4. TRAINING & PERFORMANCE EVALUATION
# ==============================================================================
target_meta = {
    'tem': ('Temperature', '°C', 'red'),
    'pm2_5': ('Pollution (PM2.5)', 'µg/m³', 'green'),
    'tsr': ('Energy (Solar Radiation)', 'W/m²', 'orange')
}

for target in targets:
    y_train, y_test = train[target], test[target]
    
    # Random Forest with high-depth logic for complex weather patterns
    model = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    models[target] = model
    
    # Store metrics for interpretation
    results[target] = {
        'r2': r2_score(y_test, preds),
        'mae': mean_absolute_error(y_test, preds),
        'preds': preds,
        'actual': y_test.values
    }

# ==============================================================================
# 5. VISUALIZATION
# ==============================================================================
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

for i, target in enumerate(targets):
    label, unit, color = target_meta[target]
    res = results[target]
    
    # Plot 1: Actual vs Predicted (Showing the last 150 points for detail)
    axes[i, 0].plot(res['actual'][-150:], label='Actual', color='black', alpha=0.4, linewidth=2)
    axes[i, 0].plot(res['preds'][-150:], label='Predicted', color=color, linestyle='--', linewidth=2)
    axes[i, 0].set_title(f"{label}: Actual vs Predicted\n$R^2$: {res['r2']:.3f} | MAE: {res['mae']:.2f}")
    axes[i, 0].set_ylabel(unit)
    axes[i, 0].legend()
    axes[i, 0].grid(True, alpha=0.3)
    
    # Plot 2: Feature Importance (Which sensors influenced this target most?)
    importances = pd.Series(models[target].feature_importances_, index=features).sort_values()
    importances.plot(kind='barh', ax=axes[i, 1], color=color, alpha=0.7)
    axes[i, 1].set_title(f"Key Drivers: {label}")

plt.tight_layout()
plt.show()

# ==============================================================================
# 6. REAL-TIME PREDICTION (Next Step)
# ==============================================================================
last_row = processed_df.iloc[-1]
latest_data = pd.DataFrame([{
    'hum': last_row['hum'], 'pressure': last_row['pressure'], 'wind_speed': last_row['wind_speed'],
    'illumination': last_row['illumination'], 'co2': last_row['co2'],
    'hour_sin': last_row['hour_sin'], 'hour_cos': last_row['hour_cos'],
    'tem_lag1': last_row['tem'], 'pm2_5_lag1': last_row['pm2_5'], 'tsr_lag1': last_row['tsr']
}])

print("\n--- NEXT STEP PREDICTIONS (Using Latest Data) ---")
for target in targets:
    pred_val = models[target].predict(latest_data)[0]
    print(f"Predicted {target_meta[target][0]}: {pred_val:.2f} {target_meta[target][1]}")
    

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("weather-station-eee-block_2024_07_13.csv")
df.columns = df.columns.str.strip()
df['time'] = pd.to_datetime(df['time'], errors='coerce')

print("Original rows:", len(df))

# Only drop if target missing
df = df.dropna(subset=['tem', 'pm2_5'])

# Fill feature columns instead of dropping
df['hum'] = df['hum'].fillna(df['hum'].mean())
df['pressure'] = df['pressure'].fillna(df['pressure'].mean())
df['wind_speed'] = df['wind_speed'].fillna(df['wind_speed'].mean())
df['co2'] = df['co2'].fillna(df['co2'].mean())

# Sort by time
df = df.sort_values('time')

# Create lag features
df['Temp_Lag1'] = df['tem'].shift(1)
df['PM_Lag1'] = df['pm2_5'].shift(1)

# Only remove first row created by shift
df = df.iloc[1:]

print("Rows after preprocessing:", len(df))

# Features
X = df[['hum', 'pressure', 'wind_speed', 'co2',
        'Temp_Lag1', 'PM_Lag1']]

y_temp = df['tem']
y_pm = df['pm2_5']

print("Training samples:", len(X))

# Train models
model_temp = RandomForestRegressor(n_estimators=300, random_state=42)
model_pm = RandomForestRegressor(n_estimators=300, random_state=42)

model_temp.fit(X, y_temp)
model_pm.fit(X, y_pm)

# Save models
joblib.dump(model_temp, "temperature_model.pkl")
joblib.dump(model_pm, "pm25_model.pkl")

print("✅ Models trained and saved successfully!")


import pandas as pd
import joblib

# Load dataset
df = pd.read_csv("weather-station-eee-block_2024_07_13.csv")
df.columns = df.columns.str.strip()
df['time'] = pd.to_datetime(df['time'])

df = df.dropna(subset=['tem', 'pm2_5'])

# Sort by time
df = df.sort_values('time')

# Take LAST available day
last_date = df['time'].dt.date.max()
df_last_day = df[df['time'].dt.date == last_date]

print("Last available date in dataset:", last_date)

# Compute daily averages
daily = df_last_day.mean(numeric_only=True)

# Load trained models
model_temp = joblib.load("temperature_model.pkl")
model_pm = joblib.load("pm25_model.pkl")

# Prepare input
X_new = [[
    daily['hum'],
    daily['pressure'],
    daily['wind_speed'],
    daily['co2'],
    daily['tem'],
    daily['pm2_5']
]]

# Predict next day
pred_temp = model_temp.predict(X_new)[0]
pred_pm = model_pm.predict(X_new)[0]

print("\n===== NEXT DAY PREDICTION =====")
print("Predicted Temperature:", round(pred_temp, 2))
print("Predicted PM2.5:", round(pred_pm, 2))


import pandas as pd
import joblib

# Load dataset
df = pd.read_csv("weather-station-eee-block_2024_07_13.csv")
df.columns = df.columns.str.strip()
df['time'] = pd.to_datetime(df['time'])

# Get last available date
last_date = df['time'].max()
print("Last available date in dataset:", last_date.date())

# Get last day's data
df_last_day = df[df['time'].dt.date == last_date.date()]

# Daily average
daily = df_last_day.mean(numeric_only=True)

# Load trained models
model_temp = joblib.load("temperature_model.pkl")
model_pm = joblib.load("pm25_model.pkl")

# Prepare input with correct feature names
X_new = pd.DataFrame([{
    'hum': daily['hum'],
    'pressure': daily['pressure'],
    'wind_speed': daily['wind_speed'],
    'co2': daily['co2'],
    'Temp_Lag1': daily['tem'],
    'PM_Lag1': daily['pm2_5']
}])

# Make prediction
pred_temp = model_temp.predict(X_new)[0]
pred_pm = model_pm.predict(X_new)[0]

print("\n===== NEXT DAY PREDICTION =====")
print(f"Predicted Temperature: {pred_temp:.2f} °C")
print(f"Predicted PM2.5: {pred_pm:.2f}")

# -------------------------------------
# AQI CATEGORY FUNCTION
# -------------------------------------
import pandas as pd

def get_aqi_category(pm):

    bins = [0, 30, 60, 90, 120, 250, float("inf")]
    labels = ["Good", "Satisfactory", "Moderate",
              "Poor", "Very Poor", "Severe"]

    category = pd.cut([pred_pm], bins=bins, labels=labels)[0]
    return category
    
aqi_category = get_aqi_category(pred_pm)

print(f"AQI Category: {aqi_category}")

# -------------------------------------
# POLLUTION ALERT SYSTEM
# -------------------------------------
print("\n===== POLLUTION ALERT =====")

alert_messages = {
    "Good": "✅ Air quality expected to be safe.",
    "Satisfactory": "✅ Air quality expected to be safe.",
    "Moderate": "⚠️ Moderate pollution expected.\nSensitive groups should take precautions.",
    "Poor": "🚨 ALERT: High pollution expected tomorrow!\n⚠️ Avoid outdoor activities.\n😷 Wear mask if necessary.",
    "Very Poor": "🚨 ALERT: High pollution expected tomorrow!\n⚠️ Avoid outdoor activities.\n😷 Wear mask if necessary.",
    "Severe": "🚨 EMERGENCY: Severe pollution expected!\n❌ Avoid outdoor exposure."
}

print(alert_messages.get(aqi_category, "Status Unknown"))
print("\nSystem Ready ✅")

Developed by:SENASH AYISHA F
RegisterNumber: 212225040394
*/
*/
```

## Output:
<img width="627" height="800" alt="image" src="https://github.com/user-attachments/assets/4f311d59-c006-43a8-abbc-b34c645f08bd" />
<img width="569" height="408" alt="image" src="https://github.com/user-attachments/assets/ea616033-2fd0-4014-8626-b6bbedf2411c" />



## Result:
