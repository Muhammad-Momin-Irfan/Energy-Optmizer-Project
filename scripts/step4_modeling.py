import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

print("\n🤖 Starting Machine Learning Training Pipeline...")

# ==========================================
# 1. TRAIN DEMAND MODEL (Random Forest)
# ==========================================
print("Training Demand Forecast Model...")
df_hotel = pd.read_csv('data/raw/hotel_booking.csv')

# --- Feature Engineering ---
# Map months to numbers so the AI can understand them
month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 
             'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
df_hotel['month_num'] = df_hotel['arrival_date_month'].map(month_map)

# Calculate totals
df_hotel['guests'] = df_hotel['adults'] + df_hotel['children'].fillna(0) + df_hotel['babies'].fillna(0)
df_hotel['length_of_stay'] = df_hotel['stays_in_weekend_nights'] + df_hotel['stays_in_week_nights']

# Filter outliers to match our UI limits
df_hotel = df_hotel[(df_hotel['guests'] >= 1) & (df_hotel['guests'] <= 10)]
df_hotel = df_hotel[(df_hotel['length_of_stay'] >= 1) & (df_hotel['length_of_stay'] <= 14)]

# Create a normalized 'Occupancy Rate' target (0.10 to 1.0)
# We calculate this based on proximity to peak season (August) + group size + stay length
month_factor = 1.0 - (np.abs(df_hotel['month_num'] - 8) / 12.0)
guest_factor = df_hotel['guests'] / 10.0
stay_factor = df_hotel['length_of_stay'] / 14.0

# Combine factors, add slight natural noise, and clip to valid percentages
np.random.seed(42)
df_hotel['occupancy_rate'] = (month_factor * 0.6) + (guest_factor * 0.2) + (stay_factor * 0.2)
df_hotel['occupancy_rate'] += np.random.normal(0, 0.05, len(df_hotel))
df_hotel['occupancy_rate'] = np.clip(df_hotel['occupancy_rate'], 0.10, 1.0)

# --- CRITICAL FIX: EXPLICIT COLUMN ORDER ---
# This order MUST perfectly match the order in app/main.py
X_demand = df_hotel[['month_num', 'guests', 'length_of_stay']]
y_demand = df_hotel['occupancy_rate']

# Train the Random Forest
demand_model = RandomForestRegressor(n_estimators=100, random_state=42)
demand_model.fit(X_demand, y_demand)
joblib.dump(demand_model, 'models/demand_model.pkl')
print("✅ Demand Model trained and saved!")

# ==========================================
# 2. TRAIN ENERGY MODEL (Linear Regression)
# ==========================================
print("Training Building Energy Model...")
df_energy = pd.read_excel('data/raw/energy_efficiency.xlsx')

# Rename columns
rename_dict = {'X2': 'Surface Area', 'X3': 'Wall Area', 'X4': 'Roof Area', 
               'X5': 'Building Height', 'Y1': 'Heating Load', 'Y2': 'Cooling Load'}
df_energy = df_energy.rename(columns=rename_dict)

# --- CRITICAL FIX: EXPLICIT COLUMN ORDER ---
# This order MUST perfectly match the order in app/main.py
X_energy = df_energy[['Surface Area', 'Wall Area', 'Roof Area', 'Building Height']]
y_energy = df_energy[['Heating Load', 'Cooling Load']] # Multi-output target!

# --- CRITICAL FIX: STANDARD SCALING ---
# We must scale the training data to Z-scores so the model understands the UI sliders (-2.0 to 2.0)
scaler = StandardScaler()
X_energy_scaled = scaler.fit_transform(X_energy)

# Train the Linear Regression on the SCALED data
energy_model = LinearRegression()
energy_model.fit(X_energy_scaled, y_energy)
joblib.dump(energy_model, 'models/energy_model.pkl')
print("✅ Energy Model trained and saved!")

print("\n🎉 Step 4 Complete: All models are ready for the dashboard!")