import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Create the processed data folder if it doesn't exist
os.makedirs('data/processed', exist_ok=True)

# ==========================================
# 1. HOTEL BOOKING DATASET PREPROCESSING
# ==========================================
print("--- Processing Hotel Booking Dataset ---")

# Load the raw dataset
df_hotel = pd.read_csv('data/raw/hotel_booking.csv')

# Fill missing 'children' values with 0
df_hotel['children'] = df_hotel['children'].fillna(0)

# Feature Engineering
month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 
             'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
df_hotel['month'] = df_hotel['arrival_date_month'].map(month_map)
df_hotel['number_of_guests'] = df_hotel['adults'] + df_hotel['children'] + df_hotel['babies']
df_hotel['length_of_stay'] = df_hotel['stays_in_weekend_nights'] + df_hotel['stays_in_week_nights']

# Target Variable: Occupancy
# 1 - 1 (canceled) = 0% occupancy. 1 - 0 (not canceled) = 100% occupancy.
df_hotel['occupancy'] = 1 - df_hotel['is_canceled']

# Feature Selection
hotel_features = ['month', 'number_of_guests', 'length_of_stay', 'occupancy']
df_hotel_processed = df_hotel[hotel_features].copy()

# Remove anomalies
df_hotel_processed = df_hotel_processed[(df_hotel_processed['length_of_stay'] > 0) & 
                                        (df_hotel_processed['number_of_guests'] > 0)]

# NOTE: StandardScaler removed here so the model trains on raw inputs (Month 1-12, Guests 1-10)
# This ensures it matches the raw inputs coming from the Streamlit main.py UI.

# Save processed data
df_hotel_processed.to_csv('data/processed/hotel_processed.csv', index=False)
print("✅ Hotel data processed and saved (Raw Inputs Kept)!\n")

# ==========================================
# 2. ENERGY EFFICIENCY DATASET PREPROCESSING
# ==========================================
print("--- Processing Energy Efficiency Dataset ---")

# Load the raw dataset
df_energy = pd.read_excel('data/raw/energy_efficiency.xlsx')

# Rename features
rename_dict = {
    'X2': 'surface_area',
    'X3': 'wall_area',
    'X4': 'roof_area',
    'X5': 'building_height',
    'Y1': 'heating_load',
    'Y2': 'cooling_load'
}
df_energy = df_energy.rename(columns=rename_dict)

# Feature Selection
energy_features = ['surface_area', 'wall_area', 'roof_area', 'building_height', 'heating_load', 'cooling_load']
df_energy_processed = df_energy[energy_features].dropna().copy()

# Feature Scaling
# We MUST scale these to Z-scores (-2.0 to 2.0) to match the sliders in the web app.
scaler_energy = StandardScaler()
energy_inputs = ['surface_area', 'wall_area', 'roof_area', 'building_height']
df_energy_processed[energy_inputs] = scaler_energy.fit_transform(df_energy_processed[energy_inputs])

# Save processed data
df_energy_processed.to_csv('data/processed/energy_processed.csv', index=False)
print("✅ Energy data processed and scaled via Z-scores!")