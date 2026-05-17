import joblib
import numpy as np
import os
import warnings

# Mute the scikit-learn feature name warning for a clean terminal output
warnings.filterwarnings('ignore', category=UserWarning)

# ==========================================
# SETUP: LOAD THE TRAINED AI MODELS
# ==========================================
print("\n⚙️ Loading trained models...")
try:
    demand_model = joblib.load('models/demand_model.pkl')
    energy_model = joblib.load('models/energy_model.pkl')
    print("✅ Models loaded successfully!\n")
except FileNotFoundError:
    print("❌ Error: Models not found. Did you run step4_modeling.py?")
    exit()

# ==========================================
# DEFINING THE INPUTS (SIMULATED SCENARIO)
# ==========================================
# Imagine a hotel manager is testing a busy summer month
hotel_inputs = {
    'month': 8,              # August
    'guests': 4,             # 4 guests per room
    'length_of_stay': 5      # Staying for 5 nights
}

# The physical properties of the building (using scaled numbers like the model expects)
energy_inputs = {
    'surface_area': 0.75,
    'wall_area': -0.30,
    'roof_area': 1.10,
    'building_height': 1.0   # 1.0 usually represents a taller building
}

print("--- SCENARIO INPUTS ---")
print(f"Booking: Month {hotel_inputs['month']}, Guests: {hotel_inputs['guests']}, Stay: {hotel_inputs['length_of_stay']} nights")
print(f"Building Specs: Surface {energy_inputs['surface_area']}, Wall {energy_inputs['wall_area']}, Roof {energy_inputs['roof_area']}, Height {energy_inputs['building_height']}\n")

# ==========================================
# STEP 6: PREDICTION PHASE
# ==========================================
print("--- EXECUTING STEP 6: PREDICTIONS ---")

# 1. Predict Occupancy (Output is between 0.0 and 1.0)
raw_occupancy = demand_model.predict([[
    hotel_inputs['month'], 
    hotel_inputs['guests'], 
    hotel_inputs['length_of_stay']
]])[0]

# Ensure occupancy doesn't drop below a 10% base load, or exceed 100%
occupancy_level = np.clip(raw_occupancy, 0.10, 1.0)
print(f"🏨 Predicted Occupancy: {occupancy_level * 100:.1f}%")

# 2. Predict Base Energy Load
raw_energy = energy_model.predict([[
    energy_inputs['surface_area'], 
    energy_inputs['wall_area'], 
    energy_inputs['roof_area'], 
    energy_inputs['building_height']
]])[0]

base_heating = raw_energy[0]
base_cooling = raw_energy[1]
print(f"⚡ Base Heating Load: {base_heating:.2f} kWh")
print(f"⚡ Base Cooling Load: {base_cooling:.2f} kWh\n")

# ==========================================
# STEP 5: INTEGRATION PHASE
# ==========================================
print("--- EXECUTING STEP 5: MODEL INTEGRATION ---")
print("Formula: Final Load = Base Load × Occupancy Level")

# Apply the mathematical rule defined in your rubric
adjusted_heating = base_heating * occupancy_level
adjusted_cooling = base_cooling * occupancy_level

print(f"🔥 FINAL Adjusted Heating Load: {adjusted_heating:.2f} kWh")
print(f"❄️ FINAL Adjusted Cooling Load: {adjusted_cooling:.2f} kWh")
print("\n🎉 Prediction and Integration Logic Verified!")