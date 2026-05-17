import streamlit as st
import joblib
import numpy as np
import warnings

# Mute the scikit-learn feature name warning
warnings.filterwarnings('ignore', category=UserWarning)

# ==========================================
# PAGE CONFIGURATION (WIDE LAYOUT)
# ==========================================
st.set_page_config(page_title="Smart Hotel Optimizer", layout="wide", page_icon="🏨")

# Custom Title Header
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🏨 Smart Hotel Energy Optimizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Predict future hotel occupancy to intelligently adjust your building's heating and cooling loads.</p>", unsafe_allow_html=True)
st.markdown("---")

# ==========================================
# LOAD THE TRAINED AI MODELS
# ==========================================
try:
    demand_model = joblib.load('models/demand_model.pkl')
    energy_model = joblib.load('models/energy_model.pkl')
except FileNotFoundError:
    st.error("⚠️ Trained models not found. Please run 'python scripts/step4_modeling.py' first.")
    st.stop()

# ==========================================
# UI: MAIN DASHBOARD LAYOUT
# ==========================================
# Split the screen into Left (Inputs) and Right (Results)
left_col, right_col = st.columns([1.2, 1])

with left_col:
    # 1. Booking Details Container
    with st.container(border=True):
        st.subheader("📅 1. Booking Forecast")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            month_name = st.selectbox("Forecast Month", 
                ['January', 'February', 'March', 'April', 'May', 'June', 
                 'July', 'August', 'September', 'October', 'November', 'December'])
            month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 
                         'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
            month_num = month_map[month_name]
        with c2:
            guests = st.number_input("Average Guests", min_value=1, max_value=10, value=2, step=1)
        with c3:
            length_of_stay = st.number_input("Avg. Nights", min_value=1, max_value=14, value=3, step=1)

    # 2. Building Specifications Container
    with st.container(border=True):
        st.subheader("🏗️ 2. Building Specifications")
        st.caption("Values are Z-scores. (0.0 = Dataset Average)")
        
        b1, b2 = st.columns(2)
        with b1:
            surface_area = st.slider("Surface Area", -2.0, 2.0, 0.0, 0.1, help="Total outside skin of the building.")
            wall_area = st.slider("Wall Area", -2.0, 2.0, 0.0, 0.1, help="More wall area = better insulation.")
        with b2:
            roof_area = st.slider("Roof Area", -2.0, 2.0, 0.0, 0.1, help="More roof = higher cooling load from sun.")
            building_height = st.selectbox("Building Height", [-1.0, 1.0], format_func=lambda x: "Tall (1.0)" if x == 1.0 else "Short (-1.0)")

# ==========================================
# PREDICTION & RESULTS PANEL
# ==========================================
with right_col:
    with st.container(border=True):
        st.subheader("⚡ Analytics Engine")
        st.write("Click below to process the inputs through the Machine Learning models.")
        
        if st.button("Process Energy Load", type="primary", use_container_width=True):
            with st.spinner("Running AI Models..."):
                
                # --- Prediction Phase ---
                occupancy_pred = demand_model.predict([[month_num, guests, length_of_stay]])[0]
                occupancy_level = np.clip(occupancy_pred, 0.10, 1.0) 
                
                energy_pred = energy_model.predict([[surface_area, wall_area, roof_area, building_height]])[0]
                base_heating = energy_pred[0]
                base_cooling = energy_pred[1]
                
                # --- Integration Phase ---
                adjusted_heating = base_heating * occupancy_level
                adjusted_cooling = base_cooling * occupancy_level
                
                # --- Display Results ---
                st.success("Analysis Complete!")
                
                st.markdown("##### Occupancy Forecast")
                st.progress(float(occupancy_level), text=f"Predicted Load: {occupancy_level * 100:.1f}%")
                
                st.markdown("##### Optimized Resource Allocation")
                r1, r2 = st.columns(2)
                with r1:
                    st.metric(label="🔥 Heating Output", value=f"{adjusted_heating:.2f} kWh", 
                              delta=f"Base: {base_heating:.2f}", delta_color="off")
                with r2:
                    st.metric(label="❄️ Cooling Output", value=f"{adjusted_cooling:.2f} kWh", 
                              delta=f"Base: {base_cooling:.2f}", delta_color="off")
        else:
            st.info("Awaiting user input to calculate the final energy loads.")