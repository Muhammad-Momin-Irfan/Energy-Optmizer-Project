import streamlit as st
import joblib
import numpy as np
import os
import warnings
from PIL import Image

# Mute warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Smart Hotel Optimizer", layout="wide", page_icon="🏨")

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🏨 Navigation")
st.sidebar.write("Select a module to view:")
page = st.sidebar.radio(
    label="Navigation Menu", 
    options=["📖 Project Overview", "📊 Data Insights (EDA)", "⚡ AI Optimizer Tool"], 
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.caption("Developed for BSCS Machine Learning Final Project.")

# ==========================================
# PAGE 1: PROJECT OVERVIEW
# ==========================================
if page == "📖 Project Overview":
    st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>Smart Hotel Energy Optimizer</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: gray;'>Bridging Hospitality Demand with Building Physics via Machine Learning</h4>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("The Problem")
        st.write("""
        Modern hotels waste massive amounts of energy by heating and cooling buildings based on static, scheduled routines rather than actual human occupancy. A hotel might be pumping 100% of its heating capacity on a cold January day, even if 90% of the rooms are empty.
        """)
        st.subheader("The Solution")
        st.write("""
        This dashboard utilizes a dual-model Machine Learning architecture to solve this:
        * **Model 1 (Random Forest):** Predicts exactly how full the hotel will be based on booking parameters.
        * **Model 2 (Linear Regression):** Calculates the physical energy required to heat/cool the building based on its architecture.
        """)
    with col2:
        st.info("**System Architecture:**\n\n1. User inputs booking details & building specs.\n2. Demand AI forecasts % Occupancy.\n3. Energy AI calculates physical Base Load.\n4. Integration Engine scales the physical load by the human demand to output an optimized energy setting.")
        
# ==========================================
# PAGE 2: DATA INSIGHTS (EDA)
# ==========================================
elif page == "📊 Data Insights (EDA)":
    st.title("📊 Exploratory Data Analysis")
    st.write("Visualizations generated from our raw hospitality and energy datasets.")
    st.markdown("---")
    
    # Helper function to safely load images
    def load_image(filename):
        path = f"report/figures/{filename}"
        if os.path.exists(path):
            return Image.open(path)
        return None

    st.subheader("1. Hotel Booking Trends")
    c1, c2 = st.columns(2)
    with c1:
        img = load_image("hotel_monthly_trends.png")
        if img: st.image(img, use_container_width=True)
    with c2:
        img = load_image("hotel_guest_distribution.png")
        if img: st.image(img, use_container_width=True)
        
    st.markdown("---")
    st.subheader("2. Building Energy Physics")
    
    st.write("**Heating Load Relationships**")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        img = load_image("heating_vs_surface_area.png")
        if img: st.image(img, use_container_width=True)
    with r2:
        img = load_image("heating_vs_wall_area.png")
        if img: st.image(img, use_container_width=True)
    with r3:
        img = load_image("heating_vs_roof_area.png")
        if img: st.image(img, use_container_width=True)
    with r4:
        img = load_image("heating_vs_building_height.png")
        if img: st.image(img, use_container_width=True)

    st.write("**Feature Correlations**")
    img = load_image("energy_correlations.png")
    if img: 
        # Center the heatmap
        left, center, right = st.columns([1, 2, 1])
        with center:
            st.image(img, use_container_width=True)

# ==========================================
# PAGE 3: AI OPTIMIZER TOOL (The Calculator)
# ==========================================
elif page == "⚡ AI Optimizer Tool":
    st.title("⚡ Interactive AI Optimizer")
    st.write("Adjust the parameters below to see real-time AI energy scaling.")
    st.markdown("---")
    
    try:
        demand_model = joblib.load('models/demand_model.pkl')
        energy_model = joblib.load('models/energy_model.pkl')
    except FileNotFoundError:
        st.error("⚠️ Trained models not found. Please run 'python scripts/step4_modeling.py' first.")
        st.stop()

    left_col, right_col = st.columns([1.2, 1])

    with left_col:
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

        with st.container(border=True):
            st.subheader("🏗️ 2. Building Specifications")
            st.caption("Values are Z-scores. (0.0 = Dataset Average)")
            b1, b2 = st.columns(2)
            with b1:
                surface_area = st.slider("Surface Area", -2.0, 2.0, 0.0, 0.1)
                wall_area = st.slider("Wall Area", -2.0, 2.0, 0.0, 0.1)
            with b2:
                roof_area = st.slider("Roof Area", -2.0, 2.0, 0.0, 0.1)
                building_height = st.selectbox("Building Height", [-1.0, 1.0], format_func=lambda x: "Tall (1.0)" if x == 1.0 else "Short (-1.0)")

    with right_col:
        with st.container(border=True):
            st.subheader("⚙️ Analytics Engine")
            
            if st.button("Process Energy Load", type="primary", use_container_width=True):
                with st.spinner("Running AI Models..."):
                    occupancy_pred = demand_model.predict([[month_num, guests, length_of_stay]])[0]
                    occupancy_level = np.clip(occupancy_pred, 0.10, 1.0) 
                    
                    energy_pred = energy_model.predict([[surface_area, wall_area, roof_area, building_height]])[0]
                    base_heating = energy_pred[0]
                    base_cooling = energy_pred[1]
                    
                    adjusted_heating = base_heating * occupancy_level
                    adjusted_cooling = base_cooling * occupancy_level
                    
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