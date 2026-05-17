import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Mute minor seaborn warnings to keep the terminal clean
warnings.filterwarnings('ignore', category=FutureWarning)

# Create the folder to save your report images
os.makedirs('report/figures', exist_ok=True)

# Set the visual style for the graphs
sns.set_theme(style="whitegrid")

# ==========================================
# 1. HOTEL BOOKING EDA 
# ==========================================
print("Generating Hotel Booking Visualizations...")

df_hotel_raw = pd.read_csv('data/raw/hotel_booking.csv')
df_hotel_raw['total_guests'] = df_hotel_raw['adults'] + df_hotel_raw['children'].fillna(0) + df_hotel_raw['babies']

# Graph A: Monthly Booking Trends
plt.figure(figsize=(10, 6))
months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                'July', 'August', 'September', 'October', 'November', 'December']
sns.countplot(data=df_hotel_raw, x='arrival_date_month', order=months_order, hue='arrival_date_month', palette='viridis', legend=False)
plt.title('Monthly Booking Trends', fontsize=16)
plt.xlabel('Month', fontsize=12)
plt.ylabel('Number of Bookings', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('report/figures/hotel_monthly_trends.png', dpi=300)
plt.close()

# Graph B: Distribution of Number of Guests
plt.figure(figsize=(8, 6))
sns.histplot(df_hotel_raw[df_hotel_raw['total_guests'] <= 10]['total_guests'], bins=10, kde=False, color='coral')
plt.title('Distribution of Number of Guests', fontsize=16)
plt.xlabel('Total Guests per Booking', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.tight_layout()
plt.savefig('report/figures/hotel_guest_distribution.png', dpi=300)
plt.close()

# ==========================================
# 2. ENERGY EFFICIENCY EDA
# ==========================================
print("Generating Energy Efficiency Visualizations...")

df_energy_raw = pd.read_excel('data/raw/energy_efficiency.xlsx')
rename_dict = {'X2': 'Surface Area', 'X3': 'Wall Area', 'X4': 'Roof Area', 
               'X5': 'Building Height', 'Y1': 'Heating Load', 'Y2': 'Cooling Load'}
df_energy_raw = df_energy_raw.rename(columns=rename_dict)

# Graph C: Heating Load Scatter Plots (Now separated into 4 distinct files!)
features = ['Surface Area', 'Wall Area', 'Roof Area', 'Building Height']
colors = ['#E63946', '#2A9D8F', '#E9C46A', '#264653'] # Distinct colors for each graph

for feature, color in zip(features, colors):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_energy_raw, x=feature, y='Heating Load', alpha=0.7, color=color, s=60)
    plt.title(f'Heating Load vs {feature}', fontsize=16)
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Heating Load', fontsize=12)
    plt.tight_layout()
    
    # Format the filename so there are no spaces (e.g., 'Surface Area' becomes 'surface_area')
    safe_filename = feature.replace(' ', '_').lower()
    plt.savefig(f'report/figures/heating_vs_{safe_filename}.png', dpi=300)
    plt.close()

# Graph D: Correlation Heatmap
plt.figure(figsize=(8, 6))
correlation_matrix = df_energy_raw[features + ['Heating Load', 'Cooling Load']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap (Cooling & Heating Loads)', fontsize=14)
plt.tight_layout()
plt.savefig('report/figures/energy_correlations.png', dpi=300)
plt.close()

print("✅ All EDA graphs have been generated and saved separately!")