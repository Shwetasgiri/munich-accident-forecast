import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the data
# Using the specific filename from your folder
df = pd.read_csv('accidents.csv')

# 2. Cleaning
# Keep data until 2020 and remove 'Summe' annual totals
df = df[(df['JAHR'] <= 2020) & (df['MONAT'] != 'Summe')]
df['WERT'] = pd.to_numeric(df['WERT'], errors='coerce')
df = df.dropna(subset=['WERT'])

# --- EXPERT VISUALIZATION: 3-PANEL DASHBOARD WITH X-AXIS ON EVERY GRAPH ---
# We create 3 subplots. sharex=False ensures every graph has its own x-axis labels.
fig, axes = plt.subplots(3, 1, figsize=(12, 18), sharex=False)
fig.suptitle('Historical Accident Analysis: Yearly Trends per Category', fontsize=18, fontweight='bold')

categories = ['Alkoholunfälle', 'Fluchtunfälle', 'Verkehrsunfälle']
colors = ['#d32f2f', '#f9a825', '#2e7d32'] # Professional Red, Amber, Green

for i, cat in enumerate(categories):
    # Filter data for each category (using 'insgesamt' for the total)
    cat_df = df[(df['MONATSZAHL'] == cat) & (df['AUSPRAEGUNG'] == 'insgesamt')]
    yearly = cat_df.groupby('JAHR')['WERT'].sum()
    
    # Plotting each subplot
    axes[i].plot(yearly.index, yearly.values, marker='o', color=colors[i], linewidth=2.5, markersize=8)
    
    # Adding titles and labels to EVERY subplot
    axes[i].set_title(f'Category: {cat}', fontsize=14, fontweight='bold', pad=10)
    axes[i].set_xlabel('Year', fontsize=12)
    axes[i].set_ylabel('Total Number of Accidents', fontsize=12)
    
    # Customizing the grid and ticks for better readability
    axes[i].grid(True, linestyle='--', alpha=0.6)
    axes[i].set_xticks(yearly.index) # Show every year
    axes[i].tick_params(axis='x', rotation=45) # Rotate years for clarity

# Adjust the spacing so titles and labels don't overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save the expert dashboard
plt.savefig('accident_dashboard_detailed.png')
print("Dashboard saved as: accident_dashboard_detailed.png")

# --- DATA PREP FOR MODEL ---
# Isolate alcohol data only for the prediction mission
alcohol_df = df[(df['MONATSZAHL'] == 'Alkoholunfälle') & (df['AUSPRAEGUNG'] == 'insgesamt')].copy()
alcohol_df['MONAT_NUM'] = alcohol_df['MONAT'].astype(str).str[-2:].astype(int)
alcohol_df.to_csv('cleaned_alcohol_data.csv', index=False)
print("Training data saved as: cleaned_alcohol_data.csv")