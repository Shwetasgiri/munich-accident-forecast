import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib
import warnings
warnings.filterwarnings("ignore")

# 1. Load Data
df = pd.read_csv('accidents.csv')
df = df[df['MONAT'] != 'Summe']
df['date'] = pd.to_datetime(df['MONAT'], format='%Y%m')
df = df.set_index('date').sort_index()

categories = ['Alkoholunfälle', 'Fluchtunfälle', 'Verkehrsunfälle']
models_dict = {}

print("--- 📉 Training Conservative Models (Lower Error) ---")

for cat in categories:
    print(f"Refining model for: {cat}...")
    # Filter for category and 'insgesamt' (total)
    cat_df = df[(df['MONATSZAHL'] == cat) & (df['AUSPRAEGUNG'] == 'insgesamt')]
    
    # MISSION CONSTRAINT: Only train on data up to Dec 2020
    train_series = cat_df[cat_df.index.year <= 2020]['WERT'].interpolate()
    
    # CONSERVATIVE SARIMA: 
    # order=(0,1,1) removes the upward 'drift' or bias.
    # trend='n' ensures it doesn't just assume growth because of past years.
    model = SARIMAX(train_series, 
                    order=(0, 1, 1), 
                    seasonal_order=(0, 1, 1, 12),
                    trend='n', 
                    enforce_stationarity=True)
    
    model_fit = model.fit(disp=False)
    models_dict[cat] = model_fit

# Save all 3 models in one file
joblib.dump(models_dict, 'models_dict.pkl')
print("\n✅ Success! Multi-category models saved to 'models_dict.pkl'")