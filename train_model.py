import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import joblib

# 1. Load and Clean
df = pd.read_csv('accidents.csv')
df = df[(df['MONAT'] != 'Summe') & (df['AUSPRAEGUNG'] == 'insgesamt')]

# 2. Prepare Time Series
df['date'] = pd.to_datetime(df['MONAT'], format='%Y%m')
df = df.set_index('date').sort_index()

# 3. Filter for Target
target_cat = 'Alkoholunfälle'
cat_df = df[df['MONATSZAHL'] == target_cat]

# 4. Mission 1 Cutoff (Train only on data <= 2020)
train_series = cat_df[cat_df.index.year <= 2020]['WERT'].astype(float)

# 5. Improved Modeling: Log Transform to stabilize variance
# Adding 1 to avoid log(0) issues
train_log = np.log1p(train_series)

# Using tuned parameters for better accuracy
model = SARIMAX(train_log, 
                order=(1, 1, 1),             
                seasonal_order=(0, 1, 1, 12), 
                enforce_stationarity=False,
                enforce_invertibility=False)

model_fit = model.fit(disp=False)

# 6. Save fresh model
joblib.dump(model_fit, 'model.pkl')
print("SUCCESS: Model re-trained with Log-Transform for higher accuracy.")