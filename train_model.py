import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load the alcohol-specific data
df = pd.read_csv('cleaned_alcohol_data.csv')

# Define input and output
X = df[['JAHR', 'MONAT_NUM']]
y = df['WERT']

# Train the model
model = LinearRegression()
model.fit(X, y)

# Save the brain
joblib.dump(model, 'model.pkl')
print("Model trained and saved as 'model.pkl'")

# Quick test for Jan 2021
test_input = pd.DataFrame([[2021, 1]], columns=['JAHR', 'MONAT_NUM'])
prediction = model.predict(test_input)
print(f"Prediction for Jan 2021: {int(prediction[0])}")