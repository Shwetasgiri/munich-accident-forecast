import pandas as pd
import joblib
import os

# 1. Load the Model and the Original Data
if not os.path.exists('model.pkl'):
    print("Error: model.pkl not found. Please run train_model.py first.")
    exit()

model = joblib.load('model.pkl')
data_file = 'accidents.csv'
df = pd.read_csv(data_file)

def run_evaluation():
    print("--- 🧠 AI Model Evaluation Tool ---")
    try:
        # User input for the date
        year = int(input("Enter Year (e.g., 2021): "))
        month = int(input("Enter Month (1-12): "))
        
        if not (1 <= month <= 12):
            print("Invalid month. Please enter a number between 1 and 12.")
            return

        # Prepare input for the model
        input_data = pd.DataFrame([[year, month]], columns=['JAHR', 'MONAT_NUM'])
        
        # 2. Get AI Prediction
        prediction = model.predict(input_data)[0]
        
        # 3. Search for Ground Truth (Real Data) in the CSV
        # Format the month string to match the CSV (e.g., 202101)
        monat_str = f"{year}{month:02d}"
        
        real_row = df[
            (df['MONATSZAHL'] == 'Alkoholunfälle') & 
            (df['AUSPRAEGUNG'] == 'insgesamt') & 
            (df['MONAT'] == monat_str)
        ]
        
        print("\n" + "="*40)
        print(f"RESULTS FOR: {year}-{month:02d}")
        print(f"AI Prediction:  {prediction:.2f} accidents")
        
        if not real_row.empty:
            actual = real_row['WERT'].values[0]
            # Handle cases where WERT might be NaN in the CSV
            if pd.isna(actual):
                print("Actual Reality: Data exists in CSV but is empty (NaN).")
            else:
                error = abs(prediction - actual)
                print(f"Actual Reality: {actual} accidents")
                print(f"Absolute Error: {error:.2f}")
                
                # Expert Commentary
                if error < 2:
                    print("Status: 🌟 Extremely Accurate!")
                elif error < 5:
                    print("Status: ✅ Good Prediction.")
                else:
                    print("Status: ⚠️ Higher variance detected.")
        else:
            print("Actual Reality: No real data found for this date (True Future).")
            print("Status: 🔮 Pure Forecast Mode.")
        
        print("="*40 + "\n")

    except ValueError:
        print("Error: Please enter valid numbers for Year and Month.")

if __name__ == "__main__":
    while True:
        run_evaluation()
        cont = input("Do you want to check another date? (y/n): ").lower()
        if cont != 'y':
            break