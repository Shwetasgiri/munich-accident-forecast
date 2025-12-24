from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the models dictionary
models = joblib.load('models_dict.pkl')
app = FastAPI()

class PredictionRequest(BaseModel):
    year: int
    month: int

@app.get("/")
def home():
    return {"status": "API is online"}

@app.post("/predict")
def predict(data: PredictionRequest):
    # Default to Alkoholunfälle as per DPS specific mission target
    target_date = pd.to_datetime(f"{data.year}-{data.month:02d}-01")
    model = models['Alkoholunfälle']
    
    forecast = model.get_prediction(start=target_date, end=target_date)
    prediction = int(forecast.predicted_mean[0])
    
    return {"prediction": prediction}