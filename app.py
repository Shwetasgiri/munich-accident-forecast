from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

model = joblib.load('model.pkl')
app = FastAPI()

class PredictionRequest(BaseModel):
    year: int
    month: int

@app.post("/predict")
def predict(data: PredictionRequest):
    target_date = pd.to_datetime(f"{data.year}-{data.month:02d}-01")
    
    # Generate prediction (this will be in log scale)
    forecast = model.get_prediction(start=target_date, end=target_date)
    
    # Reverse the Log Transform: exp(x) - 1
    prediction_value = int(round(np.expm1(forecast.predicted_mean[0])))
    
    return {"prediction": prediction_value}