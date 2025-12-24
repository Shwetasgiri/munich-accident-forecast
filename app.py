from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the saved model
model = joblib.load('model.pkl')

app = FastAPI()

class PredictionInput(BaseModel):
    year: int
    month: int

@app.get("/")
def home():
    return {"message": "DPS AI Challenge API is live!"}

@app.post("/predict")
def get_prediction(data: PredictionInput):
    input_df = pd.DataFrame([[data.year, data.month]], columns=['JAHR', 'MONAT_NUM'])
    prediction = model.predict(input_df)[0]
    return {"prediction": int(prediction)}