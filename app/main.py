from fastapi import FastAPI
from app.schemas import PredictionRequest, PredictionResponse
from app.model_loader import load_model

app = FastAPI(title="Customer Churn Prediction API")

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Add prediction logic
    pass
