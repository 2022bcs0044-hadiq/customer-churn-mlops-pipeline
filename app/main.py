from fastapi import FastAPI
import pandas as pd
from app.model_loader import get_model_components
from app.schemas import CustomerData

app = FastAPI(title="Customer Churn Prediction API")

model, scaler, threshold, feature_names = get_model_components()

@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}

@app.post("/predict")
def predict(data: CustomerData):
    df = pd.DataFrame([data.dict()])

    # Add missing columns
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # Ensure same order
    df = df[feature_names]

    # Scale
    X_scaled = scaler.transform(df)

    # Predict
    prob = model.predict_proba(X_scaled)[0][1]
    prediction = int(prob >= threshold)

    return {
        "churn_probability": float(prob),
        "prediction": prediction
    }