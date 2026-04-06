from fastapi import FastAPI, HTTPException
import pandas as pd
from app.model_loader import get_model_components
from app.schemas import CustomerData

app = FastAPI(title="Customer Churn Prediction API")

# Lazy-loaded model components — loaded on first request, not at import time
_model = None
_scaler = None
_threshold = None
_feature_names = None


def _load_components():
    global _model, _scaler, _threshold, _feature_names
    if _model is None:
        try:
            _model, _scaler, _threshold, _feature_names = get_model_components()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model artifacts not available: {e}"
            )


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def predict(data: CustomerData):
    _load_components()

    df = pd.DataFrame([data.dict()])

    # Add missing columns with default 0
    for col in _feature_names:
        if col not in df.columns:
            df[col] = 0

    # Ensure same column order as training
    df = df[_feature_names]

    # Scale and predict
    X_scaled = _scaler.transform(df)
    prob = _model.predict_proba(X_scaled)[0][1]
    prediction = int(prob >= _threshold)

    return {
        "churn_probability": float(prob),
        "prediction": prediction,
    }