from fastapi import FastAPI, HTTPException
import pandas as pd
from app.model_loader import get_model_components
from app.schemas import CustomerData

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-ready real-time inference API for predicting customer churn probability.",
    version="1.1.0",
)

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
    return {
        "message": "Customer Churn Prediction API is running",
        "status": "healthy",
        "version": "1.1.0"
    }


@app.post("/predict")
def predict(data: CustomerData):
    _load_components()

    # Convert Pydantic model to dict, then DataFrame
    raw_dict = data.model_dump() if hasattr(data, "model_dump") else data.dict()
    df_raw = pd.DataFrame([raw_dict])

    # Perform one-hot encoding on categorical fields
    df_encoded = pd.get_dummies(df_raw, dtype=int)

    # Reindex to exact 41 feature columns learned during training, filling absent columns with 0
    df_aligned = df_encoded.reindex(columns=_feature_names, fill_value=0).astype(float)

    # Scale features using the persisted training StandardScaler
    X_scaled = _scaler.transform(df_aligned)

    # Predict churn probability
    prob = float(_model.predict_proba(X_scaled)[0][1])
    prediction = int(prob >= _threshold)

    return {
        "churn_probability": round(prob, 4),
        "prediction": prediction,
        "risk_level": "High" if prob >= _threshold else "Low",
        "threshold_used": _threshold,
    }