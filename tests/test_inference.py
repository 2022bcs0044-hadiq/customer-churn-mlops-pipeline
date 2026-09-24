import pytest
import os

# Skip entire module gracefully if model artifacts are missing
# (e.g. when running tests in isolation before dvc repro)
MODELS_AVAILABLE = all(
    os.path.exists(p)
    for p in [
        "models/churn_model.pkl",
        "models/scaler.pkl",
        "models/threshold.json",
        "models/feature_names.json",
    ]
)

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()



@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Model artifacts not found — run dvc repro first")
def test_predict_endpoint():
    # Minimal payload — missing feature columns default to 0 inside /predict
    payload = {
        "gender": 1,
        "SeniorCitizen": 0,
        "Partner": 0,
        "Dependents": 0,
        "tenure": 12,
        "PhoneService": 1,
        "PaperlessBilling": 1,
        "MonthlyCharges": 50.0,
        "TotalCharges": 600.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "churn_probability" in body
    assert "prediction" in body
    assert body["prediction"] in (0, 1)


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Model artifacts not found — run dvc repro first")
def test_predict_full_payload():
    payload = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 36,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 95.5,
        "TotalCharges": 3438.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "churn_probability" in body
    assert "prediction" in body
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["risk_level"] in ("High", "Low")

