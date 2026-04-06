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

if MODELS_AVAILABLE:
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Model artifacts not found — run dvc repro first")
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
