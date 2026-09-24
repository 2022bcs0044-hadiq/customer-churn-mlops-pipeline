import pytest
import os
import json
import joblib
import numpy as np

FEATURE_NAMES_PATH = "models/feature_names.json"
MODEL_PATH = "models/churn_model.pkl"
SCALER_PATH = "models/scaler.pkl"
THRESHOLD_PATH = "models/threshold.json"

MODELS_AVAILABLE = all(
    os.path.exists(p)
    for p in [MODEL_PATH, SCALER_PATH, THRESHOLD_PATH, FEATURE_NAMES_PATH]
)


def test_feature_names_file_validity():
    assert os.path.exists(FEATURE_NAMES_PATH), "feature_names.json must exist"
    with open(FEATURE_NAMES_PATH) as f:
        features = json.load(f)
    assert isinstance(features, list)
    assert len(features) > 0
    assert all(isinstance(col, str) for col in features)


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Model artifacts not found — run dvc repro first")
def test_threshold_value_validity():
    with open(THRESHOLD_PATH) as f:
        data = json.load(f)
    assert "threshold" in data
    assert 0.0 <= data["threshold"] <= 1.0


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="Model artifacts not found — run dvc repro first")
def test_model_prediction_output():
    with open(FEATURE_NAMES_PATH) as f:
        features = json.load(f)

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    # Mock input with zeros matching feature count
    dummy_input = np.zeros((1, len(features)))
    scaled_input = scaler.transform(dummy_input)

    probs = model.predict_proba(scaled_input)
    assert probs.shape == (1, 2)
    assert 0.0 <= probs[0][1] <= 1.0
