import joblib
import json
import os

MODEL_DIR = "models"

MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.json")
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_names.json")

def get_model_components():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(THRESHOLD_PATH) as f:
        threshold = json.load(f)["threshold"]

    with open(FEATURE_PATH) as f:
        feature_names = json.load(f)

    return model, scaler, threshold, feature_names