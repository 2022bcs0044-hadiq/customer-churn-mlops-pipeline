import pandas as pd
import numpy as np
import os
import json
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Paths
X_TRAIN = "data/processed/X_train.csv"
Y_TRAIN = "data/processed/y_train.csv"
X_TEST = "data/processed/X_test.csv"
Y_TEST = "data/processed/y_test.csv"

MODEL_DIR = "models"
METRICS_DIR = "metrics"

def load_data():
    X_train = pd.read_csv(X_TRAIN)
    y_train = pd.read_csv(Y_TRAIN)
    X_test = pd.read_csv(X_TEST)
    y_test = pd.read_csv(Y_TEST)
    return X_train, X_test, y_train, y_test

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()

    # Save feature names (IMPORTANT for FastAPI inference)
    feature_names = list(X_train.columns)
    with open(f"{MODEL_DIR}/feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=4)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_scaled, y_train.values.ravel())

    # Start MLflow
    mlflow.set_experiment("customer-churn")

    with mlflow.start_run():

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            random_state=42
        )

        model.fit(X_res, y_res)

        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        print("Accuracy:", acc)
        print("F1 Score:", f1)
        print("ROC AUC:", auc)

        # Log MLflow
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 6)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)

        # Save model & scaler
        joblib.dump(model, f"{MODEL_DIR}/churn_model.pkl")
        joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

        # Save threshold
        threshold = 0.5
        with open(f"{MODEL_DIR}/threshold.json", "w") as f:
            json.dump({"threshold": threshold}, f, indent=4)

        # Save metrics
        metrics = {
            "accuracy": acc,
            "f1_score": f1,
            "roc_auc": auc
        }

        with open(f"{METRICS_DIR}/train_scores.json", "w") as f:
            json.dump(metrics, f, indent=4)

        # Log artifacts to MLflow
        mlflow.log_artifact(f"{MODEL_DIR}/churn_model.pkl")
        mlflow.log_artifact(f"{MODEL_DIR}/scaler.pkl")
        mlflow.log_artifact(f"{MODEL_DIR}/threshold.json")
        mlflow.log_artifact(f"{MODEL_DIR}/feature_names.json")
        mlflow.log_artifact(f"{METRICS_DIR}/train_scores.json")

if __name__ == "__main__":
    train()