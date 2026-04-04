import pandas as pd
import json
import joblib
import os

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

X_TEST = "data/processed/X_test.csv"
Y_TEST = "data/processed/y_test.csv"
MODEL_PATH = "models/churn_model.pkl"

METRICS_DIR = "metrics"

def evaluate():
    os.makedirs(METRICS_DIR, exist_ok=True)

    X_test = pd.read_csv(X_TEST)
    y_test = pd.read_csv(Y_TEST)

    model = joblib.load(MODEL_PATH)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)

    print("Evaluation Metrics")
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("ROC AUC:", auc)
    print("Confusion Matrix:")
    print(cm)

    metrics = {
        "accuracy": acc,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist()
    }

    with open(f"{METRICS_DIR}/test_scores.json", "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    evaluate()