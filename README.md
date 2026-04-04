# Customer Churn MLOps Pipeline

This repository contains an end-to-end Machine Learning pipeline for predicting customer churn.

## Features
- **Data Versioning**: DVC (Data Version Control) tracking raw & processed datasets.
- **Experiment Tracking**: MLflow for logging model parameters, metrics (AUC, F1), and artifacts.
- **Model Training**: XGBoost binary classifier.
- **Serving**: FastAPI for real-time predictions.
- **CI/CD**: GitHub Actions for automated testing.
- **Containerization**: Docker & Docker Compose for reproducibility.
