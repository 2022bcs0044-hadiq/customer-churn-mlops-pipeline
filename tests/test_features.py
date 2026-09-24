import pytest
import pandas as pd
import numpy as np
from src.data.ingest_data import clean_data


def test_clean_data_removes_customer_id():
    df = pd.DataFrame({
        "customerID": ["001-ABC", "002-DEF"],
        "gender": ["Female", "Male"],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "No"],
        "PhoneService": ["Yes", "Yes"],
        "PaperlessBilling": ["Yes", "No"],
        "TotalCharges": ["100.5", "200.0"],
        "Churn": ["No", "Yes"],
    })
    cleaned = clean_data(df)
    assert "customerID" not in cleaned.columns


def test_clean_data_handles_empty_total_charges():
    df = pd.DataFrame({
        "customerID": ["001-ABC", "002-DEF"],
        "gender": ["Female", "Male"],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "No"],
        "PhoneService": ["Yes", "Yes"],
        "PaperlessBilling": ["Yes", "No"],
        "TotalCharges": ["100.5", " "],  # Empty string should be coerced and dropped
        "Churn": ["No", "Yes"],
    })
    cleaned = clean_data(df)
    assert len(cleaned) == 1
    assert cleaned["TotalCharges"].iloc[0] == 100.5


def test_clean_data_maps_yes_no_to_binary():
    df = pd.DataFrame({
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "Yes"],
        "PhoneService": ["Yes", "No"],
        "PaperlessBilling": ["Yes", "No"],
        "TotalCharges": ["50.0", "60.0"],
        "Churn": ["No", "Yes"],
    })
    cleaned = clean_data(df)
    assert set(cleaned["Partner"].unique()).issubset({0, 1})
    assert set(cleaned["Churn"].unique()).issubset({0, 1})
