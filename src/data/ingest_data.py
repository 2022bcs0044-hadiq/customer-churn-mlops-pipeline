import pandas as pd
import os

def load_data(input_path):
    print("Loading dataset...")
    df = pd.read_csv(input_path)
    return df

def clean_data(df):
    print("Cleaning dataset...")

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Remove missing values
    df = df.dropna()

    # Drop customerID
    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    # Convert Yes/No columns to 1/0
    yes_no_cols = [
        "Partner", "Dependents", "PhoneService",
        "PaperlessBilling", "Churn"
    ]

    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    return df

def save_data(df, output_path):
    print("Saving cleaned dataset...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

def main():
    input_path = "data/raw/telco_churn_v1.csv"
    output_path = "data/processed/cleaned.csv"

    df = load_data(input_path)
    df_clean = clean_data(df)
    save_data(df_clean, output_path)

    print("Dataset version 2 created successfully!")

if __name__ == "__main__":
    main()