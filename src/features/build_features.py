import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def feature_engineering(df):
    print("Performing feature engineering...")

    # Target variable
    y = df["Churn"]
    X = df.drop("Churn", axis=1)

    # One-hot encoding for categorical variables
    X = pd.get_dummies(X)

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)

    return X, y

def split_data(X, y):
    print("Splitting dataset...")
    return train_test_split(X, y, test_size=0.2, random_state=42)

def save_data(X_train, X_test, y_train, y_test):
    os.makedirs("data/processed", exist_ok=True)

    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

def main():
    df = load_data("data/processed/cleaned.csv")
    X, y = feature_engineering(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    save_data(X_train, X_test, y_train, y_test)

    print("Feature engineering completed!")

if __name__ == "__main__":
    main()