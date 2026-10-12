"""
Train the loan approval model and save it as model/loan_model.sav

Run:  python train_model.py
"""
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "loan_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "loan_model.sav")

NUMERIC = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History",
]
CATEGORICAL = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area",
]


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Download the dataset (see README) "
            "or run generate_sample_data.py to create test data."
        )

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Loan_Status"])

    X = df[NUMERIC + CATEGORICAL]
    y = df["Loan_Status"].map({"Y": 1, "N": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Preprocessing is part of the pipeline, so the app can pass raw values
    preprocess = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])

    model = Pipeline([
        ("preprocess", preprocess),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=5,
            random_state=42, class_weight="balanced",
        )),
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.3f}\n")
    print(classification_report(y_test, preds, target_names=["Rejected", "Approved"]))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")

    # Save accuracy so the app can display it
    with open(os.path.join(BASE_DIR, "model", "accuracy.txt"), "w") as f:
        f.write(f"{acc:.4f}")


if __name__ == "__main__":
    main()
