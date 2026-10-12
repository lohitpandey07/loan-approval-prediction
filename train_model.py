"""
Train the loan approval model and save it as model/loan_model.sav

Run:  python train_model.py
"""
import os
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from loan_model import ACCURACY_PATH, FEATURES, MODEL_PATH, BASE_DIR, build_pipeline

DATA_PATH = os.path.join(BASE_DIR, "data", "loan_data.csv")


def train(df, test_size=0.2, seed=42):
    """Train on a DataFrame. Returns (model, X_test, y_test)."""
    df = df.dropna(subset=["Loan_Status"])
    X = df[FEATURES]
    y = df["Loan_Status"].map({"Y": 1, "N": 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    model = build_pipeline()
    model.fit(X_train, y_train)
    return model, X_test, y_test


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Download the dataset (see README) "
            "and save it as data/loan_data.csv."
        )

    model, X_test, y_test = train(pd.read_csv(DATA_PATH))

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc:.3f}\n")
    print(classification_report(y_test, preds, target_names=["Rejected", "Approved"]))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {MODEL_PATH}")

    # Save accuracy so the app can display it
    with open(ACCURACY_PATH, "w") as f:
        f.write(f"{acc:.4f}")


if __name__ == "__main__":
    main()
