"""
Generates a SYNTHETIC dataset with the same columns as the Kaggle
"Loan Prediction" dataset. It is used by the automated tests, and you can
run it to try the project without downloading anything:

    python generate_sample_data.py

For real results, use the real dataset (see README.md) as data/loan_data.csv.
"""
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def make_sample_data(n=600, seed=42):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "Loan_ID": [f"LP{str(i).zfill(6)}" for i in range(n)],
        "Gender": rng.choice(["Male", "Female"], n, p=[0.8, 0.2]),
        "Married": rng.choice(["Yes", "No"], n, p=[0.65, 0.35]),
        "Dependents": rng.choice(["0", "1", "2", "3+"], n, p=[0.57, 0.17, 0.16, 0.10]),
        "Education": rng.choice(["Graduate", "Not Graduate"], n, p=[0.78, 0.22]),
        "Self_Employed": rng.choice(["Yes", "No"], n, p=[0.14, 0.86]),
        "ApplicantIncome": rng.lognormal(8.5, 0.5, n).round(0),
        "CoapplicantIncome": np.where(
            rng.random(n) < 0.4, 0, rng.lognormal(7.5, 0.7, n).round(0)
        ),
        "LoanAmount": rng.normal(140, 60, n).clip(20, 600).round(0),
        "Loan_Amount_Term": rng.choice(
            [360, 180, 480, 300, 240, 120], n,
            p=[0.8, 0.07, 0.05, 0.04, 0.02, 0.02],
        ),
        "Credit_History": rng.choice([1.0, 0.0], n, p=[0.84, 0.16]),
        "Property_Area": rng.choice(["Urban", "Semiurban", "Rural"], n, p=[0.33, 0.38, 0.29]),
    })

    # Rule-based approval so the model has a pattern to learn
    score = (
        3.0 * df["Credit_History"]
        + 0.0004 * (df["ApplicantIncome"] + df["CoapplicantIncome"])
        - 0.004 * df["LoanAmount"]
        + 0.4 * (df["Property_Area"] == "Semiurban")
        + 0.2 * (df["Education"] == "Graduate")
        + rng.normal(0, 0.6, n)
    )
    df["Loan_Status"] = np.where(score > score.quantile(0.32), "Y", "N")

    # Add some missing values like the real dataset
    for col in ["Gender", "Married", "Dependents", "Self_Employed", "LoanAmount",
                "Loan_Amount_Term", "Credit_History"]:
        df.loc[rng.random(n) < 0.04, col] = np.nan

    return df


if __name__ == "__main__":
    data = make_sample_data()
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    data.to_csv(os.path.join(BASE_DIR, "data", "loan_data.csv"), index=False)
    print("Saved data/loan_data.csv", data.shape)
