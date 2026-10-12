"""
Shared model code, used by train_model.py, app.py and the tests.

Keeping the feature lists and the pipeline in one place guarantees that
training and the web app always agree on the column names.
"""
import os
import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "loan_model.sav")
ACCURACY_PATH = os.path.join(BASE_DIR, "model", "accuracy.txt")

NUMERIC = [
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History",
]
CATEGORICAL = [
    "Gender", "Married", "Dependents", "Education",
    "Self_Employed", "Property_Area",
]
FEATURES = NUMERIC + CATEGORICAL


def build_pipeline():
    """Imputation + one-hot encoding + Random Forest, as one object."""
    preprocess = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=5,
            random_state=42, class_weight="balanced",
        )),
    ])


def make_input_frame(row):
    """Turn one applicant (a dict) into a one-row DataFrame in training order."""
    return pd.DataFrame([row])[FEATURES]


def load_model(path=MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)
