# 🏦 Loan Approval Prediction

A machine learning web app built with **Streamlit** that predicts whether a loan
application is likely to be approved, based on applicant details.

> ⚠️ For educational purposes only. Not real financial advice.

**Live demo:** loan-approval-prediction-9krvqxvcsgt5hpaqw4kcbn.streamlit.app

## Features

- Random Forest model with a full scikit-learn pipeline (missing-value handling and encoding are built in)
- Clean form with dropdowns and number inputs
- Shows the approval probability, not just a yes/no
- Displays model accuracy in the sidebar

## Project structure

```
loan-approval-prediction/
├── app.py                   # Streamlit web app
├── train_model.py           # Trains the model and creates model/loan_model.sav
├── generate_sample_data.py  # Optional: creates SYNTHETIC test data
├── requirements.txt
├── data/
│   └── loan_data.csv        # You add this (see below)
└── model/
    ├── loan_model.sav       # Created by train_model.py
    └── accuracy.txt         # Created by train_model.py
```

## Dataset

Download the **Loan Prediction** dataset from Kaggle (search "Loan Prediction Problem Dataset"
and use `train.csv`). Save it as `data/loan_data.csv`.

It must have these columns: `Loan_ID, Gender, Married, Dependents, Education,
Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
Credit_History, Property_Area, Loan_Status`.

No dataset handy? Run `python generate_sample_data.py` to create synthetic data just to
test that the project works. Results from synthetic data are not meaningful.

## Run locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (creates model/loan_model.sav)
python train_model.py

# 3. Start the app
streamlit run app.py
```

## How it works

1. `train_model.py` loads the CSV, splits it 80/20, and trains a pipeline
   (median/mode imputation, one-hot encoding, Random Forest).
2. The whole pipeline is saved with `pickle` as a `.sav` file.
3. `app.py` loads the pipeline, builds a DataFrame from the form inputs, and calls
   `predict()` and `predict_proba()`.

## Tech stack

Python, pandas, NumPy, scikit-learn, Streamlit

## Possible improvements

- Compare more models (XGBoost, Logistic Regression) and pick the best
- Add feature importance charts
- Add input validation rules (for example income vs loan amount)

## License

MIT
