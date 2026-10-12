"""
Automated tests. They run in CI on every push (see .github/workflows/ci-cd.yml).

They train a small model on SYNTHETIC data, so they pass even when the real
dataset is not in the repository.
"""
import os

import pytest

import loan_model
from generate_sample_data import make_sample_data
from train_model import train


@pytest.fixture(scope="module")
def trained():
    return train(make_sample_data(n=600, seed=0))


def applicant(**changes):
    row = {
        "ApplicantIncome": 5000, "CoapplicantIncome": 0, "LoanAmount": 130,
        "Loan_Amount_Term": 360, "Credit_History": 1.0, "Gender": "Male",
        "Married": "Yes", "Dependents": "0", "Education": "Graduate",
        "Self_Employed": "No", "Property_Area": "Urban",
    }
    row.update(changes)
    return row


def test_input_frame_has_training_columns():
    frame = loan_model.make_input_frame(applicant())
    assert list(frame.columns) == loan_model.FEATURES
    assert len(frame) == 1


def test_model_beats_a_trivial_baseline(trained):
    model, X_test, y_test = trained
    accuracy = (model.predict(X_test) == y_test).mean()
    baseline = max(y_test.mean(), 1 - y_test.mean())
    assert accuracy > baseline


def test_probability_is_between_0_and_1(trained):
    model, _, _ = trained
    proba = model.predict_proba(loan_model.make_input_frame(applicant()))[0][1]
    assert 0.0 <= proba <= 1.0


def test_good_credit_scores_higher_than_bad_credit(trained):
    model, _, _ = trained
    good = model.predict_proba(loan_model.make_input_frame(applicant(Credit_History=1.0)))[0][1]
    bad = model.predict_proba(loan_model.make_input_frame(applicant(Credit_History=0.0)))[0][1]
    assert good > bad


def test_missing_values_and_unknown_category_do_not_crash(trained):
    model, _, _ = trained
    row = applicant(Gender=None, LoanAmount=None, Property_Area="Mars")
    prediction = model.predict(loan_model.make_input_frame(row))
    assert prediction[0] in (0, 1)


@pytest.mark.skipif(
    not os.path.exists(loan_model.MODEL_PATH),
    reason="model/loan_model.sav is not in the repository",
)
def test_saved_model_loads_and_predicts():
    model = loan_model.load_model()
    proba = model.predict_proba(loan_model.make_input_frame(applicant()))[0][1]
    assert 0.0 <= proba <= 1.0
