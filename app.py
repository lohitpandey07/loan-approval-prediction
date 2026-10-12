"""
Loan Approval Prediction - Streamlit app
Run:  streamlit run app.py
"""
import os
import pickle

import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "loan_model.sav")
ACC_PATH = os.path.join(BASE_DIR, "model", "accuracy.txt")

st.set_page_config(page_title="Loan Approval Prediction", page_icon="🏦", layout="centered")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def load_accuracy():
    if os.path.exists(ACC_PATH):
        with open(ACC_PATH) as f:
            return float(f.read())
    return None


def main():
    st.title("🏦 Loan Approval Prediction")
    st.caption("For educational purposes only. Not real financial advice.")

    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Run `python train_model.py` first.")
        st.stop()

    model = load_model()
    acc = load_accuracy()
    if acc is not None:
        st.sidebar.metric("Model test accuracy", f"{acc * 100:.1f}%")
    st.sidebar.info(
        "Model: Random Forest\n\nDataset: Loan Prediction (Kaggle)\n\n"
        "Enter the applicant details and click Predict."
    )

    st.subheader("Applicant details")

    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self employed", ["No", "Yes"])
        property_area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"])
    with col2:
        applicant_income = st.number_input("Applicant income (monthly)", min_value=0, value=5000, step=500)
        coapplicant_income = st.number_input("Co-applicant income (monthly)", min_value=0, value=0, step=500)
        loan_amount = st.number_input("Loan amount (in thousands)", min_value=1, value=130, step=10)
        loan_term = st.selectbox("Loan term (months)", [360, 180, 480, 300, 240, 120, 84, 60, 36, 12], index=0)
        credit_history = st.selectbox(
            "Credit history",
            [1.0, 0.0],
            format_func=lambda v: "Good (meets guidelines)" if v == 1.0 else "Bad / no history",
        )

    if st.button("Predict", type="primary"):
        input_df = pd.DataFrame([{
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "Property_Area": property_area,
        }])

        try:
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0][1]

            if prediction == 1:
                st.success("✅ Loan is likely to be APPROVED")
            else:
                st.error("❌ Loan is likely to be REJECTED")

            st.write(f"Approval probability: **{proba * 100:.1f}%**")
            st.progress(float(proba))
        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
