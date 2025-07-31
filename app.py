# app.py
import streamlit as st
import numpy as np
import joblib

model = joblib.load('loan_model.pkl')

st.title("🏦 Loan Eligibility Predictor")

def user_input():
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    
    applicant_income = st.text_input("Applicant Income", placeholder="e.g. 5000")
    coapplicant_income = st.text_input("Coapplicant Income", placeholder="e.g. 1500")
    loan_amount = st.text_input("Loan Amount (in thousands)", placeholder="e.g. 100")
    loan_amount_term = st.text_input("Loan Term (in months)", placeholder="e.g. 12")
    
    credit_history = st.selectbox("Credit History", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    # Ensure all numeric fields are filled
    if not all([applicant_income, coapplicant_income, loan_amount, loan_amount_term]):
        st.warning("Please fill in all income and loan fields to continue.")
        return None

    try:
        applicant_income = float(applicant_income)
        coapplicant_income = float(coapplicant_income)
        loan_amount = float(loan_amount)
        loan_amount_term = float(loan_amount_term)
    except ValueError:
        st.error("Please enter valid numbers in the income and loan fields.")
        return None

    # Encode categorical inputs
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    dependents = {"0": 0, "1": 1, "2": 2, "3+": 3}[dependents]
    education = 0 if education == "Graduate" else 1
    self_employed = 1 if self_employed == "Yes" else 0
    credit_history = 1 if credit_history == "Yes" else 0
    property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

    data = np.array([[gender, married, dependents, education, self_employed,
                      applicant_income, coapplicant_income, loan_amount,
                      loan_amount_term, credit_history, property_area]])

    return data


input_data = user_input()

if st.button("Predict"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")
