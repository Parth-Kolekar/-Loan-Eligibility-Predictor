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
    applicant_income = st.number_input("Applicant Income", min_value=0)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
    loan_amount_term = st.number_input("Loan Term (in months)", min_value=0)
    credit_history = st.selectbox("Credit History", ["Yes", "No"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    # Encode inputs to match training
    gender = 1 if gender == "Male" else 0
    married = 1 if married == "Yes" else 0
    dependents = {"0": 0, "1": 1, "2": 2, "3+": 3}[dependents]
    education = 0 if education == "Graduate" else 1
    self_employed = 1 if self_employed == "Yes" else 0
    credit_history = 1 if credit_history == "Yes" else 0
    loan_amount_term = loan_amount_term // 30  # Convert days to months
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
