
# 🏦 Loan Eligibility Predictor

This project is a Machine Learning-based Loan Eligibility Predictor that helps determine whether a loan applicant is likely to be approved for a loan or not.

## 🔍 Overview

Using historical loan data, this model learns the relationship between applicant details (like income, credit history, etc.) and loan approval decisions. The system can be used to:
- Predict loan approval for new applicants
- Help banks/lenders automate initial loan screening
- Provide insights into what factors affect eligibility

## 🧠 Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Streamlit (for UI)
- Jupyter Notebook (for EDA)
- Matplotlib / Seaborn (optional for visualization)

## 📁 Project Structure

```
loan_eligibility_predictor/
│
├── data/
│   ├── train.csv
│   └── test.csv
├── train_and_predict.py      # Training & prediction script
├── loan_model.pkl            # Saved ML model
├── submission.csv            # Predicted results on test set
├── app.py                    # Streamlit web app
├── requirements.txt
└── README.md
```

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/loan-eligibility-predictor.git
cd loan-eligibility-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model and generate predictions
```bash
python train_and_predict.py
```

### 4. Launch the web app
```bash
streamlit run app.py
```

### 🔗 Live Demo

You can try the deployed Loan Eligibility Predictor app here:  
👉 [Click to open the app] (https://parth-kolekar-loan-eligibility.streamlit.app/)

## 🧾 Features
- Cleaned and preprocessed loan data
- Trained ML model (Logistic Regression / Random Forest)
- Interactive web app for predicting loan eligibility
- Submission-ready CSV output for evaluation

## 📦 Future Enhancements
- Add model explainability (e.g., SHAP or feature importance)
- Deploy the app online (Render/Hugging Face)
- Convert to `.exe` using PyInstaller

## 🙌 Author
**Parth Kolekar**  
IT Engineering Student  
Pune University
