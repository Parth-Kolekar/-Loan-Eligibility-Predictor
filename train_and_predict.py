# train_and_predict.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# ========== Step 1: Load training data ==========
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Store Loan_IDs for final submission
test_loan_ids = test_df['Loan_ID']

# Drop Loan_ID column
train_df.drop('Loan_ID', axis=1, inplace=True)
test_df.drop('Loan_ID', axis=1, inplace=True)

# ========== Step 2: Handle missing values ==========

def fill_missing(df):
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    return df

train_df = fill_missing(train_df)
test_df = fill_missing(test_df)

# ========== Step 3: Label Encoding ==========

# Combine train + test to encode identically
combined_df = pd.concat([train_df.drop('Loan_Status', axis=1), test_df], axis=0)

# Encode all categorical columns
le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Dependents']:
    combined_df[col] = le.fit_transform(combined_df[col].astype(str))

# Split back
X_train = combined_df[:len(train_df)]
X_test = combined_df[len(train_df):]

# Encode target variable
y_train = LabelEncoder().fit_transform(train_df['Loan_Status'])

# ========== Step 4: Model Training ==========
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate (on part of train)
X_train_sub, X_val, y_train_sub, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
y_pred = model.predict(X_val)
print("📊 Evaluation Report:")
print(classification_report(y_val, y_pred))

# Save model
joblib.dump(model, "loan_model.pkl")

# ========== Step 5: Predict on Test Set ==========
test_preds = model.predict(X_test)
test_preds_label = ['Y' if i == 1 else 'N' for i in test_preds]

submission_df = pd.DataFrame({
    'Loan_ID': test_loan_ids,
    'Loan_Status': test_preds_label
})

submission_df.to_csv("submission.csv", index=False)
print("✅ submission.csv created!")
