import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

# ---- Load Dataset ----
@st.cache_data
def load_data():
    df = pd.read_csv("employee_attrition.csv")
    return df

df = load_data()

# ---- Basic Preprocessing ----
df = df.dropna()

df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("Attrition_Yes", axis=1)
y = df_encoded["Attrition_Yes"]

# ---- Train Model ----
@st.cache_resource
def train_model():
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)
    return model, scaler

model, scaler = train_model()

# ---- UI ----
st.title("📊 Employee Attrition Prediction")

st.sidebar.header("Enter Employee Details")

monthly_income = st.sidebar.number_input("Monthly Income", min_value=1000)
years_at_company = st.sidebar.number_input("Years At Company", min_value=0)
job_satisfaction = st.sidebar.slider("Job Satisfaction (1-4)", 1, 4)

# Create input dataframe
input_dict = {
    "MonthlyIncome": monthly_income,
    "YearsAtCompany": years_at_company,
    "JobSatisfaction": job_satisfaction,
}

input_df = pd.DataFrame([input_dict])

# Match columns
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Scale
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠ High Risk of Attrition ({probability:.2f})")
    else:
        st.success(f"✅ Low Risk ({probability:.2f})")