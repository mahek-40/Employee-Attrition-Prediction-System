import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")
st.title("👩‍💼 Employee Attrition Prediction App")

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/employee_attrition.csv")
    return df

df = load_data()
df = df.dropna()

# -----------------------------
# Encode Data
# -----------------------------
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("Attrition_Yes", axis=1)
y = df_encoded["Attrition_Yes"]

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Train Model (Balanced)
# -----------------------------
@st.cache_resource
def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"   # 🔥 IMPORTANT FIX
    )
    model.fit(X_train_scaled, y_train)

    return model, scaler

model, scaler = train_model(X_train, y_train)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Enter Employee Details")

overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
business_travel = st.sidebar.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])

monthly_income = st.sidebar.number_input(
    "Monthly Income",
    min_value=int(df["MonthlyIncome"].min()),
    max_value=int(df["MonthlyIncome"].max()),
    value=int(df["MonthlyIncome"].mean())
)

years_at_company = st.sidebar.number_input(
    "Years At Company",
    min_value=0,
    max_value=int(df["YearsAtCompany"].max()),
    value=3
)

job_satisfaction = st.sidebar.slider("Job Satisfaction (1-4)", 1, 4, 3)

# -----------------------------
# Create Input
# -----------------------------
raw_input = pd.DataFrame({
    "OverTime": [overtime],
    "MaritalStatus": [marital_status],
    "BusinessTravel": [business_travel],
    "MonthlyIncome": [monthly_income],
    "YearsAtCompany": [years_at_company],
    "JobSatisfaction": [job_satisfaction]
})

input_encoded = pd.get_dummies(raw_input)
input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

input_scaled = scaler.transform(input_encoded)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Attrition Risk"):

    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")
    st.write(f"Probability of Attrition: {probability:.2f}")

    if probability < 0.4:
        st.success("🟢 Low Risk of Attrition")
    elif probability < 0.7:
        st.warning("🟡 Medium Risk of Attrition")
    else:
        st.error("🔴 High Risk of Attrition")