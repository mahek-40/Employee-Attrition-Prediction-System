import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

st.title("👩‍💼 Employee Attrition Prediction App")

# -------------------------------
# Load Dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/employee_attrition.csv")
    return df

df = load_data()

# -------------------------------
# Preprocessing
# -------------------------------
df = df.dropna()

df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("Attrition_Yes", axis=1)
y = df_encoded["Attrition_Yes"]

# -------------------------------
# Train Model
# -------------------------------
@st.cache_resource
def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    return model, scaler

model, scaler = train_model(X, y)

# -------------------------------
# Sidebar Inputs
# -------------------------------
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

job_satisfaction = st.sidebar.slider(
    "Job Satisfaction (1-4)",
    1, 4, 3
)

# -------------------------------
# Create Raw Input DataFrame
# -------------------------------
raw_input = pd.DataFrame({
    "OverTime": [overtime],
    "MaritalStatus": [marital_status],
    "BusinessTravel": [business_travel],
    "MonthlyIncome": [monthly_income],
    "YearsAtCompany": [years_at_company],
    "JobSatisfaction": [job_satisfaction]
})

# Encode input same as training
input_encoded = pd.get_dummies(raw_input)

# Align columns with training data
input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

# Scale input
input_scaled = scaler.transform(input_encoded)

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Attrition Risk"):

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    st.write(f"**Probability of Attrition:** {probability:.2f}")

    if probability < 0.4:
        st.success("🟢 Low Risk of Attrition")
    elif probability < 0.7:
        st.warning("🟡 Medium Risk of Attrition")
    else:
        st.error("🔴 High Risk of Attrition")