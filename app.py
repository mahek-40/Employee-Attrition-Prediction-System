import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

# -----------------------------
# Load Dataset Safely
# -----------------------------
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(BASE_DIR, "data", "employee_attrition.csv")
    df = pd.read_csv(file_path)
    return df

df = load_data()

# -----------------------------
# Preprocessing
# -----------------------------
df = df.dropna()

# Convert target column properly
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})

# Separate target before encoding
y = df["Attrition"]
X = df.drop("Attrition", axis=1)

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)

# -----------------------------
# Train Model
# -----------------------------
@st.cache_resource
def train_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    return model, scaler

model, scaler = train_model(X, y)

# -----------------------------
# UI Layout
# -----------------------------
st.title("📊 Employee Attrition Prediction System")
st.markdown("Predict whether an employee is at risk of leaving the company.")

st.sidebar.header("📝 Enter Employee Details")

# --- Categorical Inputs ---
overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
business_travel = st.sidebar.selectbox("Business Travel", 
                                       ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
department = st.sidebar.selectbox("Department",
                                  ["Sales", "Research & Development", "Human Resources"])

# --- Numerical Inputs ---
monthly_income = st.sidebar.number_input("Monthly Income", min_value=1000)
years_at_company = st.sidebar.number_input("Years At Company", min_value=0)
job_satisfaction = st.sidebar.slider("Job Satisfaction (1-4)", 1, 4)
num_companies = st.sidebar.number_input("Number of Companies Worked", min_value=0)

# -----------------------------
# Create Input Dictionary
# -----------------------------
input_dict = {
    "MonthlyIncome": monthly_income,
    "YearsAtCompany": years_at_company,
    "JobSatisfaction": job_satisfaction,
    "NumCompaniesWorked": num_companies,
}

# Encode categorical manually (must match training encoding)

if overtime == "Yes":
    input_dict["OverTime_Yes"] = 1

if marital_status == "Single":
    input_dict["MaritalStatus_Single"] = 1
elif marital_status == "Married":
    input_dict["MaritalStatus_Married"] = 1

if business_travel == "Travel_Frequently":
    input_dict["BusinessTravel_Travel_Frequently"] = 1
elif business_travel == "Travel_Rarely":
    input_dict["BusinessTravel_Travel_Rarely"] = 1

if department == "Sales":
    input_dict["Department_Sales"] = 1
elif department == "Research & Development":
    input_dict["Department_Research & Development"] = 1

# -----------------------------
# Prepare Final Input
# -----------------------------
input_df = pd.DataFrame([input_dict])

# Match training columns
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Scale input
input_scaled = scaler.transform(input_df)

# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("🔍 Prediction Result")

if st.button("Predict Attrition Risk"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.write("### 🎯 Attrition Probability")
    st.progress(float(probability))

    if probability > 0.7:
        st.error(f"⚠ High Risk of Attrition ({probability:.2f})")
    elif probability > 0.4:
        st.warning(f"⚡ Moderate Risk ({probability:.2f})")
    else:
        st.success(f"✅ Low Risk ({probability:.2f})")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("Developed by Mahek | AIML Student 🚀")