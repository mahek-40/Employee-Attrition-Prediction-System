import streamlit as st
import pandas as pd
import pickle
import os

# ---------------- PAGE ----------------
st.set_page_config(page_title="Employee Attrition Predictor",
                   layout="wide")

st.title("👩‍💼 Employee Attrition Prediction System")

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR,
                                      "model/attrition_model.pkl"), "rb"))

columns = pickle.load(open(os.path.join(BASE_DIR,
                                        "model/model_columns.pkl"), "rb"))

scaler = pickle.load(open(os.path.join(BASE_DIR,
                                       "model/scaler.pkl"), "rb"))

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("Enter Employee Details")

age = st.sidebar.slider("Age", 18, 60, 30)

income = st.sidebar.slider(
    "Monthly Income", 1000, 30000, 8000)

years = st.sidebar.slider(
    "Years At Company", 0, 40, 5)

total_years = st.sidebar.slider(
    "Total Working Years", 0, 40, 8)

job_satisfaction = st.sidebar.slider(
    "Job Satisfaction", 1, 4, 3)

overtime = st.sidebar.selectbox(
    "OverTime", ["Yes", "No"])

marital = st.sidebar.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"])

travel = st.sidebar.selectbox(
    "Business Travel",
    ["Travel_Rarely",
     "Travel_Frequently",
     "Non-Travel"])

# ---------------- CREATE INPUT ----------------
input_dict = {
    "Age": age,
    "MonthlyIncome": income,
    "YearsAtCompany": years,
    "TotalWorkingYears": total_years,
    "JobSatisfaction": job_satisfaction,

    # Encoding
    "OverTime_Yes": 1 if overtime == "Yes" else 0,
    "MaritalStatus_Married": 1 if marital == "Married" else 0,
    "MaritalStatus_Single": 1 if marital == "Single" else 0,
    "BusinessTravel_Travel_Frequently":
        1 if travel == "Travel_Frequently" else 0,
    "BusinessTravel_Travel_Rarely":
        1 if travel == "Travel_Rarely" else 0
}

input_df = pd.DataFrame([input_dict])

# MATCH TRAINING COLUMNS
input_df = input_df.reindex(columns=columns,
                            fill_value=0)

# SCALE INPUT
input_scaled = scaler.transform(input_df)

# ---------------- PREDICTION ----------------
if st.button("Predict Attrition"):

    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    # ✅ 3 LEVEL OUTPUT
    if prob < 0.35:
        st.success(f"✅ Low Risk of Attrition ({prob:.2f})")

    elif prob < 0.65:
        st.warning(f"⚠ Medium Risk of Attrition ({prob:.2f})")

    else:
        st.error(f"🚨 High Risk of Attrition ({prob:.2f})")