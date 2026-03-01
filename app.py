import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

st.title("👩‍💼 Employee Attrition Prediction")

# -----------------------------
# Load Model + Columns
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR,"model","attrition_model.pkl"),"rb"))
model_columns = pickle.load(open(os.path.join(BASE_DIR,"model","model_columns.pkl"),"rb"))

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Employee Details")

overtime = st.sidebar.selectbox("OverTime", ["Yes","No"])
marital_status = st.sidebar.selectbox(
    "Marital Status",
    ["Single","Married","Divorced"]
)

business_travel = st.sidebar.selectbox(
    "Business Travel",
    ["Travel_Rarely","Travel_Frequently","Non-Travel"]
)

monthly_income = st.sidebar.slider("Monthly Income",1000,30000,8000)
years_at_company = st.sidebar.slider("Years At Company",0,40,5)
job_satisfaction = st.sidebar.slider("Job Satisfaction",1,4,3)

# -----------------------------
# Create Full Feature Vector
# -----------------------------
input_dict = {col:0 for col in model_columns}

# numeric
if "MonthlyIncome" in input_dict:
    input_dict["MonthlyIncome"] = monthly_income

if "YearsAtCompany" in input_dict:
    input_dict["YearsAtCompany"] = years_at_company

if "JobSatisfaction" in input_dict:
    input_dict["JobSatisfaction"] = job_satisfaction

# categorical encoding
input_dict[f"OverTime_Yes"] = 1 if overtime=="Yes" else 0
input_dict[f"MaritalStatus_{marital_status}"] = 1
input_dict[f"BusinessTravel_{business_travel}"] = 1

input_df = pd.DataFrame([input_dict])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Attrition Risk"):

    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Result")

    st.write(f"Attrition Probability: **{probability:.2f}**")

    if probability < 0.4:
        st.success("🟢 Low Risk Employee")
    elif probability < 0.7:
        st.warning("🟡 Medium Risk Employee")
    else:
        st.error("🔴 High Risk Employee")