import streamlit as st
import pickle
import pandas as pd

# ---- Page Config ---- #
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide"
)

# ---- Load Model ---- #
import streamlit as st
import pandas as pd
import os
import pickle

# ---- Load Model Safely ---- #

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model", "attrition_model.pkl")
columns_path = os.path.join(BASE_DIR, "model", "model_columns.pkl")

model = pickle.load(open(model_path, "rb"))
model_columns = pickle.load(open(columns_path, "rb"))

# ---- Title ---- #
st.title("📊 Employee Attrition Prediction System")
st.markdown("Predict whether an employee is at risk of leaving the company.")

# ---- Sidebar Input ---- #
st.sidebar.header("📝 Enter Employee Details")

overtime = st.sidebar.selectbox("OverTime", ["Yes", "No"])
marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
department = st.sidebar.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
business_travel = st.sidebar.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])

monthly_income = st.sidebar.number_input("Monthly Income", min_value=1000)
years_at_company = st.sidebar.number_input("Years At Company", min_value=0)
num_companies_worked = st.sidebar.number_input("Number of Companies Worked", min_value=0)
job_satisfaction = st.sidebar.slider("Job Satisfaction (1-4)", 1, 4)

# ---- Prepare Input Data ---- #
input_dict = {
    "MonthlyIncome": monthly_income,
    "YearsAtCompany": years_at_company,
    "NumCompaniesWorked": num_companies_worked,
    "JobSatisfaction": job_satisfaction,
    "OverTime_Yes": 1 if overtime == "Yes" else 0,
    "MaritalStatus_Single": 1 if marital_status == "Single" else 0,
    "Department_Sales": 1 if department == "Sales" else 0,
    "BusinessTravel_Travel_Frequently": 1 if business_travel == "Travel_Frequently" else 0,
}

input_df = pd.DataFrame(columns=model_columns)
input_df.loc[0] = 0

for key in input_dict:
    if key in input_df.columns:
        input_df.at[0, key] = input_dict[key]

# ---- Prediction Section ---- #
st.subheader("🔍 Prediction Result")

if st.button("Predict Attrition Risk"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.write("### 🎯 Attrition Probability")
    st.progress(float(probability))

    if probability > 0.7:
        st.error(f"⚠️ High Risk of Attrition ({probability:.2f})")
    elif probability > 0.4:
        st.warning(f"⚡ Moderate Risk ({probability:.2f})")
    else:
        st.success(f"✅ Low Risk ({probability:.2f})")





# ---- Footer ---- #
st.markdown("---")
st.markdown("Developed by Mahek | AIML Student | Machine Learning Project 🚀")