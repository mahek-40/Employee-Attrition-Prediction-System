import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Employee Attrition Predictor")

st.title("👩‍💼 Employee Attrition Prediction")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# LOAD EVERYTHING
model = pickle.load(open(os.path.join(BASE_DIR,"model","attrition_model.pkl"),"rb"))
model_columns = pickle.load(open(os.path.join(BASE_DIR,"model","model_columns.pkl"),"rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR,"model","scaler.pkl"),"rb"))

# ---------------- INPUTS ----------------
st.sidebar.header("Employee Details")

overtime = st.sidebar.selectbox("OverTime",["Yes","No"])
marital = st.sidebar.selectbox("Marital Status",
                               ["Single","Married","Divorced"])
travel = st.sidebar.selectbox("Business Travel",
                              ["Travel_Rarely",
                               "Travel_Frequently",
                               "Non-Travel"])

income = st.sidebar.slider("Monthly Income",1000,30000,8000)
years = st.sidebar.slider("Years At Company",0,40,5)
satisfaction = st.sidebar.slider("Job Satisfaction",1,4,3)

# ---------------- FEATURE VECTOR ----------------
input_dict = {col:0 for col in model_columns}

input_dict["MonthlyIncome"] = income
input_dict["YearsAtCompany"] = years
input_dict["JobSatisfaction"] = satisfaction

input_dict["OverTime_Yes"] = 1 if overtime=="Yes" else 0
input_dict[f"MaritalStatus_{marital}"] = 1
input_dict[f"BusinessTravel_{travel}"] = 1

input_df = pd.DataFrame([input_dict])

# ✅ SCALE INPUT (CRITICAL FIX)
input_scaled = scaler.transform(input_df)

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    prob = model.predict_proba(input_scaled)[0][1]

    st.write(f"Attrition Probability: {prob:.2f}")

    if prob < 0.4:
        st.success("🟢 Low Risk")
    elif prob < 0.7:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")