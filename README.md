# 📊 Employee Attrition Prediction System

A Machine Learning web application that predicts whether an employee is at risk of leaving the company.

This project uses classification algorithms to analyze employee data and estimate attrition probability.

---

## 🚀 Live Demo
(After deployment, paste your Streamlit link here)

---

## 🎯 Project Objective

The goal of this project is to build a predictive model that helps organizations identify employees who are likely to leave. This helps HR teams take proactive measures to reduce attrition and improve retention.

---

## 🧠 Machine Learning Approach

- Data Cleaning & Preprocessing
- Handling Missing Values
- Encoding Categorical Variables
- Feature Selection
- Model Training (Logistic Regression / Random Forest)
- Model Evaluation (Accuracy, Confusion Matrix, ROC-AUC)
- Deployment using Streamlit

---

## 📂 Project Structure
Employee-Attrition-Prediction/
│
├── app.py # Streamlit Web Application
├── requirements.txt # Required Libraries
├── README.md # Project Documentation
│
├── model/
│ ├── attrition_model.pkl
│ └── model_columns.pkl
│
└── src/ # Source Code (Training Scripts)
├── data_preprocessing.py
├── model_training.py
└── evaluation.py

---

## 📊 Features of Web App

- User-friendly sidebar input
- Attrition probability display
- Risk level classification (Low / Moderate / High)
- Clean UI design
- Real-time predictions

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

---

## 📈 Model Output

The model predicts:

- 0 → Employee Will Stay
- 1 → Employee Likely to Leave

It also shows probability score for better risk understanding.

---

## 🚀 How to Run Locally

```bash
git clone <your-repo-link>
cd Employee-Attrition-Prediction
pip install -r requirements.txt
python -m streamlit run app.py