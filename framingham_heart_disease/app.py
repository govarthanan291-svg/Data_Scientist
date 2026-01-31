import streamlit as st
import pickle
import numpy as np
import os

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="❤️ Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction App")
st.write("Logistic Regression using Framingham Heart Disease Dataset")

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "heart_disease_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ----------------------------
# User Inputs
# ----------------------------
st.subheader("🧍 Patient Details")

age = st.number_input("Age", min_value=20, max_value=100, value=40)
sex = st.selectbox("Sex", ["Male", "Female"])
cigsPerDay = st.number_input("Cigarettes Per Day", min_value=0, max_value=70, value=0)
totChol = st.number_input("Total Cholesterol", min_value=100, max_value=400, value=200)
sysBP = st.number_input("Systolic BP", min_value=80, max_value=250, value=120)
diaBP = st.number_input("Diastolic BP", min_value=50, max_value=150, value=80)
BMI = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
heartRate = st.number_input("Heart Rate", min_value=40, max_value=200, value=70)
glucose = st.number_input("Glucose", min_value=50, max_value=300, value=90)

# Convert sex to numeric
sex = 1 if sex == "Male" else 0

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔍 Predict Heart Disease Risk"):
    input_data = np.array([[age, sex, cigsPerDay, totChol,
                             sysBP, diaBP, BMI, heartRate, glucose]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("🩺 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease\n\nRisk Probability: {probability:.2%}")
    else:
        st.success(f"✅ Low Risk of Heart Disease\n\nRisk Probability: {probability:.2%}")
