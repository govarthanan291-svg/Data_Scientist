import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="KNN Prediction App")

st.title("🔍 KNN Prediction App")

# Load model
with open("knn_model.pkl", "rb") as f:
    model, scaler, columns = pickle.load(f)

st.subheader("Enter Input Values")

user_input = {}

for col in columns:
    user_input[col] = st.number_input(f"{col}", value=0.0)

input_df = pd.DataFrame([user_input])

# Scale input
input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    prediction = model.predict(input_scaled)
    st.success(f"✅ Prediction Result: {prediction[0]}")