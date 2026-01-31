import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("Logistic Regression using Framingham Heart Disease Dataset")

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("framingham_heart_disease.csv")
    df = df.dropna()
    return df

df = load_data()

# -------------------------------------------------
# Train Logistic Regression Model
# -------------------------------------------------
@st.cache_resource
def train_model(data):
    X = data[['age', 'male', 'cigsPerDay', 'sysBP', 'BMI']]
    y = data['TenYearCHD']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Save model as PKL
    with open("heart_disease_model.pkl", "wb") as f:
        pickle.dump(model, f)

    return model, accuracy

model, accuracy = train_model(df)

# -------------------------------------------------
# Show Accuracy
# -------------------------------------------------
st.success(f"Model Accuracy: {accuracy * 100:.2f}%")

# -------------------------------------------------
# User Input Section
# -------------------------------------------------
st.header("🔍 Enter Patient Details")

age = st.number_input("Age", min_value=1, max_value=100, value=45)

sex = st.selectbox("Sex", ["Male", "Female"])
male = 1 if sex == "Male" else 0

cigs = st.number_input("Cigarettes Per Day", min_value=0, max_value=100, value=0)

sysBP = st.number_input(
    "Systolic Blood Pressure",
    min_value=80,
    max_value=250,
    value=120
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=50.0,
    value=25.0
)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("Predict Heart Disease"):
    input_data = [[age, male, cigs, sysBP, bmi]]
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("⚠️ High Risk: Person may develop Heart Disease")
    else:
        st.success("✅ Low Risk: Person may not develop Heart Disease")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Developed by Gova | Logistic Regression | Streamlit")