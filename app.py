import pandas as pd
import streamlit as st
import pickle
import numpy as np

# Load trained model
with open("MODEL.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🩺 Diabetes Prediction App")
st.write("Enter your details below to check if you may have diabetes:")

age = st.number_input("Age", min_value=0, max_value=120, value=30)
tension_bool = st.selectbox("Do you have hypertension?", (True, False))
heart_bool = st.selectbox("Do you have heart disease?", (True, False))
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.5)
hba1c = st.number_input("HbA1c Level", min_value=0.0, max_value=20.0, value=5.5)
glucose = st.number_input("Blood Glucose Level", min_value=0.0, max_value=500.0, value=120.0)

smoke = st.selectbox(
    "Smoking History",
    options=["never", "former", "current", "not current", "ever"]
)

tension = 1 if tension_bool else 0
heart = 1 if heart_bool else 0

smoke_map = {
    "never": 0,
    "former": 1,
    "current": 2,
    "not current": 3,
    "ever": 4
}

smoke_val = smoke_map[smoke]

if st.button("Predict"):
    X_single_test = pd.DataFrame([{
        "age": age,
        "hypertension": tension,
        "heart_disease": heart,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose,
        "smoking_history_num": smoke_val
    }])

    prediction = model.predict(X_single_test)

    if prediction[0] == 1:
        st.error("⚠️ You might have diabetes. Please consult a doctor.")
    else:
        st.success("✅ You don't have diabetes.")