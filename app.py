import os
os.system("pip install joblib")

import joblib

import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("salary_model.pkl")

# Streamlit UI
st.title("💼 Salary Prediction App")
st.write("Enter your years of experience to predict the salary.")

# Input field
years_of_experience = st.number_input("Years of Experience", min_value=0.0, max_value=50.0, step=0.1)

# Predict button
if st.button("Predict Salary"):
    prediction = model.predict(np.array([[years_of_experience]]))[0]
    st.success(f"Predicted Salary: ${prediction:.2f}")
