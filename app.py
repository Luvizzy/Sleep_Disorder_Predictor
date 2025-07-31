import streamlit as st
import joblib
import numpy as np

# Load the model and scaler
Model = joblib.load('models/final_sleep_disorder_model.pkl')
scaler = joblib.load('models/scaler.pkl')
# Load the label encoder
le = joblib.load('models/label_encoder.pkl')

# Input fields for features
age = st.number_input("Age", min_value=0, max_value=120, value=30)
sleep_duration = st.slider("Sleep Duration (hours)", min_value=0, max_value=24, value=7)
quality_sleep = st.slider("Sleep Quality (1-10)", min_value=1, max_value=10, value=7)
stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)
activity_level = st.slider("Activity Level (1-10)", min_value=1, max_value=10, value=5)
heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=70)
daily_steps = st.number_input("Daily Steps", min_value=0, value=5000)


#collect the input features into a list
input_data = np.array([[age, sleep_duration, quality_sleep, stress_level, activity_level, heart_rate, daily_steps]])
scaled_input = scaler.transform(input_data)

if st.button("Predict"):
    prediction = Model.predict(scaled_input)
    label_map = ["Insomnia", "None", "Sleep Apnea"]
    predicted_label = label_map[int(prediction[0])]
    st.success(f"Predicted Sleep Disorder: {predicted_label}")