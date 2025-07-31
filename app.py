import streamlit as st
import joblib
import numpy as np

# Load the model and scaler
Model = joblib.load('models/final_sleep_disorder_model.pkl')
scaler = joblib.load('models/scaler.pkl')
# Load the label encoder
le = joblib.load('models/label_encoder.pkl')

# Gender: 0 for female, 1 for male
gender = st.selectbox("Gender", options=["Female", "Male"])
gender = 1 if gender == "Male" else 0

# Input fields for features
age = st.number_input("Age", min_value=0, max_value=120, value=30)
sleep_duration = st.slider("Sleep Duration (hours)", min_value=0, max_value=24, value=7)
quality_sleep = st.slider("Sleep Quality (1-10)", min_value=1, max_value=10, value=7)
activity_level = st.slider("Activity Level (1-10)", min_value=1, max_value=10, value=5)
stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)
heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=70)
daily_steps = st.number_input("Daily Steps", min_value=0, value=5000)

#High_BP and Low_BP
high_bp = st.number_input("High Blood Pressure (e.g. 120)", min_value=80, max_value=200, value=120)
low_bp = st.number_input("Low Blood Pressure (e.g. 80)", min_value=40, max_value=130, value=80)

#Occupation one-hot encoding
occupation = st.selectbox("Occupation", options=["Doctor", "Engineer","Lawyer","Nurse", "Other", "Salesperson", "Teacher"
])

occupation_doctor = 1 if occupation == "Doctor" else 0
occupation_engineer = 1 if occupation == "Engineer" else 0
occupation_lawyer = 1 if occupation == "Lawyer" else 0
occupation_nurse = 1 if occupation == "Nurse" else 0
occupation_other = 1 if occupation == "Other" else 0
occupation_salesperson = 1 if occupation == "Salesperson" else 0
occupation_teacher = 1 if occupation == "Teacher" else 0

#BMI Category
bmi_category = st.selectbox("BMI Category", options=[ "Normal", "Overweight", "Obese"])
bmi_overweight = 1 if bmi_category == "Overweight" else 0



#collect the input features into a list
input_data = np.array([[
    gender, age, sleep_duration, quality_sleep, activity_level,
    stress_level, heart_rate, daily_steps, high_bp, low_bp,
    occupation_doctor, occupation_engineer, occupation_lawyer, occupation_nurse,
    occupation_other, occupation_salesperson, occupation_teacher,
    bmi_overweight
]])

scaled_input = scaler.transform(input_data)

if st.button("Predict"):
    prediction = Model.predict(scaled_input)
    label_map = ["Insomnia", "None", "Sleep Apnea"]
    predicted_label = label_map[int(prediction[0])]
    st.success(f"Predicted Sleep Disorder: {predicted_label}")