import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Sleep Disorder Assessment", layout="centered")

st.markdown("<h2 style='text-align: center; color: #1BA1F3;'>Sleep Disorder Assessment</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Complete your comprehensive sleep disorder assessment</p>", unsafe_allow_html=True)

# --- Load Model & Scaler ---
try:
    Model = joblib.load('models/final_sleep_disorder_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    le = joblib.load('models/label_encoder.pkl')
except Exception as e:
    st.error(f"Error loading model, scaler, or label encoder: {e}")
    st.stop()

# --- Basic Information ---
st.markdown("### 🧾 Basic Information")
col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", options=["Select gender", "Female", "Male"])
    gender = 1 if gender == "Male" else 0
    
with col2:
    occupation = st.selectbox("Occupation", options=["Doctor", "Engineer", "Lawyer", "Nurse", "Salesperson", "Teacher", "Other"])
    occupation_onehot = [int(occupation == job) for job in ["Doctor", "Engineer", "Lawyer", "Nurse", "Salesperson", "Teacher", "Other"]]
    
with col3:
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    
# BMI Category
bmi_category = st.selectbox("BMI Category", options=["Normal", "Overweight"])
bmi_overweight = 1 if bmi_category == "Overweight" else 0

# --- Sleep & Health Data ---
st.markdown("### 🛌 Sleep & Health Data")
col1, col2 = st.columns(2)

with col1:
    sleep_duration = st.selectbox("Sleep Hours per Night", options=list(range(0, 13)), index=7)
    activity_level = st.number_input("Physical Activity Level (minutes/day)", min_value=0, max_value=100, value=30)

with col2:
    quality_sleep = st.slider("Sleep Quality (1-10)", 1, 10, 7)
    stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)

daily_steps = st.number_input("Daily Steps", min_value=0, value=5000)

# --- Vitals ---
st.markdown("### 🩺 Vitals")
col1, col2, col3 = st.columns(3)

with col1:
    high_bp = st.number_input("Systolic Pressure (mmHg)", min_value=80, max_value=200, value=120)
    
with col2:
    low_bp = st.number_input("Diastolic Pressure (mmHg)", min_value=40, max_value=130, value=80)
    
with col3:
    heart_rate = st.number_input("Heart Rate (BPM)", min_value=40, max_value=200, value=72)

# --- Predict Button ---
if st.button("Complete Health Assessment"):
    try:
        # Combine numerical features
        numeric_values = np.array([
            age, sleep_duration, quality_sleep, activity_level,
            stress_level, heart_rate, high_bp, low_bp, daily_steps
        ])

        scaled_numeric = scaler.transform(numeric_values.reshape(1, -1))
        input_features = np.array(
            [gender] + list(scaled_numeric.flatten()) + occupation_onehot + [bmi_overweight]
        )
        
        #Check for feature mismatch
        if hasattr(Model, 'n_features_in_') and input_features.shape[0] != Model.n_features_in_:
            st.error(f"Feature mismatch: Model expects {Model.n_features_in_} features, but got {input_features.shape[0]}.")
        else:
            prediction = Model.predict(input_features.reshape(1, -1))
            predicted_label = le.inverse_transform(prediction)[0]
            
            st.markdown(f"""
            <div style='margin-top: 30px; background-color: #1BA1F3; padding: 16px; border-radius: 10px; text-align: center;'>
                <h3 style='color: white;'>Predicted Sleep Disorder: {predicted_label}</h3>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")

