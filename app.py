import streamlit as st
import joblib
import numpy as np
from PIL import Image
import os

# Get the absolute path to the assets folder
BASE_DIR = os.path.dirname(__file__)
ASSETS_DIR = os.path.join(BASE_DIR, "assets")


# Style configuration
st.set_page_config(page_title="Sleep Disorder Prediction", layout="centered")
st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background-color: #0d1117;
        color: #f0f6fc;
    }
    .stSlider > div > div {
        color: #ffffff;
    }
    .stButton > button {
        background-color: #238636;
        color: white;
        border-radius: 8px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Logo & Header
logo_path = os.path.join(ASSETS_DIR, "logo.png")
logo = Image.open(logo_path)
st.image(logo, width=150)

# --- Load Model & Scaler ---
try:
    Model = joblib.load('models/final_sleep_disorder_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# Input fields for user information
st.subheader("🧑‍⚕️ Information")
# Gender: 0 for female, 1 for male
gender = 1 if st.selectbox("Gender", options=["Female", "Male"]) == "Male" else 0
age = st.number_input("Age", min_value=0, max_value=120, value=30)

st.subheader("🛌 Sleep & Health Data")
sleep_duration = st.slider("Sleep Duration (hours)", min_value=0, max_value=24, value=7)
quality_sleep = st.slider("Sleep Quality (1-10)", min_value=1, max_value=10, value=7)
activity_level = st.slider(
    "Physical Activity Level (minutes/day)",
    min_value=0, max_value=100, value=30,
    help="Estimated number of minutes spent on physical activity daily."
)
stress_level = st.slider("Stress Level (1-10)", min_value=1, max_value=10, value=5)
heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=70)
daily_steps = st.number_input("Daily Steps", min_value=0, value=5000)

st.subheader("🩺 Vitals")
high_bp = st.number_input("High Blood Pressure (e.g. 120)", min_value=80, max_value=200, value=120)
low_bp = st.number_input("Low Blood Pressure (e.g. 80)", min_value=40, max_value=130, value=80)

st.subheader("💼 Demographic Information")
occupation = st.selectbox("Occupation", options=["Doctor", "Engineer","Lawyer","Nurse", "Other", "Salesperson", "Teacher"
])
occupation_onehot = [int(occupation == job) for job in ["Doctor", "Engineer", "Lawyer", "Nurse", "Other", "Salesperson", "Teacher"]]

#BMI Category
bmi_category = st.selectbox("BMI Category", options=[ "Normal", "Overweight"])
bmi_overweight = 1 if bmi_category == "Overweight" else 0

#Apply scaler to only the numerical features
numeric_values = np.array([
    age, sleep_duration, quality_sleep, activity_level,
    stress_level, heart_rate, daily_steps, high_bp, low_bp
])

# Scale the numeric values
scaled_numeric = scaler.transform(numeric_values.reshape(1, -1))

# Collect the input features into a list
# Make sure the order and number of features matches what the model expects
# Typically: [gender, age, sleep_duration, quality_sleep, activity_level, stress_level, heart_rate, daily_steps, high_bp, low_bp, occupation_onehot..., bmi_overweight]
input_features = np.array(
    [gender] + list(scaled_numeric.flatten()) + occupation_onehot + [bmi_overweight]
)

if st.button("Predict"):
    if hasattr(Model, 'n_features_in_') and input_features.shape[0] != Model.n_features_in_:
        st.error(f"Feature mismatch: Model expects {Model.n_features_in_} features, but got {input_features.shape[0]}.")
    else:
        prediction = Model.predict(input_features.reshape(1, -1))
        label_map = ["Insomnia", "None", "Sleep Apnea"]
        try:
            predicted_label = label_map[int(prediction[0])]
        except (IndexError, ValueError):
            predicted_label = "Unknown"
        st.markdown(f"""
            <div style='background-color: #238636; padding: 12px; border-radius: 8px; text-align: center;'>
                <h3 style='color: white;'>Predicted Sleep Disorder: {predicted_label}</h3>
            </div>
        """, unsafe_allow_html=True)
        
# Gif
gif_path = os.path.join(ASSETS_DIR, "penguin.gif")
st.image("assets/penguin.gif", use_container_width=False, width=300)



