import streamlit as st
import numpy as np
import pandas as pd
import joblib
from PIL import Image

# ----------------------------
# Configure Streamlit App
# ----------------------------
st.set_page_config(page_title="Student Depression Predictor", page_icon="🧠", layout="wide")

# Load trained model
@st.cache_resource
def load_model():
    return joblib.load("AdaBoost_model.pkl")  # Load your trained model

model = load_model()

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.title("🧠 Student Depression Predictor")
    st.markdown("Predict the likelihood of depression in students.")
    st.markdown("---")
    st.markdown("👨‍💻 Developed by: **Aditi Sahu**")
    st.markdown("🔗 GitHub: [Aditi Bhai](https://github.com/)")
    st.markdown("🔗 LinkedIn: [Profile](https://www.linkedin.com/in/)")
    st.markdown("---")
    st.markdown("✨ **Have fun exploring AI!**")

# ----------------------------
# Header
# ----------------------------
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>Student Depression Prediction 🧠</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #808080;'>Enter the student's lifestyle and habits below</h3>", unsafe_allow_html=True)

# background image
image = Image.open('Artificial Intelligence Application in Mental Health Research copy.jpg')
st.image(image, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("#### Please provide the student details below:")

# ----------------------------
# Input Form using Columns
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    id_val = st.number_input("Student ID (any numeric value)", min_value=0, step=1)
    gender = st.radio("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    city = st.text_input("City")
    profession = st.text_input("Profession")
    academic_pressure = st.slider("Academic Pressure (1-5)", 1.0, 5.0, 3.0)
    study_satisfaction = st.slider("Study Satisfaction (1-5)", 1.0, 5.0, 3.0)
    sleep_duration = st.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])

with col2:
    dietary_habits = st.radio("Dietary Habits", ["Healthy", "Unhealthy"])
    suicidal_thoughts = st.radio("Ever had suicidal thoughts?", ["Yes", "No"])
    work_pressure = st.slider("Work Pressure (1-5)", 0.0, 5.0, 0.0)
    financial_stress = st.slider("Financial Stress (1-5)", 1, 5, 3)
    family_history = st.radio("Family history of mental illness?", ["Yes", "No"])
    job_satisfaction = st.slider("Job Satisfaction (1-5)", 0.0, 5.0, 0.0)
    study_pressure_hours = st.number_input("Work/Study Hours per week", min_value=0, max_value=24, step=1)
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.01)
    degree = st.text_input("Degree")

# ----------------------------
# Mapping categorical to numeric
# ----------------------------
gender = 1 if gender == 'Male' else 0
dietary_habits = 1 if dietary_habits == 'Healthy' else 0
suicidal_thoughts = 1 if suicidal_thoughts == 'Yes' else 0
family_history = 1 if family_history == 'Yes' else 0

sleep_mapping = {
    'Less than 5 hours': 4,
    '5-6 hours': 5.5,
    '7-8 hours': 7.5,
    'More than 8 hours': 9
}
sleep_duration = sleep_mapping.get(sleep_duration, 7.5)

# ----------------------------
# Create Input DataFrame (17 Features)
# ----------------------------
columns = ['id', 'Gender', 'Age', 'City', 'Profession', 'Academic Pressure',
           'Work Pressure', 'CGPA', 'Study Satisfaction', 'Job Satisfaction',
           'Sleep Duration', 'Dietary Habits', 'Degree',
           'Have you ever had suicidal thoughts ?', 'Work/Study Hours',
           'Financial Stress', 'Family History of Mental Illness']

input_df = pd.DataFrame([[id_val, gender, age, city, profession, academic_pressure,
                          work_pressure, cgpa, study_satisfaction, job_satisfaction,
                          sleep_duration, dietary_habits, degree, suicidal_thoughts,
                          study_pressure_hours, financial_stress, family_history]],
                        columns=columns)

# ----------------------------
# Stylish Predict Button
# ----------------------------
button_style = """
<style>
.stButton>button {
    background-color: #2E8B57;
    color: white;
    font-size: 16px;
    height: 3em;
    width: 100%;
    border-radius: 12px;
    transition: background-color 0.3s ease;
}
.stButton>button:hover {
    background-color: #3CB371;
}
</style>
"""
st.markdown(button_style, unsafe_allow_html=True)

# ----------------------------
# Prediction Logic
# ----------------------------
if st.button("Predict"):
    with st.spinner("Predicting..."):
        try:
            prediction_proba = model.predict_proba(input_df)
            depression_prob = prediction_proba[0][1]  # probability of depression

            # Conditional messages based on probability
            if depression_prob < 0.2:
                st.markdown(f"<h3 style='color:green;'>Very unlikely to have depression.</h3>", unsafe_allow_html=True)
            elif 0.2 <= depression_prob < 0.4:
                st.markdown(f"<h3 style='color:green;'>Unlikely to have depression.</h3>", unsafe_allow_html=True)
            elif 0.4 <= depression_prob < 0.6:
                st.markdown(f"<h3 style='color:orange;'>May have depression.</h3>", unsafe_allow_html=True)
            elif 0.6 <= depression_prob < 0.8:
                st.markdown(f"<h3 style='color:orange;'>Likely to have depression.</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color:red;'>Highly likely to have depression.</h3>", unsafe_allow_html=True)

            st.write(f"Depression Probability: {depression_prob*100:.2f}%")

        except Exception as e:
            st.error(f"Error during prediction: {e}")