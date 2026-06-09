


# Project Logistic Regression Titanic Survival Prediction

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return pickle.load(open("mode_titanic.pkl", "rb"))

try:
    model = load_model()
except FileNotFoundError:
    st.error("❌ Model file not found! Please ensure 'mode_titanic.pkl' exists.")
    st.stop()

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
<style>

/* Main Background */
.stApp{
    background: linear-gradient(135deg,#0f172a,#1e293b,#334155);
    color:white;
}

/* Header Card */
.header-card{
    background: linear-gradient(135deg,#2563eb,#06b6d4);
    padding:25px;
    border-radius:20px;
    text-align:center;
    color:white;
    box-shadow:0px 4px 20px rgba(0,0,0,0.3);
    margin-bottom:20px;
}

.header-card h1{
    margin:0;
    font-size:38px;
}

.header-card p{
    margin-top:8px;
    font-size:18px;
}



/* Button */
.stButton > button{
    width:100%;
    height:55px;
    border:none;
    border-radius:12px;
    background:linear-gradient(135deg,#2563eb,#06b6d4);
    color:white;
    font-size:18px;
    font-weight:bold;
}

.stButton > button:hover{
    transform:scale(1.02);
    transition:0.3s;
}

/* Success Card */
.success-card{
    background:#d1fae5;
    color:#065f46;
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
    border-left:8px solid #10b981;
}

/* Error Card */
.error-card{
    background:#fee2e2;
    color:red;
    padding:20px;
    border-radius:15px;
    text-align:center;
    font-size:24px;
    font-weight:bold;
    border-left:8px solid #ef4444;
}

/* Footer */
.footer{
    text-align:center;
    color:white;
    font-size:16px;
    margin-top:20px;
    padding:15px;
}

.footer a{
    color:#06b6d4;
    text-decoration:none;
    font-weight:bold;
}

.footer a:hover{
    color:#38bdf8;
    text-decoration:underline;
}
            

/* Make all labels white */
.stNumberInput label,
.stSelectbox label,
.stTextInput label,
.stTextArea label,
.stDateInput label,
.stTimeInput label,
.stMultiSelect label {
color: white !important;
font-weight: 600 !important;
font-size: 16px !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Header Card
# -------------------------------
st.markdown("""
<div class="header-card">
    <h1>🚢 Titanic Survival Prediction</h1>
    <p>Predict whether a passenger survived the Titanic disaster</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Form Section
# -------------------------------
st.markdown(
    '<p class="bold-text">📝 Enter Passenger Details</p>',
    unsafe_allow_html=True
)

with st.container():

    st.markdown('<div class="form-card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        Pclass = st.number_input(
            "Passenger Class (Pclass)",
            min_value=1,
            max_value=3,
            value=1
        )

        Sex = st.selectbox(
            "Gender",
            options=[0, 1],
            format_func=lambda x: "Female" if x == 0 else "Male"
        )

        Age = st.number_input(
            "Age",
            min_value=0,
            max_value=100,
            value=30
        )

        SibSp = st.number_input(
            "Siblings / Spouse",
            min_value=0,
            max_value=10,
            value=0
        )

    with col2:
        Parch = st.number_input(
            "Parents / Children",
            min_value=0,
            max_value=10,
            value=0
        )

        Fare = st.number_input(
            "Fare",
            min_value=0.0,
            max_value=600.0,
            value=50.0
        )

        Embarked = st.selectbox(
            "Embarked",
            options=[0, 1, 2],
            format_func=lambda x:
            ["Cherbourg", "Queenstown", "Southampton"][x]
        )

    st.markdown("<br>", unsafe_allow_html=True)

    predict_btn = st.button("🔍 Predict Survival")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Prediction
# -------------------------------
if predict_btn:

    input_data = np.array([
        [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
    ])

    prediction = model.predict(input_data)

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction[0] == 1:

        st.markdown("""
        <div class="success-card">
            ✅ Passenger Likely Survived
        </div>
        """, unsafe_allow_html=True)

        st.success(
            "Based on the passenger details provided, survival is predicted."
        )

    else:

        st.markdown("""
        <div class="error-card">
            ❌ Passenger Likely Did Not Survive
        </div>
        """, unsafe_allow_html=True)

        st.warning(
            "Based on the passenger details provided, survival is not predicted."
        )

# -------------------------------
# -------------------------------
# Footer
# -------------------------------
st.markdown("<hr>", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    🚢 Developed by <b>Yanaguntikar Meesal</b><br><br>

📧 Yanaguntikarm@gmail.com
</div>
""", unsafe_allow_html=True)



