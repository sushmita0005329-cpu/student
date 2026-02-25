
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------
st.set_page_config(
    page_title="SOTY Predictor",
    page_icon="🏆",
    layout="wide"
)

# ----------------------------------------------------
# BACKGROUND FUNCTION
# ----------------------------------------------------
def set_background():
    # Student classroom background from Unsplash
    image_url = "https://images.unsplash.com/photo-1523240795612-9a054b0db644"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(
                rgba(0,0,0,0.35),
                rgba(0,0,0,0.35)
            ),
            url("{image_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        section.main > div {{
            background-color: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(10px);
            padding: 2rem;
            border-radius: 15px;
        }}

        h1, h2, h3, h4, h5, h6, p, label {{
            color: white !important;
        }}

        .stMetric, .stSlider label {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()

# ----------------------------------------------------
# TITLE
# ----------------------------------------------------
st.markdown("<h1 style='text-align:center;'>🏆 Student of the Year Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

# ----------------------------------------------------
# LOAD DATA
# ----------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("attendance_student3.csv")

try:
    df = load_data()
except:
    st.error("Dataset not found. Make sure attendance_student3.csv exists.")
    st.stop()

# ----------------------------------------------------
# TRAIN MODEL
# ----------------------------------------------------
@st.cache_resource
def train_model(data):
    X = data[['Attendance Percentage', 'Marks', 'Sports', 'Study_Hours']]
    y = data['SOTY']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    return model, accuracy

model, accuracy = train_model(df)

# ----------------------------------------------------
# MODEL PERFORMANCE
# ----------------------------------------------------
st.subheader("📊 Model Performance")
col1, col2 = st.columns(2)

with col1:
    st.metric("Model Accuracy", f"{accuracy*100:.2f}%")

with col2:
    st.metric("Dataset Size", f"{len(df)} Students")

st.markdown("---")

# ----------------------------------------------------
# INPUT SECTION
# ----------------------------------------------------
st.subheader("📝 Enter Student Details")

col1, col2 = st.columns(2)

with col1:
    attendance = st.slider("Attendance Percentage", 0, 100, 75)
    marks = st.slider("Marks", 0, 100, 70)

with col2:
    sports = st.slider("Sports Score", 0, 10, 5)
    study_hours = st.slider("Study Hours per Day", 0, 15, 6)

# ----------------------------------------------------
# PREDICTION
# ----------------------------------------------------
if st.button("🔍 Predict SOTY", use_container_width=True):
    input_data = np.array([[attendance, marks, sports, study_hours]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    if prediction == 1:
        st.success("🏆 YES! This student is likely to be Student of the Year!")
    else:
        st.error("❌ This student is unlikely to be Student of the Year.")

    st.info(f"Confidence Score: {probability*100:.2f}%")

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------
st.markdown("---")
st.markdown(
    "<center>Built with ❤️ using Streamlit</center>",
    unsafe_allow_html=True
)