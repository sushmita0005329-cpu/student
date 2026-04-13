
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# ---------- Background + UI Styling ----------
def set_bg():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(rgba(5,10,20,0.85), rgba(5,10,20,0.95)),
            url("https://images.unsplash.com/photo-1559757175-5700dde675bc");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: white;
        }

        /* Glass container */
        .block-container {
            background: rgba(0, 0, 0, 0.6);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0px 0px 20px rgba(0, 255, 255, 0.3);
        }

        /* Title Glow */
        .title {
            text-align: center;
            font-size: 40px;
            font-weight: bold;
            color: white;
            text-shadow: 0px 0px 10px cyan, 0px 0px 20px cyan;
        }

        /* Text */
        h1, h2, h3, h4, h5, h6, p, label {
            color: white !important;
        }

        /* Upload box */
        .stFileUploader {
            background: rgba(0,0,0,0.5);
            padding: 10px;
            border-radius: 10px;
        }

        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

# ---------- Load Model ----------
model = load_model("brain_tumor_model.h5")

IMG_SIZE = 128

# ---------- Title ----------
st.markdown("<div class='title'>🧠 AI Brain Tumor Detection</div>", unsafe_allow_html=True)

st.write("Upload an MRI scan image to detect presence of tumor")

# ---------- Upload ----------
uploaded_file = st.file_uploader("Choose MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    # Preprocess
    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.reshape(img, (1, IMG_SIZE, IMG_SIZE, 3))

    # Prediction with loader
    with st.spinner("🔍 Analyzing Brain Scan..."):
        prediction = model.predict(img)[0][0]

    st.subheader("🧾 Result:")

    if prediction > 0.5:
        st.error("🧠 Tumor Detected")
        st.markdown(f"**Confidence:** {prediction:.2f}")
    else:
        st.success("✅ No Tumor Detected")
        st.markdown(f"**Confidence:** {1 - prediction:.2f}")

# ---------- Sidebar ----------
st.sidebar.markdown("## 🧠 About")
st.sidebar.info(
    "This AI model analyzes MRI images to detect brain tumors.\n\nBuilt using Deep Learning & Streamlit."
)