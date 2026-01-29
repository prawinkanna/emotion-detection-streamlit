import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# --------------------------------------------------
# STREAMLIT PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Emotion Detection", layout="centered")

st.title("😊 Emotion Detection App")
st.caption("Level-4 | CNN + Transfer Learning")

st.warning(
    "⚠️ This application is for educational/demo purposes only. "
    "Predictions may not be 100% accurate."
)

# --------------------------------------------------
# LOAD TRAINED EMOTION MODEL
# --------------------------------------------------
@st.cache_resource
def load_emotion_model():
    # Model expects input shape: (None, 64, 64, 1)
    return load_model("emotion_model.h5", compile=False)

model = load_emotion_model()

# Emotion labels (FER-2013 standard order)
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # ----------------------------------------------
    # LOAD IMAGE
    # ----------------------------------------------
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    # ----------------------------------------------
    # PREPROCESS IMAGE (MATCH MODEL)
    # ----------------------------------------------
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize to 64x64 (MODEL REQUIREMENT)
    face_img = cv2.resize(gray, (64, 64))

    # Normalize pixel values (0–1)
    face_img = face_img / 255.0

    # Reshape to (1, 64, 64, 1)
    face_img = np.reshape(face_img, (1, 64, 64, 1))

    # ----------------------------------------------
    # PREDICTION
    # ----------------------------------------------
    preds = model.predict(face_img)
    emotion_index = np.argmax(preds)
    confidence = np.max(preds) * 100

    # ----------------------------------------------
    # DISPLAY RESULTS
    # ----------------------------------------------
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.success(f"Predicted Emotion: **{emotion_labels[emotion_index]}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

    # Show probability distribution
    st.subheader("Emotion Probability Distribution")
    st.bar_chart(preds[0])
