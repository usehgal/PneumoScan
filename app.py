import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os

# Load model
MODEL_PATH = "inception_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess image
def preprocess_image(img):
    img = cv2.imdecode(np.frombuffer(img.read(), np.uint8), 1)
    img = cv2.resize(img, (299, 299))  # InceptionV3 input size
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Streamlit UI
st.title("🩺 PneumoScan - Pneumonia Detection")

uploaded_file = st.file_uploader("Upload a Chest X-Ray", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded X-Ray", use_container_width=True)

    # Preprocess and predict
    img = preprocess_image(uploaded_file)
    prediction = model.predict(img)[0][0]  # Assuming binary classification (0 = Normal, 1 = Pneumonia)

    if prediction > 0.5:
        st.error("🔴 **Pneumonia Detected** (Confidence: {:.2f}%)".format(prediction * 100))
    else:
        st.success("🟢 **Normal Lungs** (Confidence: {:.2f}%)".format((1 - prediction) * 100))
