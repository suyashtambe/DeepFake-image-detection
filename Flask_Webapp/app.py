import streamlit as st
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import base64

model = load_model('model.h5')

def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0  
    return image

def predict_deepfake(image, model):
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    return predictions

st.subheader("Upload Image (JPG only):")
uploaded_file = st.file_uploader("Choose a JPG image...", type="jpg")

if st.button("Predict", key="predict_button"):
    if uploaded_file is not None:
        image = preprocess_image(uploaded_file)
        predictions = predict_deepfake(image, model)
        st.subheader("Prediction:")
        threshold = 0.5
        if predictions[0][0] < threshold:
            st.write("The image is DEEPFAKE.")
        else:
            st.write("The image is REAL.")
    else:
        st.write("Please upload an image before predicting.")
