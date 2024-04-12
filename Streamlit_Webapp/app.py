import streamlit as st
from tensorflow.keras.models import load_model
from streamlit_lottie import st_lottie
import cv2
import numpy as np
import base64

# Load the trained model
model = load_model(r'deepfake_autoencoder.keras')

# Function to get base64 encoding of a binary file
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set background image using base64 encoding
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
    background-image: url("data:image/png;base64,{bin_str}");
    background-size: cover;
    backdrop-filter: blur(1000px);
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the background image
set_background(r'Flask_Webapp\Templates\Images\bg_image.jpg')

# Main title and description
st.markdown("<h1 style='text-align: center; color: white; font-family: Arial, sans-serif;'>Deep Fake Image Detection using Multidisciplinary Approach</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-family: Arial, sans-serif; color: white; font-size: 18px; text-align: center;'>Collecting diverse dataset. Preprocess images. Extract attributes with CNNs. Split dataset. Select, train model. Monitor performance. Test for generalization.</p>", unsafe_allow_html=True)

# Function to load and display Lottie animation
def load_lottie_animation(json_file, width=None, height=None):
    with open(json_file, "r") as f:
        lottie_json = f.read()
    st_lottie(lottie_json, width=width, height=height)

# Lottie animation
# lottie_json_file = r"C:\Users\Suyash Tambe\Desktop\Deep-Fakee-\Animation - 1712299045488.json"  
# load_lottie_animation(lottie_json_file, width=600, height=600)

# Function to preprocess uploaded image
def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    return image

# Function to predict deepfake from image
def predict_deepfake(image, model):
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    return predictions

# File uploader for image with colored text
st.subheader(':blue[Upload Image (JPG only):]')
uploaded_file = st.file_uploader("Choose a JPG image...", type="jpg")


# Predict button
if st.button("Predict", key="predict_button", help="Click to predict if the uploaded image is a deepfake or real."):
    if uploaded_file is not None:
        image = preprocess_image(uploaded_file)
        predictions = predict_deepfake(image, model)
        st.subheader(':blue[Prediction:]')
        threshold = 0.5
        if predictions[0][0] < threshold:
            st.write("<p style='color: red; font-size: 20px;'>The image is DEEPFAKE.</p>", unsafe_allow_html=True)
        else:
            st.write("<p style='color: green; font-size: 20px;'>The image is REAL.</p>", unsafe_allow_html=True)
    else:
        st.write("<p style='color: orange; font-size: 18px;'>Please upload an image before predicting.</p>", unsafe_allow_html=True)
