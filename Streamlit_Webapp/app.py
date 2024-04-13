import streamlit as st
from tensorflow.keras.models import load_model
from streamlit_lottie import st_lottie
import cv2
import numpy as np
import base64

model = load_model(r"trained_fake_model.h5")


def get_base64(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f"""
    <style>
    .stApp {{
    background-image: url("data:image/png;base64,{bin_str}");
    background-size: cover;
    backdrop-filter: blur(1000px);
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)


def load_lottie_animation(json_file, width=None, height=None):
    with open(json_file, "r") as f:
        lottie_json = f.read()
    st_lottie(lottie_json, width=width, height=height)


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


def main():
    set_background(r"Flask_Webapp\Templates\Images\bg_image.jpg")

    st.markdown(
        "<h1 style='text-align: center; color: white; font-family: Arial, sans-serif;'>Deep Fake Image Detection using Multidisciplinary Approach</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-family: Arial, sans-serif; color: white; font-size: 18px; text-align: center;'>Collecting diverse dataset. Preprocess images. Extract attributes with CNNs. Split dataset. Select, train model. Monitor performance. Test for generalization.</p>",
        unsafe_allow_html=True,
    )

    st.subheader(":blue[Upload Image (JPG only):]")
    uploaded_file = st.file_uploader("Choose a JPG image...", type="jpg")

    if st.button(
        "Predict",
        key="predict_button",
        help="Click to predict if the uploaded image is a deepfake or real.",
    ):
        if uploaded_file is not None:
            image = preprocess_image(uploaded_file)
            predictions = predict_deepfake(image, model)
            st.subheader(":blue[Prediction:]")
            threshold = 0.9
            if predictions[0][0] < threshold:
                st.error("The image is DEEPFAKE", icon="🚨")
            if predictions[0][0] >= threshold:
                st.success("The image is REAL", icon="✅")
            st.write(predictions[0][0])
        else:
            st.info("Please upload an image before predicting.", icon="⚠️")


if __name__ == "__main__":
    main()
