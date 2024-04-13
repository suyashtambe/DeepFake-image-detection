from flask import Flask, render_template, request
from keras.layers import Flatten
import pickle as pk
import numpy as np
import cv2

app = Flask(__name__)

model = pk.load(open(r"Flask_webapp\models\rfr.pkl", "rb"))

# def preprocess_image(image_path, image_size):
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, image_size)
#     image = image.astype(np.float32) / 255.0
#     return np.expand_dims(image, axis=0)

# def predict_image(model, image_path, image_size):
#     input_image = preprocess_image(image_path, image_size)
#     prediction = model.predict(input_image)
#     if prediction[0][0] >= 0.9:
#         return "Real"
#     else:
#         return "Deep Fake"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/results", methods=["GET", "POST"])
def results():
    context = 0
    if request.method == "POST":
        try:
            image_file = request.files["image"]
            if image_file:
                image = cv2.imdecode(
                    np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR
                )
                image = cv2.resize(image, (224, 224))
                image = image / 255.0
                image_flat = image.flatten().reshape(1, -1)
                predictions = model.predict(image_flat)
                if predictions is not None:
                    context = predictions[0]
                    return render_template("results.html", context=context)
        except Exception as e:
            print(f"Error processing request: {e}")
            context["predictions"] = "Error processing request"

    return render_template("results.html", context=context)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
