from flask import Flask, render_template,request
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)

def get_model(model_path):
    model = load_model(model_path)
    return model

def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes.read(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = np.expand_dims(image, axis=0)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    return image

def predict_deepfake(image, path):
    model = get_model(path)
    image = preprocess_image(image)
    predictions = model.predict(image)
    return predictions

@app.route('/')
def home():
    return render_template(r'index.html')

@app.route('/results',methods =["GET", "POST"])
def results():
    context = {
        "predictions": ""
    }
    
    if request.method == "POST":
        image = request.files['image']
        # preds = predict_deepfake(image, 'models/deepfake_autoencoder.keras')
        context["predictions"] = "These are predictions"
        return render_template(r'results.html', context=context)
    return render_template(r'results.html', context=context)
    


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)