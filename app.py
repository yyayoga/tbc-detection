import os
import cv2
import numpy as np
import traceback
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Constants
MODEL_PATH = os.path.join("models", "tbc_model.h5")
IMG_SIZE = 224

# Load Model
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print(f"Warning: Model not found at {MODEL_PATH}. Please run train_model.py first.")
except Exception as e:
    print(f"Error loading model: {e}")

def predict_image(img):
    try:
        # Resize image
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Convert grayscale to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            # OpenCV loads as BGR, convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize and expand dimensions
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict
        prediction = model.predict(img, verbose=0)
        pred_value = prediction[0][0] if prediction.ndim == 2 else prediction[0]

        # 'Normal' vs 'TBC' (Alphabetical: Normal=0, TBC=1)
        if pred_value > 0.5:
            label = "TBC"
            confidence = pred_value * 100
        else:
            label = "Normal"
            confidence = (1.0 - pred_value) * 100

        return label, confidence
    except Exception as e:
        print(f"Error in predict_image: {e}")
        raise

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 400

    try:
        if "image" not in request.files:
            return jsonify({"error": "No image provided."}), 400
        
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No selected file."}), 400

        # Read image from memory
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({"error": "Invalid image format."}), 400

        label, confidence = predict_image(img)
        
        return jsonify({
            "label": label,
            "confidence": f"{confidence:.2f}%"
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal Server Error: " + str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)