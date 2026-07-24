import numpy as np

from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import IMAGE_SIZE

model = load_model("models/best_shoe_detector.keras")

def preprocess_image(uploaded_image):
    img = uploaded_image.convert("RGB")
    img = img.resize(IMAGE_SIZE)
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x

def predict_image(uploaded_image):
    x = preprocess_image(uploaded_image)
    prediction = model.predict(x, verbose=0)[0][0]
    label = "Real" if prediction >= 0.5 else "AI"
    confidence = prediction if label == "Real" else 1 - prediction
    return label, confidence