import numpy as np

from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from src.config import IMAGE_SIZE

def main():
    model = load_model("models/best_shoe_detector.keras")

    #Ask for Image Path
    image_path = input("Enter image path: ").strip()

    #Load the image
    img = Image.open(image_path).convert("RGB")

    #Resize image
    img = img.resize(IMAGE_SIZE)

    #Convert image to numpy array
    x = image.img_to_array(img)

    #Normalize image
    x = x / 255.0

    #Add Batch Dimension
    x = np.expand_dims(x, axis=0)

    #Make the Prediction
    prediction = model.predict(x, verbose=0)[0][0]

    #Convert Probability to Label
    label = "Real" if prediction >= 0.5 else "AI"

    #Calculate Confidence
    confidence = prediction if label == "Real" else 1 - prediction

    #Display the Result
    print("\nPrediction Result")
    print("-" * 30)

    print(f"Prediction : {label}")
    print(f"Confidence : {confidence:.2%}")

if __name__ == "__main__":
    main()