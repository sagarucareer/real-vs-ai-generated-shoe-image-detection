import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import load_model

from src.preprocessing import get_data_generators


def main():
    #Load Trained Model
    model = load_model("models/best_shoe_detector.keras")

    #Load Validation Data
    _, validation_generator = get_data_generators()

    #Get True Labels
    y_true = validation_generator.classes

    #Predict Probabilities
    y_pred_prob = model.predict(
        validation_generator,
        verbose=1
    ).ravel()
    #Because model.predict() returns a 2D array (n,1) but most scikit-learn functions expect a 1D array. so we use .ravel() to convert from 2d to 1d

    #Convert Probabilities to Classes
    y_pred = (y_pred_prob >= 0.5).astype(int)

    #Create Confusion Matrix
    cm = confusion_matrix(
        y_true,
        y_pred
    )

    #Display Confusion Matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["AI", "Real"]
    )

    disp.plot(cmap="Blues", values_format="d")

    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    main()