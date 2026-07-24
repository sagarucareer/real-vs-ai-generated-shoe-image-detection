import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc
)

from tensorflow.keras.models import load_model

from src.preprocessing import get_data_generators


def main():

    # Load model
    model = load_model("models/best_shoe_detector.keras")

    # Load validation data
    _, validation_generator = get_data_generators()

    # True labels
    y_true = validation_generator.classes

    # Predicted probabilities
    y_pred_prob = model.predict(
        validation_generator,
        verbose=1
    ).ravel()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_pred_prob
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    # Plot
    plt.figure(figsize=(7, 7))

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"AUC = {roc_auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()