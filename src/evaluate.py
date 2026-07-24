from tensorflow.keras.models import load_model

from src.preprocessing import get_data_generators

def main():
    #load model
    model = load_model("models/best_shoe_detector.keras")

    #validation generator
    _, validation_generator = get_data_generators()

    #Evaluate the Model
    loss, accuracy, precision, recall, auc = model.evaluate(
        validation_generator,
        verbose=1
    )

    #Calculate F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7) #1e-7 is to avoid division by 0

    print("\n📊 MODEL EVALUATION RESULTS")
    print("-" * 35)

    print(f"Loss       : {loss:.4f}")
    print(f"Accuracy   : {accuracy:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"F1 Score   : {f1_score:.4f}")
    print(f"ROC-AUC    : {auc:.4f}")

if __name__ == "__main__":
    main()