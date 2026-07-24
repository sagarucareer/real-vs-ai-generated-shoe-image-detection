import tensorflow as tf

from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    BatchNormalization,
    Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

from src.config import (
    IMAGE_SIZE,
    LEARNING_RATE,
    EPOCHS
)

from src.preprocessing import get_data_generators

def main():

    #model creation
    base_model = EfficientNetB3(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3)
    )

    # Freeze most layers
    for layer in base_model.layers[:-50]:
        layer.trainable = False

    # Fine-tune the last 50 layers
    for layer in base_model.layers[-50:]:
        layer.trainable = True

    #Build the Classification Head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)

    #Create the Final Model
    model = Model(
        inputs=base_model.input,
        outputs=output
    )

    #Compile the Model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC()
        ]
    )

    #print model architecture
    model.summary()

    #callbacks
    callbacks = [
        ModelCheckpoint(
            filepath="models/best_shoe_detector.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),

        EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    #training model
    train_generator, validation_generator = get_data_generators()

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    #Save the Final Model
    model.save("models/shoe_detector_final.keras")

    #final verdict
    print("✅ Training completed successfully.")
    print("📁 Best model saved to: models/best_shoe_detector.keras")
    print("📁 Final model saved to: models/shoe_detector_final.keras")

if __name__ == "__main__":
    main()