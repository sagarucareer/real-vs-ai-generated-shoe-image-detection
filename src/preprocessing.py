from tensorflow.keras.preprocessing.image import ImageDataGenerator

from src.config import (
    DATASET_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    VALIDATION_SPLIT
)

def get_data_generators():
    #Create Training Data Generator
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT,
        rotation_range=25,
        zoom_range=0.2,
        width_shift_range=0.15,
        height_shift_range=0.15,
        brightness_range=(0.8, 1.2),
        horizontal_flip=True,
        fill_mode="nearest"
    )

    #Create Validation Data Generator
    validation_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT
    )

    #Create the Training Generator
    train_generator = train_datagen.flow_from_directory(
        directory=DATASET_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",#categorical if multiple classes
        subset="training",
        shuffle=True
    )

    #Create Validation Generator
    validation_generator = validation_datagen.flow_from_directory(
        directory=DATASET_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        subset="validation",
        shuffle=False
    )

    return train_generator, validation_generator