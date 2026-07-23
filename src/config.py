"""
Project-wide configuration settings.

This file contains constants shared across multiple modules.
Keeping configuration in one place makes the project easier to
maintain and avoids hardcoded values scattered throughout the codebase.
"""

# ==========================
# Dataset Paths
# ==========================

DATASET_DIR = "data/processed"
TEST_DATA_DIR = "data/test"

# ==========================
# Model Paths
# ==========================

MODEL_DIR = "models"
MODEL_NAME = "shoe_detector_efficientnet.keras"

# ==========================
# Image Configuration
# ==========================

IMAGE_SIZE = (224, 224)

# ==========================
# Training Configuration
# ==========================

BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 3e-5
VALIDATION_SPLIT = 0.20

# ==========================
# Prediction Configuration
# ==========================

PREDICTION_THRESHOLD = 0.5

# ==========================
# Reproducibility
# ==========================

RANDOM_SEED = 42