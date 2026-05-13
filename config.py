"""
Configuration module for real estate pricing engine.
Centralizes all project settings and constants.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SRC_DIR = PROJECT_ROOT / "src"

# Data paths
RAW_DATA_PATH = DATA_DIR / "propertyfinder_20k.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.csv"

# Model paths
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
MODEL_METRICS_PATH = MODELS_DIR / "model_metrics.json"

# Data configuration
TARGET_COLUMN = "Price"
NUMERIC_FEATURES = ["Area", "Beds", "Baths"]
CATEGORICAL_FEATURES = ["Type", "Finishing", "Location", "City"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Remove features from modeling
FEATURES_TO_DROP = ["Title", "Price_per_sqm", "URL"]

# Preprocessing configuration
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Outlier detection (IQR method)
IQR_MULTIPLIER = 1.5

# Model hyperparameters
POLYNOMIAL_DEGREE = 2
RANDOM_FOREST_N_ESTIMATORS = 100
XGBOOST_N_ESTIMATORS = 100

# Display configuration
CURRENCY_SYMBOL = "EGP"
CURRENCY_FORMAT = f"{{:,.0f}} {CURRENCY_SYMBOL}"

# Logging
LOG_LEVEL = "INFO"
