"""
Preprocessing module for data cleaning and feature engineering.
Implements data loading, cleaning, and sklearn Pipeline for consistency.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import logging

from config import (
    RAW_DATA_PATH,
    TARGET_COLUMN,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    FEATURES_TO_DROP,
    IQR_MULTIPLIER,
)

logger = logging.getLogger(__name__)


def load_data(filepath=RAW_DATA_PATH):
    """
    Load CSV data from filepath.
    """
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Data shape: {df.shape}")
    return df


def clean_price_column(df):
    """
    Convert Price column from string to numeric.
    """
    df = df.copy()

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = pd.to_numeric(
            df[TARGET_COLUMN].astype(str)
            .str.replace("EGP", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip(),
            errors="coerce"
        )

        logger.info("Price column converted to numeric")

    return df


def clean_area_column(df):
    """
    Convert Area column from string to numeric.
    """
    df = df.copy()

    if "Area" in df.columns:
        df["Area"] = pd.to_numeric(
            df["Area"].astype(str)
            .str.replace("sqm", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip(),
            errors="coerce"
        )

        logger.info("Area column converted to numeric")

    return df


def remove_duplicates(df):
    """
    Remove duplicate rows.
    """
    initial_shape = df.shape[0]
    df = df.drop_duplicates()
    duplicates_removed = initial_shape - df.shape[0]

    logger.info(f"Removed {duplicates_removed} duplicate rows")
    return df


def drop_unnecessary_features(df):
    """
    Drop features not needed for modeling.
    """
    df = df.copy()

    cols_to_drop = [col for col in FEATURES_TO_DROP if col in df.columns]

    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Dropped columns: {cols_to_drop}")

    return df


def validate_features(df):
    """
    Validate required features.
    """
    required_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    missing_features = [f for f in required_features if f not in df.columns]

    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found")

    logger.info("Feature validation passed")
    return True


def handle_missing_values(df):
    """
    Log missing values.
    """
    cols = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c in df.columns]
    missing_counts = df[cols].isnull().sum()

    if missing_counts.sum() > 0:
        logger.info(f"Missing values detected:\n{missing_counts[missing_counts > 0]}")

    return df


def remove_outliers_iqr(df, column):
    """
    Remove outliers using IQR.
    """
    df = df.copy()

    df[column] = pd.to_numeric(df[column], errors="coerce")

    values = df[column].dropna().values

    if len(values) == 0:
        logger.warning(f"No valid numeric values in {column}")
        return df

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    lower_bound = q1 - (IQR_MULTIPLIER * iqr)
    upper_bound = q3 + (IQR_MULTIPLIER * iqr)

    initial_shape = df.shape[0]

    df = df[
        (df[column] >= lower_bound) &
        (df[column] <= upper_bound)
    ]

    removed = initial_shape - df.shape[0]

    logger.info(
        f"Removed {removed} outliers from {column} "
        f"(bounds: {lower_bound:.0f} - {upper_bound:.0f})"
    )

    return df


def preprocess_data(df):
    """
    Full preprocessing pipeline.
    """
    logger.info("Starting data preprocessing pipeline")

    df = clean_price_column(df)
    df = clean_area_column(df)
    df = remove_duplicates(df)
    df = drop_unnecessary_features(df)
    validate_features(df)
    df = handle_missing_values(df)
    df = remove_outliers_iqr(df, TARGET_COLUMN)

    logger.info(f"Preprocessing complete. Final shape: {df.shape}")

    return df


def build_preprocessing_pipeline():
    """
    Build sklearn preprocessing pipeline.
    """

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    logger.info("Preprocessing pipeline built successfully")

    return preprocessor


if __name__ == "__main__":
    logging.basicConfig(level="INFO")

    df = load_data()
    df = preprocess_data(df)

    print(df.head())
    print(df.shape)