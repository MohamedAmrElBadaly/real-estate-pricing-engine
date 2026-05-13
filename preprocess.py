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
from pathlib import Path

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
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        DataFrame with loaded data
    """
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Data shape: {df.shape}")
    return df


def clean_price_column(df):
    """
    Convert Price column from string format (e.g., '7,500,000 EGP') to numeric.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with cleaned Price column
    """
    df = df.copy()
    if df[TARGET_COLUMN].dtype == 'object':
        # Remove 'EGP' and commas, convert to numeric
        df[TARGET_COLUMN] = (
            df[TARGET_COLUMN]
            .str.replace(' EGP', '')
            .str.replace(',', '')
            .astype(float)
        )
        logger.info("Price column converted to numeric")
    return df


def clean_area_column(df):
    """
    Convert Area column from string format (e.g., '256 sqm') to numeric.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with cleaned Area column
    """
    df = df.copy()
    if 'Area' in df.columns and df['Area'].dtype == 'object':
        df['Area'] = (
            df['Area']
            .str.replace(' sqm', '')
            .str.replace(',', '')
            .astype(float)
        )
        logger.info("Area column converted to numeric")
    return df


def remove_duplicates(df):
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with duplicates removed
    """
    initial_shape = df.shape[0]
    df = df.drop_duplicates()
    duplicates_removed = initial_shape - df.shape[0]
    logger.info(f"Removed {duplicates_removed} duplicate rows")
    return df


def remove_outliers_iqr(df, column=TARGET_COLUMN, multiplier=IQR_MULTIPLIER):
    """
    Remove outliers using Interquartile Range (IQR) method.
    
    Args:
        df: Input DataFrame
        column: Column to detect outliers on
        multiplier: IQR multiplier (default 1.5)
        
    Returns:
        DataFrame with outliers removed
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    initial_shape = df.shape[0]
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    outliers_removed = initial_shape - df.shape[0]
    
    logger.info(
        f"Removed {outliers_removed} outliers from {column} "
        f"(bounds: {lower_bound:.0f} - {upper_bound:.0f})"
    )
    return df


def handle_missing_values(df):
    """
    Handle missing values with logging.
    Numeric columns: mean imputation will be done in pipeline.
    Categorical columns: 'Unknown' will be filled in pipeline.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missing values handled
    """
    missing_counts = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].isnull().sum()
    if missing_counts.sum() > 0:
        logger.info(f"Missing values detected:\n{missing_counts[missing_counts > 0]}")
    return df


def drop_unnecessary_features(df):
    """
    Drop features not needed for modeling.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with unnecessary features dropped
    """
    df = df.copy()
    cols_to_drop = [col for col in FEATURES_TO_DROP if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"Dropped columns: {cols_to_drop}")
    return df


def validate_features(df):
    """
    Validate that all required features are present.
    
    Args:
        df: Input DataFrame
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
    missing_features = [f for f in required_features if f not in df.columns]
    
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in data")
    
    logger.info("Feature validation passed")
    return True


def preprocess_data(df):
    """
    Apply full preprocessing pipeline.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Preprocessed DataFrame ready for modeling
    """
    logger.info("Starting data preprocessing pipeline")
    
    # Step 1: Clean columns
    df = clean_price_column(df)
    df = clean_area_column(df)
    
    # Step 2: Remove duplicates
    df = remove_duplicates(df)
    
    # Step 3: Drop unnecessary features
    df = drop_unnecessary_features(df)
    
    # Step 4: Validate features
    validate_features(df)
    
    # Step 5: Handle missing values (mark them)
    df = handle_missing_values(df)
    
    # Step 6: Remove outliers on target variable
    df = remove_outliers_iqr(df, column=TARGET_COLUMN)
    
    logger.info(f"Preprocessing complete. Final shape: {df.shape}")
    return df


def build_preprocessing_pipeline():
    """
    Build sklearn ColumnTransformer pipeline for consistent preprocessing.
    This pipeline prevents data leakage by fitting only on training data.
    
    Returns:
        sklearn ColumnTransformer with transformations
    """
    
    # Numeric pipeline: impute missing, then scale
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: fill missing with 'Unknown', then one-hot encode
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine both pipelines
    preprocessor = ColumnTransformer([
        ('numeric', numeric_pipeline, NUMERIC_FEATURES),
        ('categorical', categorical_pipeline, CATEGORICAL_FEATURES)
    ])
    
    logger.info("Preprocessing pipeline built successfully")
    return preprocessor


if __name__ == "__main__":
    # For testing the module
    logging.basicConfig(level="INFO")
    
    df = load_data()
    df = preprocess_data(df)
    print(df.head())
    print(f"\nFinal shape: {df.shape}")
