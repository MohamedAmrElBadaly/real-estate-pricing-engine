"""
Model training module.
Trains and evaluates multiple regression models with cross-validation.
Automatically selects best model based on CV performance.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
import joblib

from config import (
    TEST_SIZE,
    RANDOM_STATE,
    CV_FOLDS,
    POLYNOMIAL_DEGREE,
    RANDOM_FOREST_N_ESTIMATORS,
    XGBOOST_N_ESTIMATORS,
    TARGET_COLUMN,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    BEST_MODEL_PATH,
    PREPROCESSOR_PATH,
)
from preprocess import preprocess_data, build_preprocessing_pipeline, load_data

logger = logging.getLogger(__name__)


def prepare_training_data():
    """
    Load, preprocess, and split data for training.
    Ensures no data leakage by splitting BEFORE preprocessing.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    logger.info("Preparing training data")
    
    # Load and clean data
    df = load_data()
    df = preprocess_data(df)
    
    # Separate features and target
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]
    
    # Split BEFORE pipeline to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    logger.info(
        f"Data split - Train: {X_train.shape}, Test: {X_test.shape}"
    )
    
    return X_train, X_test, y_train, y_test


def build_models(preprocessor):
    """
    Build dictionary of models to train and evaluate.
    Each model includes preprocessing in the pipeline.
    
    Args:
        preprocessor: sklearn ColumnTransformer for data preprocessing
        
    Returns:
        dict: {model_name: sklearn Pipeline}
    """
    
    models = {
        'Linear Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('model', LinearRegression())
        ]),
        
        'Ridge Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('model', Ridge(alpha=1.0))
        ]),
        
        'Lasso Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('model', Lasso(
                alpha=0.01,
                max_iter=20000
            ))
        ]),


        
        'Random Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(
                n_estimators=RANDOM_FOREST_N_ESTIMATORS,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),
        
        'XGBoost': Pipeline([
            ('preprocessor', preprocessor),
            ('model', XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                verbosity=0,
                n_jobs=-1
            ))
        ])
    }
    
    logger.info(f"Built {len(models)} models: {list(models.keys())}")
    return models


def train_and_evaluate_models(X_train, X_test, y_train, y_test, models):
    """
    Train models and evaluate with cross-validation and test set metrics.
    
    Args:
        X_train, X_test, y_train, y_test: Train/test data
        models: dict of sklearn pipelines
        
    Returns:
        dict: Model metrics and trained models
    """
    
    results = {}
    
    for model_name, pipeline in models.items():
        logger.info(f"Training {model_name}...")
        
        try:
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Cross-validation score (using R² metric)
            cv_scores = cross_val_score(
                pipeline, X_train, y_train,
                cv=CV_FOLDS,
                scoring='r2'
            )
            
            # Test set predictions
            y_pred_train = pipeline.predict(X_train)
            y_pred_test = pipeline.predict(X_test)
            
            # Calculate metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            mae = mean_absolute_error(y_test, y_pred_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2 = r2_score(y_test, y_pred_test)
            
            results[model_name] = {
                'model': pipeline,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'train_r2': r2_score(y_train, y_pred_train)
            }
            
            logger.info(
                f"{model_name} - CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f}) | "
                f"Test R²: {r2:.4f} | RMSE: {rmse:.2f} | MAE: {mae:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            continue
    
    return results


def select_best_model(results):
    """
    Select best model based on cross-validation R² score.
    
    Args:
        results: dict of model metrics from train_and_evaluate_models
        
    Returns:
        tuple: (best_model_name, best_results)
    """
    
    best_model_name = max(
        results.keys(),
        key=lambda x: results[x]['cv_mean']
    )
    best_results = results[best_model_name]
    
    logger.info(f"Best model selected: {best_model_name}")
    logger.info(f"CV R² Score: {best_results['cv_mean']:.4f}")
    
    return best_model_name, best_results


def save_model_and_preprocessor(best_model, preprocessor):
    """
    Save trained model and preprocessor for inference.
    
    Args:
        best_model: Trained sklearn pipeline
        preprocessor: Fitted preprocessor
    """
    joblib.dump(best_model, BEST_MODEL_PATH)
    logger.info(f"Best model saved to {BEST_MODEL_PATH}")
    
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    logger.info(f"Preprocessor saved to {PREPROCESSOR_PATH}")


def train_pipeline():
    """
    Execute full training pipeline.
    
    Returns:
        dict: Training results and best model
    """
    logger.info("="*60)
    logger.info("Starting Model Training Pipeline")
    logger.info("="*60)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_training_data()
    
    # Build preprocessing pipeline (fit on training data only)
    preprocessor = build_preprocessing_pipeline()

    
    # Build models
    models = build_models(preprocessor)
    
    # Train and evaluate
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test, models)
    
    # Select best model
    best_model_name, best_results = select_best_model(results)
    
    # Save model
    save_model_and_preprocessor(best_results['model'], preprocessor)
    
    logger.info("="*60)
    logger.info("Training Pipeline Complete")
    logger.info("="*60)
    
    return {
        'best_model_name': best_model_name,
        'best_model': best_results['model'],
        'results': results,
        'X_test': X_test,
        'y_test': y_test
    }


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    train_pipeline()
