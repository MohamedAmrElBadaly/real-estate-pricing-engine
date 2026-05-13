"""
Model evaluation module.
Provides comprehensive evaluation metrics and visualization utilities.
"""

import pandas as pd
import numpy as np
import json
import logging
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)

from config import MODEL_METRICS_PATH

logger = logging.getLogger(__name__)


def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted target values
        
    Returns:
        dict: Dictionary of metrics
    """
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    metrics = {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2),
        'MAPE': float(mape),
        'Mean_Residual': float(residuals.mean()),
        'Std_Residual': float(residuals.std()),
        'Min_Residual': float(residuals.min()),
        'Max_Residual': float(residuals.max())
    }
    
    return metrics


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained sklearn pipeline
        X_test: Test features
        y_test: Test target
        
    Returns:
        dict: Test metrics
    """
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    
    logger.info("Test Set Metrics:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")
    
    return metrics


def print_model_comparison(results):
    """
    Print comparison table of all trained models.
    
    Args:
        results: dict from train_pipeline()
    """
    
    comparison_data = []
    
    for model_name, metrics in results['results'].items():
        comparison_data.append({
            'Model': model_name,
            'CV R² Mean': f"{metrics['cv_mean']:.4f}",
            'CV R² Std': f"{metrics['cv_std']:.4f}",
            'Test R²': f"{metrics['r2']:.4f}",
            'Test RMSE': f"{metrics['rmse']:.2f}",
            'Test MAE': f"{metrics['mae']:.2f}",
            'Train R²': f"{metrics['train_r2']:.4f}"
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    print("\n" + "="*100)
    print("MODEL COMPARISON")
    print("="*100)
    print(df_comparison.to_string(index=False))
    print("="*100 + "\n")


def save_metrics(results, best_model_name):
    """
    Save model metrics to JSON file.
    
    Args:
        results: dict from train_pipeline()
        best_model_name: Name of best model
    """
    
    metrics_dict = {
        'best_model': best_model_name,
        'all_models': {}
    }
    
    for model_name, metrics in results['results'].items():
        metrics_dict['all_models'][model_name] = {
            'cv_r2_mean': float(metrics['cv_mean']),
            'cv_r2_std': float(metrics['cv_std']),
            'test_r2': float(metrics['r2']),
            'test_rmse': float(metrics['rmse']),
            'test_mae': float(metrics['mae']),
            'train_r2': float(metrics['train_r2'])
        }
    
    with open(MODEL_METRICS_PATH, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    
    logger.info(f"Metrics saved to {MODEL_METRICS_PATH}")


def load_metrics():
    """
    Load saved model metrics.
    
    Returns:
        dict: Loaded metrics
    """
    try:
        with open(MODEL_METRICS_PATH, 'r') as f:
            metrics = json.load(f)
        return metrics
    except FileNotFoundError:
        logger.warning(f"Metrics file not found at {MODEL_METRICS_PATH}")
        return None


def print_detailed_evaluation(model, X_test, y_test, y_pred):
    """
    Print detailed evaluation including prediction statistics.
    
    Args:
        model: Trained sklearn pipeline
        X_test: Test features
        y_test: True test target
        y_pred: Model predictions
    """
    
    print("\n" + "="*80)
    print("DETAILED MODEL EVALUATION")
    print("="*80)
    
    metrics = calculate_metrics(y_test, y_pred)
    
    print(f"\nPrediction Accuracy Metrics:")
    print(f"  Mean Absolute Error (MAE):        {metrics['MAE']:>15,.2f}")
    print(f"  Root Mean Squared Error (RMSE):  {metrics['RMSE']:>15,.2f}")
    print(f"  R² Score:                         {metrics['R2']:>15,.4f}")
    print(f"  Mean Absolute % Error (MAPE):    {metrics['MAPE']:>15,.4f}")
    
    print(f"\nResidual Statistics:")
    print(f"  Mean Residual:                    {metrics['Mean_Residual']:>15,.2f}")
    print(f"  Std Dev Residual:                 {metrics['Std_Residual']:>15,.2f}")
    print(f"  Min Residual:                     {metrics['Min_Residual']:>15,.2f}")
    print(f"  Max Residual:                     {metrics['Max_Residual']:>15,.2f}")
    
    print(f"\nPrediction Range:")
    print(f"  Min Predicted:                    {y_pred.min():>15,.2f}")
    print(f"  Max Predicted:                    {y_pred.max():>15,.2f}")
    print(f"  Mean Predicted:                   {y_pred.mean():>15,.2f}")
    
    print(f"\nActual Range:")
    print(f"  Min Actual:                       {y_test.min():>15,.2f}")
    print(f"  Max Actual:                       {y_test.max():>15,.2f}")
    print(f"  Mean Actual:                      {y_test.mean():>15,.2f}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    print("Evaluation module loaded successfully")
