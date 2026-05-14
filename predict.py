"""
Prediction module for inference with trained model.
Loads saved model and preprocessor for making predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path

from config import (
    BEST_MODEL_PATH,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    CURRENCY_FORMAT,
)

logger = logging.getLogger(__name__)


class PricingPredictor:
    """
    Predictor class for real estate price predictions.
    Handles loading model and making predictions.
    """
    
    def __init__(self):
        self.model_path = BEST_MODEL_PATH

        if not self.model_path.exists():
            logger.info("Model not found. Training model automatically...")
            
            from train import train_pipeline
            results = train_pipeline()
            self.model = results['best_model']
        else:
            self.model = joblib.load(self.model_path)
    

    
    def predict(self, input_data):
        """
        Make price prediction for single property.
        
        Args:
            input_data: dict with keys: Type, Finishing, Location, City, Area, Beds, Baths
                       OR pandas DataFrame with same columns
        
        Returns:
            float: Predicted price
        """
        # Convert dict to DataFrame if needed
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        # Validate required features
        required_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
        missing_features = [f for f in required_features if f not in input_data.columns]
        
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select only required features in correct order
        input_data = input_data[required_features]
        
        # Make prediction
        prediction = self.model.predict(input_data)[0]
        
        return float(prediction)
    
    def predict_batch(self, input_df):
        """
        Make predictions for multiple properties.
        
        Args:
            input_df: DataFrame with property features
        
        Returns:
            np.array: Array of predictions
        """
        required_features = CATEGORICAL_FEATURES + NUMERIC_FEATURES
        input_df = input_df[required_features]
        
        predictions = self.model.predict(input_df)
        return predictions
    
    def predict_with_confidence(self, input_data, percentile=95):
        """
        Make prediction with confidence interval estimate.
        Based on residual statistics from training.
        
        Args:
            input_data: Property features (dict or DataFrame)
            percentile: Confidence interval percentile
        
        Returns:
            dict: {prediction, lower_bound, upper_bound}
        """
        prediction = self.predict(input_data)
        
        # Estimate confidence interval (simplified approach)
        # In production, would use proper uncertainty quantification
        margin_of_error = prediction * 0.1  # 10% margin
        
        return {
            'prediction': prediction,
            'lower_bound': prediction - margin_of_error,
            'upper_bound': prediction + margin_of_error
        }
    
    def format_price(self, price):
        """
        Format price as Egyptian currency string.
        
        Args:
            price: Numeric price value
        
        Returns:
            str: Formatted price string
        """
        return CURRENCY_FORMAT.format(price)
    
    def predict_and_format(self, input_data):
        """
        Make prediction and return formatted price string.
        
        Args:
            input_data: Property features
        
        Returns:
            str: Formatted price prediction
        """
        price = self.predict(input_data)
        return self.format_price(price)


def load_predictor():
    """
    Convenience function to load predictor.
    
    Returns:
        PricingPredictor: Initialized predictor instance
    """
    try:
        predictor = PricingPredictor()
        return predictor
    except Exception as e:
        logger.error(f"Could not load predictor: {e}")
        logger.error("Failed during training. Please check the error above.")
        raise


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    
    # Example usage
    predictor = load_predictor()
    
    # Test prediction
    test_property = {
        'Type': 'Apartment',
        'Finishing': 'Fully Finished',
        'Location': 'The 5th Settlement',
        'City': 'New Cairo City',
        'Area': 150.0,
        'Beds': 3,
        'Baths': 2
    }
    
    price = predictor.predict(test_property)
    formatted_price = predictor.format_price(price)
    
    print(f"\nTest Property Prediction:")
    print(f"  Input: {test_property}")
    print(f"  Predicted Price: {formatted_price}")
