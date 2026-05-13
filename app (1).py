"""
Streamlit web application for real estate price prediction.
Provides user-friendly interface for making property price predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Add root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from predict import load_predictor
from config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    CURRENCY_FORMAT,
)
from preprocess import load_data

# Configure logging
logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Real Estate Pricing Engine",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .metric-box {
        background-color: #fff;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .title-main {
        color: #1f77b4;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_data():
    """
    Load model and dataset once (cached).
    """
    try:
        predictor = load_predictor()
        data = load_data()
        return predictor, data
    except FileNotFoundError as e:
        st.error(f"Error loading model: {e}")
        st.error("Please run the training script first:")
        st.code("python -m src.train")
        st.stop()


def get_unique_values(data, column):
    """
    Get unique values from a column for dropdown.
    """
    return sorted(data[column].dropna().unique().tolist())


def main():
    """
    Main Streamlit application.
    """
    
    # Load model and data
    predictor, data = load_model_and_data()
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<div class="title-main">🏠 Real Estate Pricing Engine</div>', 
                   unsafe_allow_html=True)
        st.markdown("**Advanced ML-Powered Property Price Prediction for Egyptian Real Estate Market**")
    
    with col2:
        st.info("🤖 Powered by ML")
    
    st.divider()
    
    # Sidebar for inputs
    st.sidebar.markdown("## 📋 Property Details")
    st.sidebar.markdown("Fill in the property information below to get a price prediction.")
    
    with st.sidebar.form("prediction_form"):
        st.markdown("### Property Type & Finishing")
        
        # Property Type
        property_types = get_unique_values(data, 'Type')
        property_type = st.selectbox(
            "Property Type",
            options=property_types,
            help="Select the type of property"
        )
        
        # Finishing
        finishing_options = get_unique_values(data, 'Finishing')
        finishing = st.selectbox(
            "Finishing Level",
            options=finishing_options,
            help="Select the finishing level"
        )
        
        st.markdown("### Location Information")
        
        # City
        cities = get_unique_values(data, 'City')
        city = st.selectbox(
            "City",
            options=cities,
            help="Select the city"
        )
        
        # Location (filtered by city)
        city_data = data[data['City'] == city]
        locations = get_unique_values(city_data, 'Location')
        location = st.selectbox(
            "Location/Compound",
            options=locations,
            help="Select the specific location"
        )
        
        st.markdown("### Property Specifications")
        
        # Area
        area = st.number_input(
            "Area (sqm)",
            min_value=10.0,
            max_value=5000.0,
            value=150.0,
            step=10.0,
            help="Total area in square meters"
        )
        
        # Beds
        beds = st.selectbox(
            "Bedrooms",
            options=[1, 2, 3, 4, 5, 6, 7, 8],
            index=2,
            help="Number of bedrooms"
        )
        
        # Baths
        baths = st.selectbox(
            "Bathrooms",
            options=[1, 2, 3, 4, 5, 6, 7, 8],
            index=1,
            help="Number of bathrooms"
        )
        
        st.divider()
        
        # Prediction button
        predict_button = st.form_submit_button(
            "🔮 Predict Price",
            use_container_width=True,
            type="primary"
        )
    
    # Main content area
    if predict_button:
        try:
            # Prepare input
            input_data = {
                'Type': property_type,
                'Finishing': finishing,
                'Location': location,
                'City': city,
                'Area': float(area),
                'Beds': int(beds),
                'Baths': int(baths)
            }
            
            # Make prediction
            with st.spinner("🔄 Analyzing property... Please wait"):
                predicted_price = predictor.predict(input_data)
                formatted_price = predictor.format_price(predicted_price)
                confidence = predictor.predict_with_confidence(input_data)
            
            # Display results
            st.success("✅ Prediction Complete!")
            st.divider()
            
            # Main prediction display
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(
                    f'<div class="prediction-box">'
                    f'<h2 style="margin-top: 0; color: #1f77b4;">Predicted Price</h2>'
                    f'<h1 style="color: #2ca02c; margin: 20px 0;">{formatted_price}</h1>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            with col2:
                st.markdown(
                    f'<div class="prediction-box">'
                    f'<h2 style="margin-top: 0; color: #1f77b4;">Price Range</h2>'
                    f'<p style="margin: 10px 0;color: #2ca02c"><strong>Lower Bound:</strong> {predictor.format_price(confidence["lower_bound"])}</p>'
                    f'<p style="margin: 10px 0;color: #2ca02c"><strong>Upper Bound:</strong> {predictor.format_price(confidence["upper_bound"])}</p>'
                    f'<p style="margin: 10px 0; font-size: 12px; color: #666;">95% confidence interval</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            st.divider()
            
            # Property summary
            st.markdown("### 📌 Property Summary")
            
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.markdown(
                    f'<div class="metric-box">'
                    f'<p style="margin: 0; color: #666; font-size: 12px;">PROPERTY TYPE</p>'
                    f'<p style="margin: 8px 0; font-size: 18px; font-weight: bold;color: #2ca02c">{property_type}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            with summary_col2:
                st.markdown(
                    f'<div class="metric-box">'
                    f'<p style="margin: 0; color: #666; font-size: 12px;">LOCATION</p>'
                    f'<p style="margin: 8px 0; font-size: 18px; font-weight: bold;color: #2ca02c">{city}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            with summary_col3:
                st.markdown(
                    f'<div class="metric-box">'
                    f'<p style="margin: 0; color: #666; font-size: 12px;">AREA</p>'
                    f'<p style="margin: 8px 0; font-size: 18px; font-weight: bold;color: #2ca02c">{area:.0f} sqm</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )
            
            # Additional metrics
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Type", property_type)
            
            with metric_col2:
                st.metric("Bedrooms", int(beds))
            
            with metric_col3:
                st.metric("Bathrooms", int(baths))
            
            with metric_col4:
                price_per_sqm = predicted_price / float(area)
                st.metric("Price per sqm", f"{price_per_sqm:,.0f} EGP")
            
            # Additional info
            st.divider()
            st.info(
                "💡 **Note:** This prediction is based on machine learning model trained "
                "on historical property data. Actual prices may vary based on additional "
                "factors not included in the model."
            )
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            logger.error(f"Prediction error: {e}")
    
    else:
        # Initial state
        st.markdown("""
            ### 📊 About This Tool
            
            This application uses an advanced machine learning model trained on 20,000 Egyptian 
            real estate property listings to predict property prices.
            
            **Features:**
            - 🏘️ Support for multiple property types (Apartment, Villa, Townhouse, Chalet, Duplex)
            - 📍 Coverage of major Egyptian cities and locations
            - 📈 Accurate predictions based on area, bedrooms, bathrooms, and finishing level
            - 💬 Real-time pricing recommendations
            
            **How to use:**
            1. Fill in the property details in the sidebar
            2. Click "Predict Price" button
            3. View the predicted price and price range
            
            **Model Information:**
            - Algorithm: Ensemble of multiple regression models
            - Training Data: 20,000 properties
            - Accuracy: High R² score on test data
        """)
        
        st.divider()
        
        st.markdown("### 📈 Dataset Statistics")
        
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        
        with stats_col1:
            st.metric("Total Properties", len(data))
        
        with stats_col2:
            avg_price = pd.to_numeric(
                data['Price'].astype(str)
                .str.replace('EGP', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.strip(),
                errors='coerce'
            ).mean()
            st.metric("Average Price", predictor.format_price(avg_price))
        
        with stats_col3:
            avg_area = pd.to_numeric(
                data['Area'].astype(str)
                .str.replace('sqm', '', regex=False)
                .str.replace(',', '', regex=False)
                .str.strip(),
                errors='coerce'
            ).mean()
            st.metric("Avg Area (sqm)", f"{avg_area:.0f}")
        
        with stats_col4:
            st.metric("Avg Bedrooms", f"{data['Beds'].mean():.1f}")


if __name__ == "__main__":
    main()
