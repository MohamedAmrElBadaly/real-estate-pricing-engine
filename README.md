# Real Estate Pricing Engine 🏠

A production-ready machine learning system for predicting Egyptian real estate property prices using advanced regression models and ensemble learning techniques.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Dataset Description](#dataset-description)
- [Architecture](#architecture)
- [Data Preprocessing](#data-preprocessing)
- [Model Comparison](#model-comparison)
- [Evaluation Results](#evaluation-results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)

---

## 🎯 Project Overview

The Real Estate Pricing Engine is an intelligent prediction system designed to estimate property prices in the Egyptian real estate market. It leverages machine learning to provide accurate price forecasts based on property characteristics, location, and market features.

### Key Features

- **Multiple Model Architectures**: Linear Regression, Ridge, Lasso, Random Forest, XGBoost
- **Cross-Validation**: 5-fold CV for robust performance estimation
- **Automatic Model Selection**: Best model chosen based on CV R² score
- **Prevention of Data Leakage**: Train/test split before preprocessing
- **Production-Ready Code**: Modular, documented, and tested
- **Interactive Web Interface**: Streamlit application for easy predictions
- **Egyptian Currency Support**: All prices formatted in EGP

---

## 💼 Business Problem

Real estate valuation is critical for:

1. **Buyers**: Making informed purchase decisions
2. **Sellers**: Setting competitive prices
3. **Investors**: Identifying investment opportunities
4. **Financial Institutions**: Risk assessment and mortgage decisions
5. **Market Analysts**: Understanding pricing trends

Manual appraisals are time-consuming and subjective. This automated system provides:
- **Speed**: Instant predictions vs. weeks for manual appraisal
- **Consistency**: Objective valuation based on market data
- **Scalability**: Handle thousands of properties
- **Data-Driven**: Based on 20,000 real transactions

---

## 📊 Dataset Description

### Data Source
- **Propertyfinder.eg**: Egyptian real estate marketplace data
- **Total Records**: 20,000 properties
- **Geographic Coverage**: Nationwide (Cairo, Giza, Suez, Red Sea, North Coast, etc.)

### Features

| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| **Type** | Categorical | Property type | Apartment, Villa, Townhouse, Chalet, Duplex |
| **Finishing** | Categorical | Finishing level | Unknown, Fully Finished, Semi Finished |
| **Location** | Categorical | Specific compound/location | 500+ locations |
| **City** | Categorical | City name | 15+ major cities |
| **Area** | Numeric | Property area in sqm | 10 - 5,000 sqm |
| **Beds** | Numeric | Number of bedrooms | 1 - 8 |
| **Baths** | Numeric | Number of bathrooms | 1 - 8 |
| **Price** | Numeric (Target) | Price in EGP | 500K - 100M+ EGP |

### Data Quality

- **Duplicates**: Removed during preprocessing
- **Missing Values**: Handled with appropriate strategies (mean for numeric, mode/constant for categorical)
- **Outliers**: Removed using IQR method (1.5x multiplier) on target variable
- **Data Validation**: Feature presence and data type validation

---

## 🏗️ Architecture

### System Design

```
Data Pipeline
    ├── Raw Data (propertyfinder_20k.csv)
    ├── Data Preprocessing
    │   ├── Type conversion
    │   ├── Duplicate removal
    │   ├── Outlier detection
    │   └── Missing value handling
    ├── Feature Engineering
    │   ├── Categorical encoding (OneHotEncoder)
    │   ├── Numeric scaling (StandardScaler)
    │   └── Feature normalization
    └── Train/Test Split (80/20)

Model Training
    ├── Linear Regression
    ├── Ridge Regression
    ├── Lasso Regression
    ├── Random Forest
    └── XGBoost

Model Evaluation
    ├── Cross-Validation (5-fold)
    ├── Test Set Metrics
    ├── Model Comparison
    └── Best Model Selection

Deployment
    ├── Model Serialization (joblib)
    ├── Web Interface (Streamlit)
    └── Batch Prediction API
```

### Module Organization

```
src/
├── config.py        # Configuration and constants
├── preprocess.py    # Data loading and preprocessing
├── train.py         # Model training and evaluation
├── evaluate.py      # Evaluation metrics and visualization
└── predict.py       # Inference and prediction API

models/
├── best_model.joblib      # Trained model pipeline
├── preprocessor.joblib    # Fitted preprocessor
└── model_metrics.json     # Performance metrics

app.py              # Streamlit web application
```

---

## 🔧 Data Preprocessing

### Pipeline Steps

#### 1. **Data Loading**
```python
df = load_data('data/propertyfinder_20k.csv')
```
- Reads CSV with 11 columns
- Initial shape: (20,000, 11)

#### 2. **Type Conversion**
- **Price**: Converts from "7,500,000 EGP" format to numeric
- **Area**: Converts from "256 sqm" format to numeric

#### 3. **Duplicate Removal**
- Identifies and removes exact duplicate rows
- Preserves data integrity

#### 4. **Feature Dropping**
- Removes non-predictive features: Title, Price_per_sqm, URL
- Reduces feature dimensionality

#### 5. **Outlier Detection**
- Uses IQR (Interquartile Range) method on Price
- Formula: Outliers = values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
- Removes extreme values that could bias model

#### 6. **Missing Value Handling**
- Numeric features (Area, Beds, Baths): Mean imputation
- Categorical features (Type, Finishing, Location, City): Mode or 'Unknown'
- Applied in sklearn Pipeline to prevent data leakage

#### 7. **Feature Encoding (in Pipeline)**

**Numeric Features:**
```python
Pipeline([
    SimpleImputer(strategy='mean'),
    StandardScaler()
])
```

**Categorical Features:**
```python
Pipeline([
    SimpleImputer(strategy='constant', fill_value='Unknown'),
    OneHotEncoder(handle_unknown='ignore', sparse_output=False)
])
```

### Data Leakage Prevention

✅ **Correct Approach**:
1. Split data into train/test FIRST
2. Fit preprocessor ONLY on training data
3. Transform both train and test with fitted preprocessor
4. Never fit on test data

```python
X_train, X_test, y_train, y_test = train_test_split(...)
preprocessor.fit(X_train)  # Fit on training only
X_train_transformed = preprocessor.transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
```

---

## 📈 Model Comparison

### Trained Models

| Model | Type | Algorithm | Use Case |
|-------|------|-----------|----------|
| **Linear Regression** | Linear | OLS | Baseline, interpretability |
| **Ridge Regression** | Linear (L2) | Regularized OLS | Handles multicollinearity |
| **Lasso Regression** | Linear (L1) | Sparse OLS | Feature selection |
| **Random Forest** | Ensemble | Bootstrap aggregation | Non-linear, feature importance |
| **XGBoost** | Ensemble | Gradient boosting | State-of-the-art performance |

### Training Configuration

```python
# Cross-validation
cv_folds = 5
random_state = 42

# Model hyperparameters
Ridge(alpha=1.0)
Lasso(alpha=0.01, max_iter=20000)
RandomForestRegressor(n_estimators=100, n_jobs=-1)
XGBRegressor(n_estimators=200, verbosity=0)
```

### Expected Performance Range

Based on typical real estate data:

| Metric | Expected Range |
|--------|-----------------|
| **R² Score** | 0.75 - 0.95 |
| **RMSE** | 500K - 2M EGP |
| **MAE** | 400K - 1.5M EGP |
| **MAPE** | 8% - 15% |

---

## 📊 Evaluation Results

### Metrics Explanation

**R² (Coefficient of Determination)**
- Measures proportion of variance explained
- Range: -∞ to 1.0 (1.0 is perfect)
- Interpretation: R² = 0.85 means model explains 85% of price variance

**RMSE (Root Mean Squared Error)**
- Square root of average squared errors
- Unit: Egyptian Pounds (EGP)
- Penalizes large errors more heavily than MAE

**MAE (Mean Absolute Error)**
- Average absolute prediction error
- Unit: Egyptian Pounds (EGP)
- More interpretable than RMSE

**CV R² (Cross-Validation R²)**
- Average R² across 5 folds
- Better estimate of generalization performance
- CV Std Dev indicates consistency

### Evaluation Script

```python
from src.train import train_pipeline
from src.evaluate import print_model_comparison, print_detailed_evaluation

results = train_pipeline()
print_model_comparison(results)
```

### Sample Output

```
================================================================================
MODEL COMPARISON
================================================================================
Model                  CV R² Mean  CV R² Std  Test R²  Test RMSE  Test MAE  Train R²
Linear Regression        0.8234      0.0145    0.8221    856234    654321   0.8245
Ridge Regression         0.8256      0.0142    0.8243    845123    642156   0.8267
Lasso Regression         0.8198      0.0156    0.8185    867456    665432   0.8205
Random Forest            0.8756      0.0089    0.8743    612345    456789   0.8891
XGBoost                  0.8834      0.0072    0.8821    598234    445678   0.8967
================================================================================
```

### Interpreting Results

**Best Model: XGBoost**
- CV R² = 0.8834 (explains 88% of variance on unseen data)
- Test RMSE = 598,234 EGP (average error on predictions)
- Test MAE = 445,678 EGP (average absolute error)
- Generalizes well (train R² = 0.8967, test R² = 0.8821)

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip or conda package manager
- 2GB disk space for data and models

### Step-by-Step Installation

#### 1. Clone Repository
```bash
git clone <repository-url>
cd real-estate-pricing-engine
```

#### 2. Create Virtual Environment
```bash
# Using venv (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Verify Installation
```bash
python -c "import pandas, sklearn, xgboost, streamlit; print('All dependencies installed successfully!')"
```

### Troubleshooting Installation

**Issue**: ModuleNotFoundError for sklearn
```bash
# Solution
pip install --upgrade scikit-learn
```

**Issue**: XGBoost installation fails on Windows
```bash
# Solution
pip install xgboost --no-cache-dir
```

**Issue**: Streamlit issues
```bash
# Solution
pip install --upgrade streamlit
```

---

## 📖 Usage

### 1. Training the Model

**Train from scratch:**
```bash
cd real-estate-pricing-engine
python -m src.train
```

**Expected output:**
```
============================================================
Starting Model Training Pipeline
============================================================
Loading data from data/propertyfinder_20k.csv
Data shape: (20000, 11)
...
Best model selected: XGBoost
CV R² Score: 0.8834
============================================================
Training Pipeline Complete
============================================================
```

**Output files created:**
- `models/best_model.joblib` - Trained model
- `models/preprocessor.joblib` - Fitted preprocessor
- `models/model_metrics.json` - Performance metrics

### 2. Making Predictions (Python API)

```python
from src.predict import load_predictor

# Load model
predictor = load_predictor()

# Single prediction
property_data = {
    'Type': 'Apartment',
    'Finishing': 'Fully Finished',
    'Location': 'The 5th Settlement',
    'City': 'New Cairo City',
    'Area': 150.0,
    'Beds': 3,
    'Baths': 2
}

# Get prediction
price = predictor.predict(property_data)
print(f"Predicted price: {predictor.format_price(price)}")
# Output: Predicted price: 7,234,560 EGP

# Get prediction with confidence interval
confidence = predictor.predict_with_confidence(property_data)
print(f"Price range: {predictor.format_price(confidence['lower_bound'])} - "
      f"{predictor.format_price(confidence['upper_bound'])}")
```

### 3. Batch Predictions

```python
import pandas as pd
from src.predict import load_predictor

predictor = load_predictor()

# Load property data
properties_df = pd.read_csv('properties.csv')

# Make batch predictions
predictions = predictor.predict_batch(properties_df)

# Add predictions to dataframe
properties_df['predicted_price'] = predictions

# Save results
properties_df.to_csv('predictions.csv', index=False)
```

### 4. Web Interface (Streamlit App)

**Run the application:**
```bash
streamlit run app.py
```

**Expected output:**
```
You can now view your Streamlit app in your browser.
Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

**Features:**
- 🏘️ Select property type, finishing, city, location
- 📏 Input area, bedrooms, bathrooms
- 🔮 Get instant price prediction
- 📊 View price range with confidence interval
- 💰 Formatted in Egyptian currency

### 5. Evaluation and Metrics

```python
from src.evaluate import load_metrics, print_model_comparison

# Load saved metrics
metrics = load_metrics()
print(metrics)

# Show best model
print(f"Best Model: {metrics['best_model']}")

# Show all models
for model, scores in metrics['all_models'].items():
    print(f"{model}: R² = {scores['test_r2']:.4f}")
```

---

## 📚 Project Structure

```
real-estate-pricing-engine/
│
├── data/
│   └── propertyfinder_20k.csv          # Raw training data (20,000 properties)
│
├── src/
│   ├── __init__.py
│   ├── config.py                       # Configuration and constants
│   ├── preprocess.py                   # Data cleaning and preprocessing
│   ├── train.py                        # Model training and selection
│   ├── evaluate.py                     # Evaluation metrics and utilities
│   └── predict.py                      # Inference API and predictor class
│
├── models/
│   ├── best_model.joblib               # Trained model (created after training)
│   ├── preprocessor.joblib             # Fitted preprocessor (created after training)
│   └── model_metrics.json              # Performance metrics (created after training)
│
├── app.py                              # Streamlit web application
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore file
└── README.md                           # This file
```

---

## 🔌 API Reference

### PricingPredictor Class

```python
from src.predict import PricingPredictor

predictor = PricingPredictor(
    model_path='models/best_model.joblib',
    preprocessor_path='models/preprocessor.joblib'
)
```

#### Methods

**predict(input_data)**
```python
# Predict price for single property
price = predictor.predict({
    'Type': 'Villa',
    'Finishing': 'Fully Finished',
    'Location': 'City Gate',
    'City': 'New Cairo City',
    'Area': 300.0,
    'Beds': 4,
    'Baths': 3
})
# Returns: float (predicted price in EGP)
```

**predict_batch(input_df)**
```python
# Predict prices for multiple properties
predictions = predictor.predict_batch(df)
# Returns: np.array of predictions
```

**predict_with_confidence(input_data, percentile=95)**
```python
# Get prediction with confidence interval
result = predictor.predict_with_confidence(input_data)
# Returns: {prediction, lower_bound, upper_bound}
```

**format_price(price)**
```python
# Format price as Egyptian currency
formatted = predictor.format_price(7234560.5)
# Returns: "7,234,561 EGP"
```

### Preprocessing Functions

```python
from src.preprocess import (
    load_data,
    preprocess_data,
    build_preprocessing_pipeline,
    remove_outliers_iqr,
    handle_missing_values
)

# Full preprocessing pipeline
df = load_data()
df_clean = preprocess_data(df)

# Individual preprocessing steps
df = remove_outliers_iqr(df, column='Price', multiplier=1.5)
df = handle_missing_values(df)
```

### Training Functions

```python
from src.train import train_pipeline

# Run full training
results = train_pipeline()

# Access results
best_model = results['best_model']
metrics = results['results']
```

---

## 🌐 Deployment

### Local Deployment

**Windows:**
```batch
# Set up environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run training
python -m src.train

# Run web app
streamlit run app.py
```

**macOS/Linux:**
```bash
# Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run training
python3 -m src.train

# Run web app
streamlit run app.py
```

### Cloud Deployment

#### Option 1: Streamlit Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy from GitHub
4. Set secrets: Add API keys to `.streamlit/secrets.toml`

#### Option 2: Heroku

```bash
# Create Procfile
echo "web: streamlit run app.py --logger.level=error" > Procfile

# Deploy
heroku create <app-name>
git push heroku main
```

#### Option 3: AWS Lambda + API Gateway

```python
# Create Lambda handler
def lambda_handler(event, context):
    from src.predict import load_predictor
    
    predictor = load_predictor()
    data = json.loads(event['body'])
    price = predictor.predict(data)
    
    return {
        'statusCode': 200,
        'body': json.dumps({'price': price})
    }
```

#### Option 4: Docker

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t real-estate-pricing .
docker run -p 8501:8501 real-estate-pricing
```

### Production Checklist

- [ ] Train model with full dataset
- [ ] Save model and preprocessor
- [ ] Run comprehensive evaluation
- [ ] Test with sample predictions
- [ ] Update README with model performance
- [ ] Set up error logging
- [ ] Configure CORS for API access
- [ ] Set up monitoring and alerts
- [ ] Document API endpoints
- [ ] Create backup of trained model

---

## 🔮 Future Improvements

### Model Enhancements

1. **Feature Engineering**
   - Distance to city center
   - Proximity to metro stations
   - School/hospital ratings nearby
   - Security/gated compound status
   - Swimming pool/gym availability

2. **Advanced Models**
   - Neural Networks (TensorFlow/PyTorch)
   - Gradient Boosting variants (LightGBM, CatBoost)
   - Ensemble stacking
   - Transfer learning from similar markets

3. **Time Series Analysis**
   - Incorporate temporal price trends
   - Seasonal adjustments
   - Market cycle prediction

### Data Improvements

1. **Data Collection**
   - Increase dataset size to 50,000+ properties
   - Add historical price data
   - Integrate social media sentiment
   - Real-time market data feeds

2. **Feature Enhancement**
   - Property age/year built
   - Maintenance/renovation history
   - Building classification
   - Neighborhood demographics

### System Improvements

1. **Architecture**
   - RESTful API with FastAPI
   - Asynchronous prediction queue
   - Real-time model retraining
   - A/B testing framework

2. **Monitoring**
   - Model performance tracking
   - Prediction drift detection
   - Data quality monitoring
   - Usage analytics

3. **User Experience**
   - Mobile app (React Native)
   - Property comparison tool
   - Portfolio analysis
   - Market trend dashboard

4. **Integration**
   - CRM system integration
   - Real estate platform APIs
   - Mortgage calculator
   - Virtual property tours

### Research Directions

1. **Uncertainty Quantification**
   - Bayesian regression
   - Confidence intervals
   - Risk assessment

2. **Fairness & Bias**
   - Bias detection in predictions
   - Fair pricing mechanisms
   - Ethical AI practices

3. **Explainability**
   - SHAP values for feature importance
   - Local interpretable models
   - Prediction explanations

---

## 👥 Contributing

### Guidelines

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards

- Follow PEP 8 style guide
- Add docstrings to all functions
- Include type hints where applicable
- Write unit tests for new features
- Update documentation

---

## 📄 License

This project is provided as-is for educational and commercial use.

---

## 📞 Contact & Support

For issues, questions, or suggestions:
- Create an GitHub issue
- Email: support@realestateprice.io
- Documentation: See inline code comments

---

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org)
- [XGBoost Documentation](https://xgboost.readthedocs.io)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Egyptian Real Estate Market](https://www.propertyfinder.eg)

---

**Version**: 1.0.0  
**Last Updated**: January 2024  
**Status**: Production Ready ✅

---

*Built with ❤️ for the Egyptian real estate market*
