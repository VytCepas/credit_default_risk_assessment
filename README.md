# Home Credit Default Risk Assessment

A machine learning web application for loan default risk prediction with interactive questionnaire and behavioral analysis.

## Streamlit App
The application can be accessed via link: https://creditdefaultriskassessment-3gbekkxdejds9hjbevvbn2.streamlit.app

## Overview

This project implements a credit risk assessment system using machine learning to predict loan default probability. The system includes:
- Risk prediction model (ROC-AUC: 0.63)
- Behavioral traits analysis
- Interactive Streamlit web interface
- SHAP explainability for model predictions

## Methods

### Data Processing
- **Dataset**: Home Credit Default Risk (307,511 samples, 15 features)
- **Preprocessing**: Feature engineering, missing value imputation, categorical encoding
- **Feature Groups**: 
  - Numerical (7): age, employment years, income, credit amount, etc.
  - Binary (4): gender, car ownership, housing ownership, contract type
  - Categorical (4): income type, education, family status, housing type

### Model Architecture
- **Algorithm**: Gradient Boosting Classifier
- **Pipeline**: 
  1. Feature preprocessing (StandardScaler + OneHotEncoder)
  2. Class balancing (SMOTETomek)
  3. Classification (GradientBoostingClassifier)
- **Explainability**: SHAP values for feature importance

### Evaluation
- Train/Test Split: 80/20
- Metrics: ROC-AUC, Balanced Accuracy, F1-Score
- Optimal threshold: 0.37 (maximizing balanced accuracy)

## Results

| Metric | Threshold 0.5 | Optimal (0.37) |
|--------|---------------|----------------|
| ROC-AUC | 0.6272 | 0.6272 |
| Balanced Accuracy | 0.5424 | 0.5844 |

**Total Features**: 32 (7 numerical + 4 binary + 21 one-hot encoded categorical)

## Conclusions

1. **Model Performance**: The model achieves moderate predictive power (ROC-AUC: 0.63), indicating room for improvement through:
   - Additional feature engineering
   - External data sources
   - Advanced ensemble methods

2. **Class Imbalance**: Default rate of 8.1% required resampling techniques (SMOTETomek) to improve minority class detection

3. **Feature Importance**: Top predictors include credit amount, income levels, and employment history

4. **Optimal Threshold**: Using 0.37 instead of 0.5 improves balanced accuracy by 4.2 percentage points, better suited for imbalanced classification

5. **Production Ready**: The system provides:
   - User-friendly questionnaire interface
   - Real-time risk assessment (0-1000 score)
   - Explainable predictions via SHAP
   - Risk categorization with recommendations

## Quick Start

See [SETUP.md](SETUP.md) for installation and usage instructions.