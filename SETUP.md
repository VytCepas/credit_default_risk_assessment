# Setup Guide

## Prerequisites
- Python 3.12

## Installation

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the risk model
python models/risk_model.py

# Optionally retrain the behavioral traits model
python models/behavioral_traits_model.py

# Run the app
streamlit run app.py
```

App opens at `http://localhost:8501`

**Note**: For Streamlit Cloud deployment, use `app.py` as the main file path.

## Usage
1. Fill out the questionnaire (personal, employment, loan details)
2. Submit for instant risk assessment
3. View risk score (0-1000), category, and recommendations

## Risk Categories
| Score | Category |
|-------|----------|
| 0-299 | Low Risk |
| 300-599 | Medium Risk |
| 600-1000 | High Risk |

## Data Files
Place in `data/` directory:
- `application_train.parquet` (required for training)
- `application_test.parquet` (optional)

Download from: [Kaggle - Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)

## Model Artifacts
Trained models are loaded from `src/assets/`:
- `risk_model.pkl`
- `behavioral_traits_model.pkl`
