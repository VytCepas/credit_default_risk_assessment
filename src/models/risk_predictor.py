from pathlib import Path
import numpy as np
import streamlit as st
from models.risk_model import RiskModel


def load_risk_predictor(model_path: str):
    """Load the trained risk prediction model."""
    try:
        if RiskModel is None:
            raise ImportError("RiskModel not available")

        # Project root is now 2 levels up from src/models/
        project_root = Path(__file__).parent.parent.parent
        data_dir = project_root / "data"

        model = RiskModel(data_directory=str(data_dir))
        model.load(model_path)

        return model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        raise e


def get_available_models():
    """Get available Risk Prediction model file paths by checking what models actually exist."""
    assets_dir = Path(__file__).parent.parent / "assets"

    potential_models = {
        "risk_prediction_model": {
            "path": assets_dir / "risk_model.pkl",
            "display_name": "Risk Prediction Model",
            "description": "Enhanced loan default risk assessment model with binary feature optimization and standardization",
            "icon": "🎯",
        },
        "behavioral_traits_model": {
            "path": assets_dir / "behavioral_traits_model.pkl",
            "display_name": "Behavioral Traits Model",
            "description": "Model incorporating behavioral traits for risk assessment",
            "icon": "🧠",
        },
        "imbalanced_fixed_model": {
            "path": assets_dir / "imbalanced_fixed_model.pkl",
            "display_name": "Imbalanced Fixed Model",
            "description": "Legacy risk assessment model with class balancing",
            "icon": "⚖️",
        },
        "home_credit_model": {
            "path": assets_dir / "home_credit_model.pkl",
            "display_name": "Home Credit Risk Model",
            "description": "Primary loan default risk assessment model",
            "icon": "🏠",
        },
        "fraud_detection_model": {
            "path": assets_dir / "fraud_detection_model.pkl",
            "display_name": "Fraud Detection Model",
            "description": "Fraud risk assessment model",
            "icon": "🚨",
        },
        "income_verification_model": {
            "path": assets_dir / "income_verification_model.pkl",
            "display_name": "Income Verification Model",
            "description": "Income stability and verification assessment",
            "icon": "💰",
        },
    }

    # Only return models that actually exist
    available_models = {}
    for model_key, model_info in potential_models.items():
        if model_info["path"].exists():
            available_models[model_key] = {
                "path": str(model_info["path"]),
                "display_name": model_info["display_name"],
                "description": model_info["description"],
                "icon": model_info["icon"],
            }

    return available_models


def predict_with_explanations(model, client_responses):
    """
    Make prediction and get SHAP explanations.

    This function expects `client_responses` to be the raw questionnaire output
    from the Streamlit form (string keys like "gender", "age", etc.). It maps
    those into the model's expected feature schema and then calls the model.

    Args:
        model: Trained RiskModel instance
        client_responses: Dict of questionnaire responses with string keys

    Returns:
        tuple: (prediction_results, shap_values, processed_features)
    """
    try:
        questionnaire_to_model_keys = {
            "gender": "gender",
            "age": "age",
            "total_income": "total_income",
            "employment_status": "employment_status",
            "years_employed": "years_employed",
            "education_level": "education_level",
            "family_status": "family_status",
            "num_children": "num_children",
            "num_family_members": "num_family_members",
            "owns_car": "owns_car",
            "owns_housing": "owns_housing",
            "housing_type": "housing_type",
            "contract_type": "contract_type",
            "credit_amount": "credit_amount",
            "loan_annuity": "loan_annuity",
        }

        client_features = {}
        for q_key, q_value in (client_responses or {}).items():
            if q_key in questionnaire_to_model_keys:
                model_key = questionnaire_to_model_keys[q_key]
                client_features[model_key] = q_value

        result = model.predict_with_explanations(client_features, top_n=10)

        prediction_results = {
            "risk_probability": result.get("risk_probability"),
            "risk_score": result.get("risk_score"),
            "risk_category": result.get("risk_category"),
            "recommendation": result.get("recommendation"),
            "color": result.get("color"),
        }

        shap_values = None
        processed_features = None

        if "shap_explanations" in result and result["shap_explanations"]:
            shap_explanations = result["shap_explanations"]
            shap_values = np.array([item["contribution"] for item in shap_explanations])
            processed_features = [item["feature"] for item in shap_explanations]

        return prediction_results, shap_values, processed_features

    except Exception as e:
        st.error(f"Error in prediction with explanations: {e}")
        raise e
