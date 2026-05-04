import streamlit as st
import pickle
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class BehavioralTraitsPredictor:
    def __init__(self):
        self.model = None
        self.is_loaded = False

    def load_model(self, model_path: str) -> bool:
        """Load the behavioral traits model from pickle file."""
        try:
            model_file = Path(model_path)
            if not model_file.exists():
                logger.error("Behavioral traits model file not found: %s", model_file)
                self.is_loaded = False
                return False

            with open(model_file, "rb") as f:
                self.model = pickle.load(f)
            self.is_loaded = True
            logger.info(
                "Behavioral traits model loaded successfully from %s", model_file
            )
            return True
        except (
            FileNotFoundError,
            OSError,
            pickle.UnpicklingError,
            AttributeError,
        ) as e:
            logger.error("Failed to load behavioral traits model: %s", e)
            self.is_loaded = False
            return False

    def predict_traits(self, questionnaire_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict behavioral traits based on questionnaire data."""
        if not self.is_loaded or self.model is None:
            return {"error": "Behavioral traits model not loaded"}

        try:
            model_data = self._transform_questionnaire_data(questionnaire_data)

            if hasattr(self.model, "predict_traits"):
                results = self.model.predict_traits(model_data)
            else:
                results = self._fallback_prediction(model_data)

            return results

        except (TypeError, ValueError, AttributeError, KeyError) as e:
            logger.error("Error predicting behavioral traits: %s", e)
            return {"error": f"Prediction failed: {str(e)}"}

    def _transform_questionnaire_data(
        self, questionnaire_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Transform questionnaire data to format expected by behavioral traits model."""
        model_data = {}

        field_mapping = {
            "age": "age",
            "total_income": "amt_income_total",
            "years_employed": "days_employed",  # Will convert to negative days
            "education_level": "name_education_type",
            "family_status": "name_family_status",
            "housing_type": "name_housing_type",
            "employment_status": "organization_type",
            "num_children": "cnt_children",
            "owns_car": "flag_own_car",
            "owns_housing": "flag_own_realty",
        }

        for questionnaire_field, model_field in field_mapping.items():
            if questionnaire_field in questionnaire_data:
                value = questionnaire_data[questionnaire_field]

                if model_field == "days_employed":
                    model_data[model_field] = -int(value * 365) if value > 0 else 0
                elif model_field == "days_birth":
                    age = questionnaire_data.get("age", 30)
                    model_data[model_field] = -int(age * 365)
                elif model_field in ["flag_own_car", "flag_own_realty"]:
                    if isinstance(value, bool):
                        model_data[model_field] = "Y" if value else "N"
                    elif isinstance(value, str):
                        model_data[model_field] = (
                            "Y" if value.lower() in ["yes", "true", "y"] else "N"
                        )
                    else:
                        model_data[model_field] = "N"
                elif model_field == "organization_type":
                    org_mapping = {
                        "Working": "Business Entity Type 3",
                        "State servant": "Government",
                        "Commercial associate": "Business Entity Type 1",
                        "Pensioner": "Government",
                        "Student": "Business Entity Type 3",
                        "Businessman": "Business Entity Type 2",
                        "Unemployed": "Business Entity Type 3",
                    }
                    model_data[model_field] = org_mapping.get(
                        value, "Business Entity Type 3"
                    )
                else:
                    model_data[model_field] = value

        defaults = {
            "amt_credit": questionnaire_data.get("total_income", 150000) * 2,
            "amt_annuity": questionnaire_data.get("total_income", 150000) * 0.05,
            "days_birth": -int(questionnaire_data.get("age", 30) * 365),
        }

        for field, default_value in defaults.items():
            if field not in model_data:
                model_data[field] = default_value

        return model_data

    def _fallback_prediction(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback prediction method if model doesn't have predict_traits method."""
        age = model_data.get("age", 30)
        income = model_data.get("amt_income_total", 150000)
        employment_days = abs(
            model_data.get("days_employed", -1825)
        )  # Convert back to positive
        education = model_data.get(
            "name_education_type", "Secondary / secondary special"
        )

        job_stability = min(
            100,
            max(
                0,
                (employment_days / 365) * 15  # Years employed * 15
                + (min(income, 500000) / 5000),  # Income factor
            ),
        )

        payment_behavior = min(
            100,
            max(
                0,
                (income / 2000)  # Income factor
                + (age - 18) * 1.5  # Age factor
                + (20 if "Higher" in education else 0),  # Education bonus
            ),
        )

        responsibility = min(
            100,
            max(
                0,
                age * 1.2  # Age is key factor
                + (30 if "Higher" in education else 0)  # Education bonus
                + (
                    10 if model_data.get("flag_own_realty") == "Y" else 0
                ),  # Asset ownership
            ),
        )

        overall_score = (job_stability + payment_behavior + responsibility) / 3

        return {
            "job_stability": round(job_stability, 1),
            "payment_behavior": round(payment_behavior, 1),
            "responsibility": round(responsibility, 1),
            "overall_behavioral_score": round(overall_score, 1),
        }


@st.cache_resource(show_spinner=False)
def load_behavioral_predictor(
    model_path: str,
) -> BehavioralTraitsPredictor:
    """Load and return behavioral traits predictor instance."""
    predictor = BehavioralTraitsPredictor()
    predictor.load_model(model_path)
    return predictor


def predict_behavioral_traits(questionnaire_data: Dict[str, Any]) -> Dict[str, Any]:
    """Predict behavioral traits using the loaded model."""
    default_model_path = (
        Path(__file__).parent.parent / "assets" / "behavioral_traits_model.pkl"
    )

    if not default_model_path.exists():
        logger.warning("Behavioral traits model file not found: %s", default_model_path)
        return {"error": f"Behavioral traits model not found at {default_model_path}"}

    predictor = load_behavioral_predictor(str(default_model_path))
    return predictor.predict_traits(questionnaire_data)


def get_available_behavioral_models() -> Dict[str, Dict[str, Any]]:
    """Return available behavioral traits model artifacts.

    The structure mirrors get_available_models() in risk_predictor.py so the
    sidebar can render both model groups consistently.
    """
    assets_dir = Path(__file__).parent.parent / "assets"

    potential_models = {
        "behavioral_traits_model": {
            "path": assets_dir / "behavioral_traits_model.pkl",
            "display_name": "Behavioral Traits Model",
            "description": "Analyzes job stability, payment behavior, and financial responsibility traits",
            "icon": "🎭",
            "type": "behavioral",
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
                "type": model_info["type"],
            }

    return available_models
