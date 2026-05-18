"""
Unit and integration tests for RiskModel prediction.

Tests that the loaded model returns valid predictions with correct
output structure, value ranges, and risk tier assignment.

Related issue: #53
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.risk_model import RiskModel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODEL_PATH = Path(__file__).parent.parent / "src" / "assets" / "risk_model.pkl"

VALID_INPUT = {
    "gender": "Male",
    "age": 35,
    "total_income": 150_000,
    "employment_status": "Working",
    "years_employed": 5,
    "education_level": "Higher education",
    "family_status": "Married",
    "num_children": 1,
    "num_family_members": 3,
    "owns_car": "Yes",
    "owns_housing": "Yes",
    "housing_type": "House / apartment",
    "contract_type": "Cash loans",
    "credit_amount": 500_000,
    "loan_annuity": 25_000,
}

HIGH_RISK_INPUT = {
    "gender": "Male",
    "age": 22,
    "total_income": 27_000,
    "employment_status": "Unemployed",
    "years_employed": 0,
    "education_level": "Lower secondary",
    "family_status": "Single / not married",
    "num_children": 3,
    "num_family_members": 4,
    "owns_car": "No",
    "owns_housing": "No",
    "housing_type": "Rented apartment",
    "contract_type": "Cash loans",
    "credit_amount": 900_000,
    "loan_annuity": 45_000,
}


@pytest.fixture(scope="module")
def model():
    """Load the trained model once for all tests in this module."""
    if not MODEL_PATH.exists():
        pytest.skip(f"Model file not found: {MODEL_PATH}")
    m = RiskModel()
    m.load(str(MODEL_PATH))
    return m


# ---------------------------------------------------------------------------
# Model state
# ---------------------------------------------------------------------------


def test_model_is_trained(model):
    assert model.is_trained is True


def test_threshold_in_expected_range(model):
    assert 0.25 <= model.optimal_threshold <= 0.50, (
        f"Threshold {model.optimal_threshold} outside expected range [0.25, 0.50]"
    )


def test_model_has_feature_names(model):
    assert model.feature_names is not None
    assert len(model.feature_names) > 0


# ---------------------------------------------------------------------------
# Prediction output structure
# ---------------------------------------------------------------------------


def test_predict_returns_dict(model):
    result = model.predict(VALID_INPUT)
    assert isinstance(result, dict)


def test_predict_has_required_keys(model):
    result = model.predict(VALID_INPUT)
    required = {"risk_probability", "risk_score", "risk_category"}
    assert required.issubset(result.keys())


# ---------------------------------------------------------------------------
# Value ranges
# ---------------------------------------------------------------------------


def test_risk_probability_in_zero_to_one(model):
    result = model.predict(VALID_INPUT)
    prob = result["risk_probability"]
    assert 0.0 <= prob <= 1.0, f"Probability {prob} outside [0, 1]"


def test_risk_score_in_zero_to_thousand(model):
    result = model.predict(VALID_INPUT)
    score = result["risk_score"]
    assert 0 <= score <= 1000, f"Risk score {score} outside [0, 1000]"


def test_risk_category_is_valid(model):
    result = model.predict(VALID_INPUT)
    assert result["risk_category"] in {"Low Risk", "Medium Risk", "High Risk"}


# ---------------------------------------------------------------------------
# Risk tier boundary logic
# ---------------------------------------------------------------------------


def test_risk_category_matches_score_tier(model):
    """Risk category must be consistent with the risk score value.

    Score boundaries (from risk_model.py):
      Low Risk    : score < 300
      Medium Risk : 300 <= score < 600
      High Risk   : score >= 600

    Note: the model assigns scores based on learned patterns in the training data,
    not on intuitive feature rankings. A 'safe-looking' applicant may still receive
    a high score if similar profiles defaulted at high rates historically.
    """
    result = model.predict(VALID_INPUT)
    score = result["risk_score"]
    category = result["risk_category"]

    if score < 300:
        assert category == "Low Risk", f"Score {score} should be Low Risk, got {category}"
    elif score < 600:
        assert category == "Medium Risk", f"Score {score} should be Medium Risk, got {category}"
    else:
        assert category == "High Risk", f"Score {score} should be High Risk, got {category}"


def test_two_different_profiles_produce_different_scores(model):
    """Two clearly different profiles must produce different risk scores."""
    result_1 = model.predict(VALID_INPUT)
    result_2 = model.predict(HIGH_RISK_INPUT)
    assert result_1["risk_score"] != result_2["risk_score"], (
        "Different profiles produced identical scores — model may not be discriminating"
    )


# ---------------------------------------------------------------------------
# Integration: full questionnaire → prediction pipeline
# ---------------------------------------------------------------------------


def test_full_pipeline_returns_integer_score(model):
    result = model.predict(VALID_INPUT)
    assert isinstance(result["risk_score"], int)


def test_full_pipeline_repeated_calls_consistent(model):
    """Same input must always produce the same output (deterministic inference)."""
    result_1 = model.predict(VALID_INPUT)
    result_2 = model.predict(VALID_INPUT)
    assert result_1["risk_score"] == result_2["risk_score"]
    assert result_1["risk_category"] == result_2["risk_category"]
