"""
Unit tests for QuestionnaireToFeatures transformer.

Tests that the transformer correctly maps raw questionnaire dict inputs
to the 15-column feature DataFrame expected by the preprocessing pipeline.

Related issue: #53
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.risk_model import QuestionnaireToFeatures  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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


@pytest.fixture
def transformer():
    return QuestionnaireToFeatures()


# ---------------------------------------------------------------------------
# Output type and shape
# ---------------------------------------------------------------------------


def test_transform_returns_dataframe(transformer):
    result = transformer.transform(VALID_INPUT)
    assert isinstance(result, pd.DataFrame)


def test_transform_has_all_expected_columns(transformer):
    result = transformer.transform(VALID_INPUT)
    expected = {
        "gender", "age_years", "total_income", "income_type",
        "years_employed", "education_level", "family_status",
        "num_children", "num_family_members", "owns_car",
        "owns_housing", "housing_type", "contract_type",
        "credit_amount", "loan_annuity",
    }
    assert expected.issubset(set(result.columns))


# ---------------------------------------------------------------------------
# Binary field encoding
# ---------------------------------------------------------------------------


def test_gender_male_encoded_as_1(transformer):
    result = transformer.transform(VALID_INPUT)
    assert result["gender"].iloc[0] == 1


def test_gender_female_encoded_as_0(transformer):
    female_input = {**VALID_INPUT, "gender": "Female"}
    result = transformer.transform(female_input)
    assert result["gender"].iloc[0] == 0


def test_owns_car_yes_encoded_as_1(transformer):
    result = transformer.transform(VALID_INPUT)
    assert result["owns_car"].iloc[0] == 1


def test_owns_car_no_encoded_as_0(transformer):
    no_car_input = {**VALID_INPUT, "owns_car": "No"}
    result = transformer.transform(no_car_input)
    assert result["owns_car"].iloc[0] == 0


def test_owns_housing_yes_encoded_as_1(transformer):
    result = transformer.transform(VALID_INPUT)
    assert result["owns_housing"].iloc[0] == 1


def test_contract_type_cash_encoded_as_1(transformer):
    result = transformer.transform(VALID_INPUT)
    assert result["contract_type"].iloc[0] == 1


def test_contract_type_revolving_encoded_as_0(transformer):
    revolving_input = {**VALID_INPUT, "contract_type": "Revolving loans"}
    result = transformer.transform(revolving_input)
    assert result["contract_type"].iloc[0] == 0


# ---------------------------------------------------------------------------
# Numerical field passthrough
# ---------------------------------------------------------------------------


def test_income_passes_through(transformer):
    result = transformer.transform(VALID_INPUT)
    assert result["total_income"].iloc[0] == 150_000


def test_credit_amount_passes_through(transformer):
    result = transformer.transform(VALID_INPUT)
    assert result["credit_amount"].iloc[0] == 500_000


def test_loan_annuity_passes_through(transformer):
    result = transformer.transform(VALID_INPUT)
    assert result["loan_annuity"].iloc[0] == 25_000


def test_num_children_passes_through(transformer):
    result = transformer.transform(VALID_INPUT)
    assert result["num_children"].iloc[0] == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_zero_children_allowed(transformer):
    zero_children = {**VALID_INPUT, "num_children": 0}
    result = transformer.transform(zero_children)
    assert result["num_children"].iloc[0] == 0


def test_missing_loan_annuity_becomes_nan(transformer):
    no_annuity = {**VALID_INPUT, "loan_annuity": 0}
    result = transformer.transform(no_annuity)
    assert pd.isna(result["loan_annuity"].iloc[0])


def test_unknown_key_ignored(transformer):
    extra_key_input = {**VALID_INPUT, "unknown_field": "ignored_value"}
    result = transformer.transform(extra_key_input)
    assert "unknown_field" not in result.columns
