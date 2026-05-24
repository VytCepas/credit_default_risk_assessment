"""Unit tests for the Stage-2 squeeze inference wrapper.

Skipped if the model bundle has not been built — see
``scripts/squeeze_top25_accuracy.py``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

BUNDLE = Path("src/assets/top25_risk_model.pkl")
pytestmark = pytest.mark.skipif(
    not BUNDLE.exists(),
    reason="top25 model bundle not built; run scripts/squeeze_top25_accuracy.py",
)


def _low_risk_profile() -> dict:
    return dict(
        gender="Male", age_years=35, num_children=2, num_family_members=4,
        family_status="Married", years_employed=8.0,
        organization_type="Business Entity Type 3", occupation_type="Managers",
        has_work_phone=True, contract_type="Cash loans",
        credit_amount=400_000, loan_annuity=20_000, goods_price=400_000,
        total_income=180_000, owns_car=True, car_age_years=5, owns_housing=True,
        years_since_id_change=4, years_at_address=7,
        region_population_relative=0.019, city_rating=2,
        works_in_different_city=False, has_landline=True,
    )


def _high_risk_profile() -> dict:
    return dict(
        gender="Female", age_years=22, num_children=0, num_family_members=1,
        family_status="Single / not married", years_employed=0.5,
        organization_type="Self-employed", occupation_type="Low-skill Laborers",
        has_work_phone=False, contract_type="Cash loans",
        credit_amount=1_500_000, loan_annuity=80_000, goods_price=0,
        total_income=40_000, owns_car=False, car_age_years=None, owns_housing=False,
        years_since_id_change=0.5, years_at_address=0.5,
        region_population_relative=0.013, city_rating=3,
        works_in_different_city=True, has_landline=False,
    )


def test_bundle_loads_and_reports_metadata():
    from models.top25_predictor import Top25Predictor

    p = Top25Predictor(BUNDLE)
    assert p.best_auc > 0.70, f"Bundle AUC unexpectedly low: {p.best_auc}"
    assert len(p.feature_set) == 31  # 25 selected + 6 derived ratios
    assert "dti" in p.derived_ratios
    assert "credit_to_income" in p.derived_ratios


def test_predict_returns_well_formed_result():
    from models.top25_predictor import Top25Predictor

    p = Top25Predictor(BUNDLE)
    result = p.predict(_low_risk_profile())
    assert {"risk_probability", "risk_score", "risk_category",
            "model_name", "model_auc"} <= set(result.keys())
    assert 0.0 <= result["risk_probability"] <= 1.0
    assert 0 <= result["risk_score"] <= 1000
    assert result["risk_category"] in {"Low", "Medium", "High"}


def test_low_vs_high_risk_differentiation():
    """Strong-profile applicant should score lower than weak-profile."""
    from models.top25_predictor import Top25Predictor

    p = Top25Predictor(BUNDLE)
    low = p.predict(_low_risk_profile())
    high = p.predict(_high_risk_profile())
    assert low["risk_probability"] < high["risk_probability"], (
        f"Differentiation failed: low {low['risk_probability']:.3f} >= "
        f"high {high['risk_probability']:.3f}"
    )


def test_predict_idempotent_for_same_input():
    from models.top25_predictor import Top25Predictor

    p = Top25Predictor(BUNDLE)
    profile = _low_risk_profile()
    r1 = p.predict(profile)
    r2 = p.predict(profile)
    assert r1["risk_score"] == r2["risk_score"]
    assert r1["risk_probability"] == r2["risk_probability"]


def test_handles_missing_optional_field():
    """Form keys may be absent (e.g., car_age when owns_car is False)."""
    from models.top25_predictor import Top25Predictor

    p = Top25Predictor(BUNDLE)
    profile = _low_risk_profile()
    profile.pop("car_age_years")  # simulate missing optional
    profile["owns_car"] = False
    r = p.predict(profile)
    assert 0 <= r["risk_score"] <= 1000
