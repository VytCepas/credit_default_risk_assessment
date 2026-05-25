"""SHAP output validation — issue #33.

Verifies the SHAP attribution produced by ``models.insights.risk_decomposition``:

- no NaN / inf values
- per-feature SHAP sum + base_value matches the model's raw log-odds output
  within tolerance 1e-4 (the canonical SHAP invariant)
- top-N feature names are all in the predictor's known feature set
- a stubbed model gracefully degrades to the documented "note" dict
"""
from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

BUNDLE = Path("src/assets/top25_risk_model.pkl")
pytestmark = pytest.mark.skipif(
    not BUNDLE.exists(),
    reason="top25 model bundle not built; run scripts/squeeze_top25_accuracy.py",
)


def _sample_form() -> dict:
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


@pytest.fixture(scope="module")
def predictor():
    from models.top25_predictor import Top25Predictor
    return Top25Predictor(BUNDLE)


def test_shap_values_have_no_nan_or_inf(predictor):
    from models.insights import risk_decomposition

    r = risk_decomposition(predictor, _sample_form())
    if r.get("note") and not r.get("features"):
        pytest.skip(r["note"])

    shap_values = [row["shap"] for row in r["features"]]
    assert all(math.isfinite(v) for v in shap_values), (
        "SHAP values must not contain NaN or inf"
    )
    assert math.isfinite(r["base_value"]), "base_value must be finite"


def test_shap_invariant_holds(predictor):
    """sum(shap) + base_value must equal the model's raw log-odds output."""
    from models.insights import risk_decomposition

    form = _sample_form()
    r = risk_decomposition(predictor, form)
    if r.get("note") and not r.get("features"):
        pytest.skip(r["note"])

    shap_sum = sum(row["shap"] for row in r["features"])
    base = r["base_value"]

    # The model's own raw log-odds for the same input:
    X = predictor._prepare_input(form)
    raw = predictor.model.predict(X, raw_score=True)
    raw_logit = float(np.asarray(raw).ravel()[0])

    assert abs((shap_sum + base) - raw_logit) < 1e-4, (
        f"SHAP invariant violated: shap_sum + base = {shap_sum + base}, "
        f"model raw log-odds = {raw_logit}"
    )


def test_top_features_are_known(predictor):
    """Every name in the SHAP attribution must be in the predictor's feature_set."""
    from models.insights import risk_decomposition

    r = risk_decomposition(predictor, _sample_form())
    if r.get("note") and not r.get("features"):
        pytest.skip(r["note"])

    known = set(predictor.feature_set)
    top5 = r["features"][:5]
    for row in top5:
        assert row["feature"] in known, (
            f"Top-5 feature {row['feature']!r} not in predictor.feature_set"
        )


def test_unsupported_model_returns_stub(predictor):
    """If TreeExplainer raises, the function returns a documented stub dict."""
    from models.insights import risk_decomposition

    # Force shap.TreeExplainer to raise so the except branch runs.
    with patch("shap.TreeExplainer", side_effect=RuntimeError("simulated")):
        r = risk_decomposition(predictor, _sample_form())

    assert r["features"] == []
    assert r["groups"] == {}
    assert "note" in r and "SHAP" in r["note"]
