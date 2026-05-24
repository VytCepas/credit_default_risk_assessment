"""Unit tests for models.insights — ADR 0002 prediction surfaces."""
from __future__ import annotations

from pathlib import Path

import pytest

BUNDLE = Path("src/assets/top25_risk_model.pkl")
pytestmark = pytest.mark.skipif(
    not BUNDLE.exists(),
    reason="top25 model bundle not built; run scripts/squeeze_top25_accuracy.py",
)


@pytest.fixture(scope="module")
def predictor():
    from models.top25_predictor import Top25Predictor
    return Top25Predictor(BUNDLE)


@pytest.fixture
def low_risk_form():
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


@pytest.fixture
def high_risk_form():
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


# P-08 — approval process time
def test_p08_process_time_table():
    from models.insights import approval_process_time
    for tier in ("Low", "Medium", "High"):
        for completeness in ("complete", "incomplete"):
            r = approval_process_time(tier, completeness)
            assert r["tier"] == tier
            assert r["expected_time"] != "Unknown"


def test_p08_unknown_tier_returns_safe_fallback():
    from models.insights import approval_process_time
    r = approval_process_time("Unknown", "complete")
    assert r["expected_time"] == "Unknown"


# P-02 — approval probability with confidence
def test_p02_approval_with_confidence_shape(predictor, low_risk_form):
    from models.insights import approval_with_confidence
    r = approval_with_confidence(predictor, low_risk_form, n_bootstrap=5)
    assert {"approval_probability", "ci_lower", "ci_upper",
            "confidence_band", "approved"} <= set(r.keys())
    assert 0.0 <= r["approval_probability"] <= 1.0
    assert r["ci_lower"] <= r["ci_upper"]
    assert r["confidence_band"] in {"high", "medium", "low"}


def test_p02_low_risk_approved(predictor, low_risk_form):
    from models.insights import approval_with_confidence
    r = approval_with_confidence(predictor, low_risk_form, n_bootstrap=5)
    assert r["approved"], f"Low-risk form should approve, got proba {r['default_probability']}"


# P-05 — loan affordability curve
def test_p05_curve_shape(predictor, low_risk_form):
    from models.insights import loan_affordability_curve
    df = loan_affordability_curve(
        predictor, low_risk_form, amounts=[100_000, 500_000, 1_000_000],
    )
    assert len(df) == 3
    assert set(df.columns) == {"amount", "probability", "score", "tier"}


def test_p05_curve_monotone_increasing_in_risk(predictor, low_risk_form):
    from models.insights import loan_affordability_curve
    df = loan_affordability_curve(
        predictor, low_risk_form,
        amounts=[200_000, 500_000, 800_000, 1_200_000],
    )
    # bigger loans should generally not *reduce* risk much; allow tiny inversion noise
    risks = df["probability"].tolist()
    assert risks[-1] >= risks[0] - 0.02, f"Risk should grow with loan amount: {risks}"


# P-06 — recommended max loan
def test_p06_returns_valid_amount(predictor, low_risk_form):
    from models.insights import recommended_max_loan
    r = recommended_max_loan(predictor, low_risk_form, max_iter=15)
    assert r["projected_tier"] in {"Low", "Medium", "High"}
    if r["amount"] is not None:
        assert r["amount"] >= 50_000


# P-01 — counter-factual recommendations
def test_p01_counter_factuals(predictor, high_risk_form):
    from models.insights import counter_factual_recommendations
    r = counter_factual_recommendations(predictor, high_risk_form, top_n=3)
    assert "recommendations" in r
    assert len(r["recommendations"]) <= 3
    for rec in r["recommendations"]:
        assert rec["delta"] < 0, "Recommendations must lower the score"


# P-07 — time to improvement
def test_p07_low_risk_already_at_target(predictor, low_risk_form):
    from models.insights import time_to_improvement
    r = time_to_improvement(predictor, low_risk_form, target_tier="Low")
    assert r["already_at_target"] or r["months_to_target"] is not None


# P-09 — risk decomposition
def test_p09_decomposition_sums_to_100(predictor, low_risk_form):
    from models.insights import risk_decomposition
    r = risk_decomposition(predictor, low_risk_form)
    groups = r.get("groups", {})
    if groups:
        total = sum(groups.values())
        assert abs(total - 100.0) < 1.0, f"Group percentages should sum to ~100, got {total}"


# P-03 — cohort percentile (skips if precompute hasn't run)
def test_p03_cohort_percentile_returns_well_formed():
    from models.insights import load_precomputed, cohort_percentile
    d = load_precomputed("scripts/results/cohort_distributions.json")
    if not d:
        pytest.skip("cohort distributions not precomputed")
    r = cohort_percentile(
        {"age_years": 35, "total_income": 150_000, "__risk_score": 200}, d,
    )
    assert "percentile" in r
    if r["percentile"] is not None:
        assert 0 <= r["percentile"] <= 100


# P-04 — industry/region benchmark
def test_p04_benchmark_shape():
    from models.insights import load_precomputed, industry_region_benchmark
    b = load_precomputed("scripts/results/industry_region_benchmarks.json")
    if not b:
        pytest.skip("benchmarks not precomputed")
    r = industry_region_benchmark(
        {"organization_type": "Business Entity Type 3", "city_rating": 2}, b,
    )
    assert "industry_rate" in r or r == {}
