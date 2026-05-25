"""End-to-end integration test — questionnaire form → prediction → all insights.

Mirrors what ``app.py::show_result_page()`` does, without Streamlit. Closes
GitHub issue #32. Skipped if the production model bundle has not been built.
"""
from __future__ import annotations

from pathlib import Path

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


def test_full_pipeline_form_to_insights():
    """A single submission flows through prediction and every result-page surface."""
    from models import insights
    from models.top25_predictor import Top25Predictor

    predictor = Top25Predictor(BUNDLE)
    form = _sample_form()

    # 1. Core prediction
    result = predictor.predict(form)
    assert 0 <= result["risk_score"] <= 1000
    assert result["risk_category"] in {"Low", "Medium", "High"}
    assert 0.0 <= result["risk_probability"] <= 1.0

    # 2. Approval probability with CI (P-02)
    approval = insights.approval_with_confidence(predictor, form, n_bootstrap=5)
    assert approval["ci_lower"] <= approval["ci_upper"]
    assert approval["confidence_band"] in {"high", "medium", "low"}

    # 3. Risk decomposition (P-09) — SHAP attribution
    decomposition = insights.risk_decomposition(predictor, form)
    assert "features" in decomposition
    if decomposition.get("features"):
        assert len(decomposition["features"]) == len(predictor.feature_set)

    # 4. Cohort percentile (P-03) — degrades gracefully if precompute absent
    cohort_data = insights.load_precomputed("scripts/results/cohort_distributions.json")
    cohort = insights.cohort_percentile(
        {**form, "__risk_score": result["risk_score"]},
        cohort_data,
    )
    assert "percentile" in cohort

    # 5. Industry/region benchmark (P-04) — also degrades gracefully
    bench_data = insights.load_precomputed("scripts/results/industry_region_benchmarks.json")
    bench = insights.industry_region_benchmark(form, bench_data)
    assert isinstance(bench, dict)

    # 6. Approval process time (P-08)
    process = insights.approval_process_time(result["risk_category"])
    assert process["expected_time"] != "Unknown"

    # 7. Counter-factuals (P-01)
    cfs = insights.counter_factual_recommendations(predictor, form, top_n=3)
    assert "recommendations" in cfs

    # 8. Time-to-improvement (P-07) — for non-Low tiers; sanity check call surface
    if result["risk_category"] != "Low":
        tti = insights.time_to_improvement(predictor, form)
        assert tti.get("already_at_target") or "months_to_target" in tti


def test_predictor_output_keys_match_app_consumers():
    """Lock the predict() contract that ``app.py`` depends on."""
    from models.top25_predictor import Top25Predictor

    predictor = Top25Predictor(BUNDLE)
    result = predictor.predict(_sample_form())

    # Keys consumed in app.py:
    #   risk_probability, risk_score, risk_category, model_name, model_auc,
    #   plus the optional affordability_gate dict.
    required = {"risk_probability", "risk_score", "risk_category",
                "model_name", "model_auc"}
    assert required.issubset(result.keys()), (
        f"predict() must expose {required}, got {set(result.keys())}"
    )
