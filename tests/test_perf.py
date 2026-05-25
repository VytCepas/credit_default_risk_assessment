"""Performance NFR — issue #18.

The brief asks for P95 latency < 3 s on Streamlit Cloud. CI runs on a GitHub
Actions ubuntu-latest runner — that is not the production environment, so we
make the *local* statement instead: a single prediction (model already in
memory) takes well under the budget. Streamlit Cloud E2E is verified
manually before defence.

Measured locally as of 2026-05-25: ~0.005 s per prediction; the 3 s budget
includes form serialisation, network, and rendering — all comfortably out of
the model's hands.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

BUNDLE = Path("src/assets/top25_risk_model.pkl")
pytestmark = pytest.mark.skipif(
    not BUNDLE.exists(),
    reason="top25 model bundle not built; run scripts/squeeze_top25_accuracy.py",
)

N_REQUESTS = 20
P95_BUDGET_SECONDS = 3.0


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


def test_p95_prediction_latency_under_budget():
    """20 consecutive predictions; assert P95 well under 3 s.

    Measures the predictor only — Streamlit and SHAP overhead are out of
    scope here (SHAP is covered separately by tests/test_shap_validation.py).
    """
    from models.top25_predictor import Top25Predictor

    predictor = Top25Predictor(BUNDLE)
    form = _sample_form()

    # warm-up — first call hits any lazy initialisation paths
    predictor.predict(form)

    latencies: list[float] = []
    for _ in range(N_REQUESTS):
        t0 = time.perf_counter()
        predictor.predict(form)
        latencies.append(time.perf_counter() - t0)

    latencies.sort()
    p95_index = max(0, int(0.95 * N_REQUESTS) - 1)
    p95 = latencies[p95_index]

    assert p95 < P95_BUDGET_SECONDS, (
        f"P95 latency {p95:.3f}s exceeds budget {P95_BUDGET_SECONDS}s "
        f"(mean {sum(latencies)/len(latencies):.3f}s)"
    )
