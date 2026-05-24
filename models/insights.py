"""Prediction-surface helpers (ADR 0002 — Additional Prediction Surfaces).

Each function takes a fitted :class:`models.top25_predictor.Top25Predictor`
and a user-form dict, and returns a small JSON-friendly dict that the
Streamlit Insights section can render directly.

Implementations covered (P-01 … P-09 from ADR 0002 / GitHub issues #75-#83):

- ``counter_factual_recommendations``         — P-01
- ``approval_with_confidence``                — P-02
- ``cohort_percentile``                       — P-03
- ``industry_region_benchmark``               — P-04
- ``loan_affordability_curve``                — P-05
- ``recommended_max_loan``                    — P-06
- ``time_to_improvement``                     — P-07
- ``approval_process_time``                   — P-08
- ``risk_decomposition``                      — P-09

P-10 (time-to-default survival analysis) is gated on the bureau-data
integration (#72) and lives in a separate module when delivered.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Static configuration
# ---------------------------------------------------------------------------

#: Form keys whose values an applicant could realistically change.
#: Counter-factuals are restricted to these (no gender, no age, no kids).
MUTABLE_FORM_KEYS: set[str] = {
    "years_employed",
    "organization_type",
    "occupation_type",
    "has_work_phone",
    "credit_amount",
    "loan_annuity",
    "goods_price",
    "total_income",
    "owns_car",
    "car_age_years",
    "owns_housing",
    "years_since_id_change",
    "years_at_address",
    "city_rating",
    "works_in_different_city",
    "has_landline",
    "contract_type",
}

#: SHAP-grouping for P-09 risk decomposition. Maps Kaggle column name → group label.
FEATURE_GROUPS: dict[str, str] = {
    # Employment
    "days_employed": "Employment",
    "organization_type": "Employment",
    "occupation_type": "Employment",
    "flag_work_phone": "Employment",
    "years_employed_ratio": "Employment",
    # Loan
    "amt_credit": "Loan size",
    "amt_annuity": "Loan size",
    "amt_goods_price": "Loan size",
    "name_contract_type": "Loan size",
    "credit_to_income": "Loan size",
    "annuity_to_credit": "Loan size",
    "credit_to_goods": "Loan size",
    "dti": "Loan size",
    # Financial
    "amt_income_total": "Financial",
    "income_per_family_member": "Financial",
    # Demographics
    "code_gender": "Demographics",
    "days_birth": "Demographics",
    "name_family_status": "Demographics",
    "cnt_children": "Demographics",
    "cnt_fam_members": "Demographics",
    # Assets
    "flag_own_car": "Assets",
    "own_car_age": "Assets",
    "flag_own_realty": "Assets",
    # Residence / context
    "days_id_publish": "Residence",
    "days_registration": "Residence",
    "region_population_relative": "Residence",
    "region_rating_client_w_city": "Residence",
    "reg_city_not_work_city": "Residence",
    "hour_appr_process_start": "Other",
    "weekday_appr_process_start": "Other",
    "flag_phone": "Other",
}

#: Suggested improved-value rules for counter-factuals (P-01).
#: For each mutable form key, a function (current_value, full_form) → suggested_value.
_PERTURBATIONS: dict[str, Any] = {
    "years_employed": lambda v, f: min(
        float(v or 0) + 3.0,
        max(float(f.get("age_years", 30)) - 16, 1),
    ),
    "years_at_address": lambda v, f: float(v or 0) + 3.0,
    "years_since_id_change": lambda v, f: float(v or 0) + 5.0,
    "credit_amount": lambda v, f: max(float(v or 0) * 0.75, 50_000),
    "loan_annuity": lambda v, f: max(float(v or 0) * 0.75, 5_000),
    "total_income": lambda v, f: float(v or 0) * 1.25,
}

#: Approval-process time lookup table for P-08.
_PROCESS_TIME_TABLE: dict[tuple[str, str], str] = {
    ("Low", "complete"):      "Instant (≈ 2 minutes)",
    ("Low", "incomplete"):    "Same business day",
    ("Medium", "complete"):   "1–2 business days",
    ("Medium", "incomplete"): "3–5 business days",
    ("High", "complete"):     "5 business days (manual review)",
    ("High", "incomplete"):   "7+ business days",
}


# ---------------------------------------------------------------------------
# P-08 — approval-process time (simplest; no dependencies)
# ---------------------------------------------------------------------------
def approval_process_time(risk_tier: str, completeness: str = "complete") -> dict:
    """Return the expected wait time for the user's risk tier.

    Parameters
    ----------
    risk_tier
        One of ``"Low"``, ``"Medium"``, ``"High"``.
    completeness
        ``"complete"`` if every form field is answered, ``"incomplete"`` otherwise.
    """
    key = (risk_tier, completeness)
    if key not in _PROCESS_TIME_TABLE:
        return {"expected_time": "Unknown", "tier": risk_tier, "completeness": completeness}
    return {
        "expected_time": _PROCESS_TIME_TABLE[key],
        "tier": risk_tier,
        "completeness": completeness,
    }


# ---------------------------------------------------------------------------
# P-02 — approval probability with confidence band
# ---------------------------------------------------------------------------
def approval_with_confidence(
    predictor,
    form: dict,
    n_bootstrap: int = 50,
    approval_threshold: float = 0.50,
    perturbation_sigma: float = 0.05,
    seed: int = 0,
) -> dict:
    """Approval decision + bootstrap confidence interval on the probability.

    Bootstraps by perturbing numeric inputs with relative gaussian noise
    (``sigma * value``) and re-predicting; this is a fast proxy for
    re-training-bootstrap that fits within an interactive request.
    """
    rng = np.random.default_rng(seed)
    base = predictor.predict(form)
    base_proba = base["risk_probability"]

    probas: list[float] = [base_proba]
    numeric_keys = [
        "age_years", "total_income", "credit_amount", "loan_annuity",
        "goods_price", "years_employed", "years_at_address",
        "years_since_id_change", "num_family_members", "num_children",
        "car_age_years",
    ]
    for _ in range(n_bootstrap):
        perturbed = dict(form)
        for k in numeric_keys:
            if k in perturbed and perturbed[k] not in (None, ""):
                try:
                    cur = float(perturbed[k])
                except (TypeError, ValueError):
                    continue
                if cur == 0:
                    continue
                noise = float(rng.normal(loc=1.0, scale=perturbation_sigma))
                perturbed[k] = max(0.0, cur * noise)
        try:
            probas.append(predictor.predict(perturbed)["risk_probability"])
        except Exception:
            continue

    arr = np.asarray(probas)
    lo, hi = float(np.percentile(arr, 5)), float(np.percentile(arr, 95))
    width = hi - lo

    if width < 0.10:
        band = "high"
    elif width < 0.20:
        band = "medium"
    else:
        band = "low"

    approved = base_proba < approval_threshold
    return {
        "approval_probability": float(1 - base_proba),  # P(no default) = approval %
        "default_probability": float(base_proba),
        "ci_lower": float(1 - hi),
        "ci_upper": float(1 - lo),
        "ci_width_proba": float(width),
        "confidence_band": band,
        "approved": bool(approved),
        "approval_threshold": approval_threshold,
    }


# ---------------------------------------------------------------------------
# P-05 — loan-affordability curve
# ---------------------------------------------------------------------------
def loan_affordability_curve(
    predictor,
    form: dict,
    amounts: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Re-score the user across a range of `credit_amount` values."""
    if amounts is None:
        cur = float(form.get("credit_amount") or 500_000)
        amounts = np.linspace(max(cur * 0.2, 50_000), cur * 2.0, 20)
    amounts = list(amounts)

    rows = []
    for amt in amounts:
        scenario = dict(form)
        scenario["credit_amount"] = float(amt)
        # Scale annuity proportionally to keep the ratio realistic
        original_credit = float(form.get("credit_amount") or amt)
        if original_credit > 0:
            ratio = float(amt) / original_credit
            base_ann = float(form.get("loan_annuity") or 25_000)
            scenario["loan_annuity"] = base_ann * ratio
        r = predictor.predict(scenario)
        rows.append({
            "amount": float(amt),
            "probability": r["risk_probability"],
            "score": r["risk_score"],
            "tier": r["risk_category"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# P-06 — recommended max loan via binary search
# ---------------------------------------------------------------------------
def recommended_max_loan(
    predictor,
    form: dict,
    target_tier: str = "Low",
    lo: float = 50_000,
    hi: float = 5_000_000,
    tol: float = 10_000,
    max_iter: int = 30,
) -> dict:
    """Largest `credit_amount` that keeps the applicant in `target_tier`."""
    tier_order = ["Low", "Medium", "High"]

    def _within(amount: float) -> tuple[bool, dict]:
        s = dict(form)
        s["credit_amount"] = float(amount)
        ratio = amount / max(float(form.get("credit_amount") or amount), 1.0)
        s["loan_annuity"] = float(form.get("loan_annuity") or 25_000) * ratio
        r = predictor.predict(s)
        idx = tier_order.index(r["risk_category"])
        return idx <= tier_order.index(target_tier), r

    feasible_at_lo, lo_result = _within(lo)
    if not feasible_at_lo:
        return {
            "amount": None,
            "projected_score": lo_result["risk_score"],
            "projected_tier": lo_result["risk_category"],
            "search_iterations": 1,
            "note": f"Not feasible at minimum {lo}; profile cannot reach {target_tier}.",
        }
    feasible_at_hi, hi_result = _within(hi)
    if feasible_at_hi:
        return {
            "amount": float(hi),
            "projected_score": hi_result["risk_score"],
            "projected_tier": hi_result["risk_category"],
            "search_iterations": 2,
            "note": "Headroom exceeds search ceiling.",
        }

    iters = 2
    best_amount = lo
    best_result = lo_result
    while hi - lo > tol and iters < max_iter:
        mid = (lo + hi) / 2.0
        feasible, result = _within(mid)
        iters += 1
        if feasible:
            best_amount = mid
            best_result = result
            lo = mid
        else:
            hi = mid
    return {
        "amount": float(best_amount),
        "projected_score": best_result["risk_score"],
        "projected_tier": best_result["risk_category"],
        "search_iterations": iters,
        "note": "Indicative; not a formal affordability assessment.",
    }


# ---------------------------------------------------------------------------
# P-01 — counter-factual recommendations (SHAP-driven)
# ---------------------------------------------------------------------------
def counter_factual_recommendations(predictor, form: dict, top_n: int = 3) -> dict:
    """Suggest realistic changes to top SHAP-negative features.

    SHAP-driven candidate selection is approximated here as a one-step
    perturbation search across the whitelisted mutable form keys —
    avoids the overhead of fitting a TreeExplainer at request time for
    the stacking/calibrated wrappers that don't expose ``base_model``.
    """
    base = predictor.predict(form)
    candidates = []
    for key in MUTABLE_FORM_KEYS:
        if key not in _PERTURBATIONS:
            continue
        if key not in form or form[key] in (None, ""):
            continue
        try:
            cur = float(form[key]) if not isinstance(form[key], bool) else int(form[key])
        except (TypeError, ValueError):
            continue
        suggested = _PERTURBATIONS[key](form[key], form)
        if suggested == cur:
            continue
        scenario = dict(form)
        scenario[key] = suggested
        try:
            new_r = predictor.predict(scenario)
        except Exception:
            continue
        delta = new_r["risk_score"] - base["risk_score"]
        if delta >= 0:  # we want improvements (lower score)
            continue
        candidates.append({
            "feature": key,
            "current_value": form[key],
            "suggested_value": suggested,
            "current_score": base["risk_score"],
            "projected_score": new_r["risk_score"],
            "delta": delta,
        })
    candidates.sort(key=lambda c: c["delta"])
    return {
        "current_score": base["risk_score"],
        "current_tier": base["risk_category"],
        "recommendations": candidates[:top_n],
    }


# ---------------------------------------------------------------------------
# P-07 — time-to-improvement projection
# ---------------------------------------------------------------------------
def time_to_improvement(
    predictor,
    form: dict,
    target_tier: str = "Low",
    max_months: int = 120,
) -> dict:
    """Project the form forward in monthly increments until target tier reached."""
    tier_order = ["Low", "Medium", "High"]
    target_idx = tier_order.index(target_tier)

    def _idx(form_state: dict) -> int:
        return tier_order.index(predictor.predict(form_state)["risk_category"])

    if _idx(form) <= target_idx:
        return {
            "months_to_target": 0,
            "already_at_target": True,
            "target_tier": target_tier,
        }

    projected = dict(form)
    for month in range(1, max_months + 1):
        years = 1.0 / 12.0
        if projected.get("years_employed") is not None:
            projected["years_employed"] = float(projected["years_employed"]) + years
        if projected.get("years_at_address") is not None:
            projected["years_at_address"] = float(projected["years_at_address"]) + years
        if projected.get("years_since_id_change") is not None:
            projected["years_since_id_change"] = (
                float(projected["years_since_id_change"]) + years
            )
        if _idx(projected) <= target_idx:
            projected_score = predictor.predict(projected)["risk_score"]
            return {
                "months_to_target": month,
                "already_at_target": False,
                "target_tier": target_tier,
                "projected_score": projected_score,
                "caveat": "Assumes current trajectory; not a promise.",
            }
    return {
        "months_to_target": None,
        "already_at_target": False,
        "target_tier": target_tier,
        "max_months_searched": max_months,
        "note": f"Target {target_tier} tier not reached within {max_months} months.",
    }


# ---------------------------------------------------------------------------
# P-09 — risk decomposition by feature group (uses SHAP if available)
# ---------------------------------------------------------------------------
def risk_decomposition(predictor, form: dict) -> dict:
    """Per-applicant SHAP attribution for the predicted probability.

    Returns signed per-feature SHAP values (positive = pushes toward default,
    negative = pushes away) plus a friendlier "group totals" view that bins
    features into the categories declared in :data:`FEATURE_GROUPS`.

    Falls back to a stub dict if SHAP cannot be constructed for the model
    type (only tree-based models are supported by ``shap.TreeExplainer``).
    """
    try:
        import shap
    except ImportError:
        return {
            "features": [],
            "groups": {},
            "note": "shap is not installed.",
        }

    model = predictor.model
    if not hasattr(model, "predict_proba") or model.__class__.__name__ not in {
        "LGBMClassifier",
        "XGBClassifier",
        "CatBoostClassifier",
        "GradientBoostingClassifier",
        "RandomForestClassifier",
    }:
        return {
            "features": [],
            "groups": {},
            "note": (
                f"SHAP attribution is only computed for tree-based models; "
                f"this bundle is {model.__class__.__name__}."
            ),
        }

    X = predictor._prepare_input(form)
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
        # LightGBM binary returns either (1, n_features) or list of 2 arrays.
        if isinstance(sv, list):
            sv = sv[1] if len(sv) > 1 else sv[0]
        contributions = np.asarray(sv).reshape(-1)
        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(np.asarray(base_value).ravel()[-1])
        else:
            base_value = float(base_value)
    except Exception as exc:  # noqa: BLE001
        return {
            "features": [],
            "groups": {},
            "note": f"SHAP computation failed: {exc!s}",
        }

    feature_rows = []
    group_totals: dict[str, float] = {}
    for col, contrib, value in zip(
        predictor.feature_set, contributions, X.iloc[0].tolist()
    ):
        feature_rows.append(
            {
                "feature": col,
                "shap": float(contrib),
                "value": float(value) if isinstance(value, (int, float, np.floating)) else value,
                "group": FEATURE_GROUPS.get(col, "Other"),
            }
        )
        group_totals.setdefault(FEATURE_GROUPS.get(col, "Other"), 0.0)
        group_totals[FEATURE_GROUPS.get(col, "Other")] += float(contrib)

    feature_rows.sort(key=lambda r: abs(r["shap"]), reverse=True)
    return {
        "method": "SHAP (TreeExplainer, log-odds units)",
        "base_value": base_value,
        "features": feature_rows,
        "groups": {g: round(v, 4) for g, v in group_totals.items()},
    }


# ---------------------------------------------------------------------------
# P-03 — cohort comparison percentile
# P-04 — industry / region benchmark
# (data-driven; rely on artefacts produced by scripts/precompute_insights.py)
# ---------------------------------------------------------------------------
def load_precomputed(artefact_path: str | Path) -> dict:
    """Load a precomputed-insight JSON artefact."""
    p = Path(artefact_path)
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def cohort_percentile(form: dict, distributions: dict) -> dict:
    """Return user's percentile within the matching cohort.

    Expects ``distributions`` shape::

        {
          "cohorts": {
              "age=25-34|income=100k-200k": {
                  "n": 12345,
                  "score_quantiles": {"5": 30, "25": 90, "50": 160, "75": 280, "95": 420}
              }, ...
          },
          "fallback": {"n": 307511, "score_quantiles": {...}}
        }
    """
    if not distributions:
        return {"cohort_label": None, "percentile": None, "n_in_cohort": 0}

    cohort = _match_cohort(form, distributions)
    if cohort is None:
        cohort = distributions.get("fallback", {})

    user_score_proxy = float(form.get("__risk_score", 0))
    if not cohort or not cohort.get("score_quantiles"):
        return {"cohort_label": "all-applicants", "percentile": None, "n_in_cohort": 0}

    quantiles = {int(k): float(v) for k, v in cohort["score_quantiles"].items()}
    sorted_pcts = sorted(quantiles.keys())
    pct = sorted_pcts[0]
    for p in sorted_pcts:
        if user_score_proxy >= quantiles[p]:
            pct = p
    return {
        "cohort_label": cohort.get("label", "matched"),
        "percentile": int(pct),
        "n_in_cohort": int(cohort.get("n", 0)),
        "interpretation": (
            "Your risk score is lower than approximately "
            f"{100 - pct}% of applicants in this cohort."
        ),
    }


def _match_cohort(form: dict, distributions: dict) -> dict | None:
    """Bucket the user into the cohort key with the highest n_in_cohort fit."""
    age = float(form.get("age_years") or 30)
    income = float(form.get("total_income") or 100_000)

    age_bucket = (
        "18-24" if age < 25 else
        "25-34" if age < 35 else
        "35-44" if age < 45 else
        "45-54" if age < 55 else
        "55+"
    )
    income_bucket = (
        "0-50k" if income < 50_000 else
        "50k-100k" if income < 100_000 else
        "100k-200k" if income < 200_000 else
        "200k+"
    )
    key = f"age={age_bucket}|income={income_bucket}"
    cohorts = distributions.get("cohorts", {})
    return cohorts.get(key)


def industry_region_benchmark(form: dict, benchmarks: dict) -> dict:
    """Return industry and region default-rate context for the user."""
    if not benchmarks:
        return {}
    industry = form.get("organization_type", "Other")
    city_rating = form.get("city_rating", 2)
    industry_rate = benchmarks.get("industry_rates", {}).get(industry)
    region_rate = benchmarks.get("region_rates", {}).get(str(city_rating))
    return {
        "industry_label": industry,
        "industry_rate": industry_rate,
        "region_label": str(city_rating),
        "region_rate": region_rate,
        "population_rate": benchmarks.get("population_rate"),
    }
