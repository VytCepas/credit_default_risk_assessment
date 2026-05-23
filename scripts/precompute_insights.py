"""Precompute lookup artefacts for the Insights module.

Generates two JSON files used at predict time by ``models.insights``:

- ``scripts/results/cohort_distributions.json`` — risk-score distribution
  per ``(age_bucket, income_bucket)`` cohort, used by P-03.
- ``scripts/results/industry_region_benchmarks.json`` — mean default rate
  per ``organization_type`` and per ``region_rating_client_w_city``, used
  by P-04.

Re-runs are cheap (~30 s on a laptop). Should be re-run any time the
training set is refreshed or the Top25 model is re-tuned.

Usage:
    .venv/bin/python scripts/precompute_insights.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from models.top25_predictor import Top25Predictor, form_to_model_row  # noqa: E402, F401

DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RANDOM_STATE = 0
SAMPLE_PER_BUCKET = 500  # cap CPU when scoring cohort samples
TIER_BOUNDARIES = (0.30, 0.60)  # not used in this script but documented


def age_bucket(years: float) -> str:
    if years < 25:
        return "18-24"
    if years < 35:
        return "25-34"
    if years < 45:
        return "35-44"
    if years < 55:
        return "45-54"
    return "55+"


def income_bucket(income: float) -> str:
    if income < 50_000:
        return "0-50k"
    if income < 100_000:
        return "50k-100k"
    if income < 200_000:
        return "100k-200k"
    return "200k+"


def main() -> None:
    print("Loading dataset and model bundle...")
    df = pd.read_parquet(DATA_DIRECTORY / "application_train.parquet")
    df.columns = df.columns.str.lower()
    df["days_employed"] = df["days_employed"].replace(365243, np.nan)
    predictor = Top25Predictor(
        Path(__file__).resolve().parent.parent / "src/assets/top25_risk_model.pkl"
    )

    print("Bucketing applicants...")
    df["age_years"] = (-df["days_birth"]) / 365
    df["age_bucket"] = df["age_years"].apply(age_bucket)
    df["income_bucket"] = df["amt_income_total"].apply(income_bucket)
    df["cohort_key"] = (
        "age=" + df["age_bucket"] + "|income=" + df["income_bucket"]
    )

    # ----- Industry & region benchmarks (P-04) -----
    print("Computing industry / region benchmarks...")
    industry_rates = (
        df.groupby("organization_type")["target"].mean().sort_values(ascending=False)
        .round(4).to_dict()
    )
    region_rates = (
        df.groupby("region_rating_client_w_city")["target"].mean()
        .round(4).to_dict()
    )
    benchmarks = {
        "industry_rates": industry_rates,
        "region_rates": {str(k): v for k, v in region_rates.items()},
        "population_rate": float(round(df["target"].mean(), 4)),
        "n_total": int(len(df)),
    }
    with open(RESULTS_DIR / "industry_region_benchmarks.json", "w") as f:
        json.dump(benchmarks, f, indent=2)
    print(
        f"  Industry rates: {len(industry_rates)} categories; "
        f"global default rate {benchmarks['population_rate']}"
    )

    # ----- Cohort risk-score distributions (P-03) -----
    print("Scoring sampled applicants per cohort (this is the slow step)...")
    cohorts = {}
    grouped = df.groupby("cohort_key")
    for cohort_key, sub in grouped:
        if len(sub) < 50:
            continue
        sample = sub.sample(
            n=min(SAMPLE_PER_BUCKET, len(sub)), random_state=RANDOM_STATE,
        )
        # Build a form-like row from each applicant so we score with the same path.
        scores: list[int] = []
        for _, raw_row in sample.iterrows():
            form_row = _row_to_form(raw_row)
            try:
                res = predictor.predict(form_row)
                scores.append(int(res["risk_score"]))
            except Exception:
                continue
        if not scores:
            continue
        quantiles = {
            "5": float(np.percentile(scores, 5)),
            "25": float(np.percentile(scores, 25)),
            "50": float(np.percentile(scores, 50)),
            "75": float(np.percentile(scores, 75)),
            "95": float(np.percentile(scores, 95)),
        }
        cohorts[cohort_key] = {
            "label": cohort_key,
            "n": int(len(sub)),
            "score_quantiles": quantiles,
        }
        print(f"  {cohort_key}: n={len(sub)} sampled={len(scores)} p50={quantiles['50']:.0f}")

    all_scores = sum((c["score_quantiles"]["50"] for c in cohorts.values()), 0)
    fallback = {
        "label": "all-applicants",
        "n": int(len(df)),
        "score_quantiles": cohorts[max(cohorts, key=lambda k: cohorts[k]["n"])]["score_quantiles"]
        if cohorts else {},
    }
    distributions = {"cohorts": cohorts, "fallback": fallback}
    with open(RESULTS_DIR / "cohort_distributions.json", "w") as f:
        json.dump(distributions, f, indent=2)
    print(f"  Saved {len(cohorts)} cohort distributions.")


def _row_to_form(row: pd.Series) -> dict:
    """Convert an `application_train` row to the friendly-form dict that
    :func:`models.top25_predictor.form_to_model_row` expects."""
    age = max(int(-row["days_birth"] // 365), 18) if pd.notnull(row["days_birth"]) else 30
    employed = (
        max(float(-row["days_employed"] / 365), 0.0)
        if pd.notnull(row["days_employed"]) else 0.0
    )
    return {
        "gender": "Female" if row.get("code_gender") == "F" else "Male",
        "age_years": age,
        "num_children": int(row.get("cnt_children", 0) or 0),
        "num_family_members": float(row.get("cnt_fam_members", 1) or 1),
        "family_status": str(row.get("name_family_status", "Married") or "Married"),
        "years_employed": employed,
        "organization_type": str(row.get("organization_type", "Other") or "Other"),
        "occupation_type": str(row.get("occupation_type", "Other") or "Other"),
        "has_work_phone": bool(row.get("flag_work_phone", 0)),
        "contract_type": str(row.get("name_contract_type", "Cash loans") or "Cash loans"),
        "credit_amount": float(row.get("amt_credit", 500_000) or 500_000),
        "loan_annuity": float(row.get("amt_annuity", 25_000) or 25_000),
        "goods_price": float(row.get("amt_goods_price", 0) or 0),
        "total_income": float(row.get("amt_income_total", 100_000) or 100_000),
        "owns_car": row.get("flag_own_car") == "Y",
        "car_age_years": (
            float(row["own_car_age"]) if pd.notnull(row.get("own_car_age")) else None
        ),
        "owns_housing": row.get("flag_own_realty") == "Y",
        "years_since_id_change": (
            max(float(-row.get("days_id_publish", 0)) / 365, 0)
            if pd.notnull(row.get("days_id_publish")) else 0
        ),
        "years_at_address": (
            max(float(-row.get("days_registration", 0)) / 365, 0)
            if pd.notnull(row.get("days_registration")) else 0
        ),
        "region_population_relative": float(
            row.get("region_population_relative", 0.02) or 0.02
        ),
        "city_rating": int(row.get("region_rating_client_w_city", 2) or 2),
        "works_in_different_city": bool(row.get("reg_city_not_work_city", 0)),
        "has_landline": bool(row.get("flag_phone", 0)),
    }


if __name__ == "__main__":
    main()
