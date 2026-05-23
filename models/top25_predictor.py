"""Inference wrapper for the Stage-2 squeeze model (top-25 self-reportable features).

Loads a pickle bundle produced by ``scripts/squeeze_top25_accuracy.py``:

    {
      "model": sklearn estimator (LightGBM / Stacking / Calibrated),
      "best_name": str,
      "best_auc": float,
      "ordinal_encoder": OrdinalEncoder,
      "feature_set": list[str],      # full feature order (25 + ratios)
      "selected_numeric": list[str],
      "selected_categorical": list[str],
      "derived_ratios": list[str],
    }

The form-side keys are friendlier than the model's column names (which
match Kaggle's schema). The mapping is encapsulated here so the Streamlit
component only needs to deal with form keys.
"""
from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EPS = 1e-9

# Mapping from form key → Kaggle column name + conversion fn.
# Form values come in as plain Python types (str / int / float / bool);
# the conversion fn produces the value the model expects.
_FORM_TO_KAGGLE: dict[str, tuple[str, Any]] = {
    # Personal
    "gender": ("code_gender", lambda v: "F" if str(v).lower().startswith("f") else "M"),
    "age_years": ("days_birth", lambda v: -float(v) * 365),
    "num_children": ("cnt_children", lambda v: int(v)),
    "num_family_members": ("cnt_fam_members", lambda v: float(v)),
    "family_status": ("name_family_status", str),
    # Employment
    "years_employed": ("days_employed", lambda v: -float(v) * 365 if v else 0),
    "organization_type": ("organization_type", str),
    "occupation_type": ("occupation_type", str),
    "has_work_phone": ("flag_work_phone", lambda v: int(bool(v))),
    # Loan
    "contract_type": ("name_contract_type", str),
    "credit_amount": ("amt_credit", float),
    "loan_annuity": ("amt_annuity", float),
    "goods_price": ("amt_goods_price", float),
    # Financial
    "total_income": ("amt_income_total", float),
    # Assets
    "owns_car": ("flag_own_car", lambda v: "Y" if v else "N"),
    "car_age_years": ("own_car_age", lambda v: float(v) if v else np.nan),
    "owns_housing": ("flag_own_realty", lambda v: "Y" if v else "N"),
    # Residence
    "years_since_id_change": ("days_id_publish", lambda v: -float(v) * 365),
    "years_at_address": ("days_registration", lambda v: -float(v) * 365),
    "region_population_relative": ("region_population_relative", float),
    "city_rating": ("region_rating_client_w_city", int),
    "works_in_different_city": ("reg_city_not_work_city", lambda v: int(bool(v))),
    # Other
    "has_landline": ("flag_phone", lambda v: int(bool(v))),
}


def form_to_model_row(form: dict[str, Any]) -> dict[str, Any]:
    """Translate the friendly form dict to a Kaggle-schema row dict."""
    row: dict[str, Any] = {}
    for form_key, (kaggle_col, conv) in _FORM_TO_KAGGLE.items():
        if form_key in form and form[form_key] is not None and form[form_key] != "":
            try:
                row[kaggle_col] = conv(form[form_key])
            except (TypeError, ValueError):
                row[kaggle_col] = np.nan

    # Auto-fill timestamp-derived fields (not asked of the user)
    now = datetime.now()
    row["hour_appr_process_start"] = now.hour
    row["weekday_appr_process_start"] = now.strftime("%A").upper()
    return row


def _add_derived_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the 6 derived ratios used by the squeeze model."""
    df = df.copy()
    df["dti"] = df["amt_annuity"] / (df["amt_income_total"] + EPS)
    df["credit_to_income"] = df["amt_credit"] / (df["amt_income_total"] + EPS)
    df["annuity_to_credit"] = df["amt_annuity"] / (df["amt_credit"] + EPS)
    df["credit_to_goods"] = df["amt_credit"] / (
        df["amt_goods_price"].fillna(df["amt_credit"]) + EPS
    )
    df["years_employed_ratio"] = (-df["days_employed"]) / ((-df["days_birth"]) + EPS)
    df["income_per_family_member"] = df["amt_income_total"] / (
        df["cnt_fam_members"] + EPS
    )
    return df


class Top25Predictor:
    """Lightweight predictor for the Stage-2 squeeze model bundle."""

    def __init__(self, bundle_path: str | Path):
        bundle_path = Path(bundle_path)
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"Top-25 model bundle not found at {bundle_path}. "
                "Run scripts/squeeze_top25_accuracy.py to produce it."
            )
        with open(bundle_path, "rb") as f:
            bundle = pickle.load(f)

        self.model = bundle["model"]
        self.best_name = bundle["best_name"]
        self.best_auc = bundle["best_auc"]
        self.ordinal_encoder = bundle["ordinal_encoder"]
        self.feature_set: list[str] = bundle["feature_set"]
        self.selected_numeric: list[str] = bundle["selected_numeric"]
        self.selected_categorical: list[str] = bundle["selected_categorical"]
        self.derived_ratios: list[str] = bundle["derived_ratios"]

        # Tier thresholds — copied from the production RiskPredictor for
        # consistency with the current Streamlit UI.
        self.low_threshold = 0.30
        self.high_threshold = 0.60

    def _prepare_input(self, form: dict[str, Any]) -> pd.DataFrame:
        row = form_to_model_row(form)
        df = pd.DataFrame([row])

        for col in self.selected_numeric:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in self.selected_categorical:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = df[col].astype(str)

        df = _add_derived_ratios(df)

        # Order columns and encode categoricals using the saved encoder
        df = df[self.feature_set]
        df[self.selected_categorical] = self.ordinal_encoder.transform(
            df[self.selected_categorical]
        )

        # Final NaN guard (LightGBM handles NaN; sklearn estimators may not)
        for col in self.selected_numeric + self.derived_ratios:
            if df[col].isnull().any():
                df[col] = df[col].fillna(0)
        return df

    def predict(self, form: dict[str, Any]) -> dict[str, Any]:
        X = self._prepare_input(form)
        proba = float(self.model.predict_proba(X)[0, 1])

        if proba < self.low_threshold:
            tier = "Low"
        elif proba < self.high_threshold:
            tier = "Medium"
        else:
            tier = "High"

        return {
            "risk_probability": proba,
            "risk_score": int(round(proba * 1000)),
            "risk_category": tier,
            "model_name": self.best_name,
            "model_auc": self.best_auc,
        }
