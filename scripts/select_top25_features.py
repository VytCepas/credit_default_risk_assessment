"""Stage 1 — Feature importance on self-reportable candidates.

Trains a LightGBM model on every column in application_train that a
loan applicant can plausibly answer themselves (no bureau-pulled
features, no apartment-stat columns, no document flags). Ranks by
LightGBM gain importance and saves the top 25 to a JSON artefact
used by the Stage-2 squeeze script and the Streamlit refactor.

What is excluded and why:
- EXT_SOURCE_1/2/3 — external bureau scores (consented-pull track #72).
- *_AVG / *_MEDI / *_MODE apartment statistics — applicant cannot answer.
- OBS_*_CNT_SOCIAL_CIRCLE, DEF_*_CNT_SOCIAL_CIRCLE — bureau-derived.
- FLAG_DOCUMENT_2 through FLAG_DOCUMENT_21 — technical, internal.
- AMT_REQ_CREDIT_BUREAU_* — bureau-derived.
- DAYS_LAST_PHONE_CHANGE — applicants don't reliably remember.
- NAME_TYPE_SUITE — "who accompanied you" — odd for online flow.

Usage:
    .venv/bin/python scripts/select_top25_features.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "data"
RANDOM_STATE = 0

# ---------------------------------------------------------------------------
# 1. Self-reportable candidate universe (~35 fields)
# ---------------------------------------------------------------------------
SELF_REPORTABLE_NUMERIC = [
    # core financial / loan
    "amt_income_total", "amt_credit", "amt_annuity", "amt_goods_price",
    # demographics / employment
    "cnt_children", "cnt_fam_members",
    "days_birth", "days_employed",
    "days_registration", "days_id_publish",
    "own_car_age",
    # region / location-related (numeric)
    "region_population_relative",
    "region_rating_client", "region_rating_client_w_city",
    # binary cross-region flags (numeric 0/1)
    "reg_region_not_live_region", "reg_region_not_work_region",
    "live_region_not_work_region",
    "reg_city_not_live_city", "reg_city_not_work_city",
    "live_city_not_work_city",
    # contact-info flags (numeric 0/1)
    "flag_mobil", "flag_emp_phone", "flag_work_phone",
    "flag_cont_mobile", "flag_phone", "flag_email",
    # application-time signals (askable indirectly; system can fill)
    "hour_appr_process_start",
]
SELF_REPORTABLE_CATEGORICAL = [
    "name_contract_type",
    "code_gender",
    "flag_own_car", "flag_own_realty",   # Y/N strings
    "name_income_type",
    "name_education_type",
    "name_family_status",
    "name_housing_type",
    "occupation_type",
    "organization_type",
    "weekday_appr_process_start",
]
ALL_CANDIDATES = SELF_REPORTABLE_NUMERIC + SELF_REPORTABLE_CATEGORICAL

print(f"Self-reportable candidates: {len(ALL_CANDIDATES)} "
      f"({len(SELF_REPORTABLE_NUMERIC)} numeric + "
      f"{len(SELF_REPORTABLE_CATEGORICAL)} categorical)")

# ---------------------------------------------------------------------------
# 2. Load and preprocess
# ---------------------------------------------------------------------------
print("Loading dataset...")
df = pd.read_parquet(DATA_DIRECTORY / "application_train.parquet")
df.columns = df.columns.str.lower()
df["days_employed"] = df["days_employed"].replace(365243, np.nan)

X = df[ALL_CANDIDATES].copy()
y = df["target"].astype(int)

# Categorical encoding (ordinal — LightGBM handles cats natively if told which cols).
enc = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1,
    encoded_missing_value=-1,
)
X[SELF_REPORTABLE_CATEGORICAL] = enc.fit_transform(
    X[SELF_REPORTABLE_CATEGORICAL].astype(str)
)

# Numeric imputation (median; LightGBM also handles NaN, but consistent)
for col in SELF_REPORTABLE_NUMERIC:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].median())

print(f"Feature matrix: {X.shape}")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
)

# ---------------------------------------------------------------------------
# 3. Train LightGBM on full candidate set
# ---------------------------------------------------------------------------
print(f"Training LightGBM on all {len(ALL_CANDIDATES)} candidates...")
t0 = time.perf_counter()
lgbm = LGBMClassifier(
    n_estimators=500, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
)
lgbm.fit(
    X_tr, y_tr,
    categorical_feature=[X_tr.columns.get_loc(c) for c in SELF_REPORTABLE_CATEGORICAL],
)
full_auc = round(float(roc_auc_score(y_te, lgbm.predict_proba(X_te)[:, 1])), 4)
elapsed = round(time.perf_counter() - t0, 1)
print(f"Full candidate-set AUC: {full_auc}  (train time: {elapsed}s)")

# ---------------------------------------------------------------------------
# 4. Rank by gain importance, pick top 25
# ---------------------------------------------------------------------------
importance_df = (
    pd.DataFrame({"feature": X.columns, "gain_importance": lgbm.feature_importances_})
    .sort_values("gain_importance", ascending=False)
    .reset_index(drop=True)
)
print("\n=== Full ranking ===")
print(importance_df.to_string(index=False))

TOP_K = 25
top25 = importance_df.head(TOP_K)
print(f"\n=== Top {TOP_K} ===")
print(top25.to_string(index=False))

selected_features = top25["feature"].tolist()
selected_numeric = [c for c in selected_features if c in SELF_REPORTABLE_NUMERIC]
selected_categorical = [c for c in selected_features if c in SELF_REPORTABLE_CATEGORICAL]

# ---------------------------------------------------------------------------
# 5. Validate: retrain on the top-25 only
# ---------------------------------------------------------------------------
print(f"\nRetraining on top {TOP_K} only...")
X_tr25, X_te25 = X_tr[selected_features], X_te[selected_features]
lgbm25 = LGBMClassifier(
    n_estimators=500, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
)
cat_idx_25 = [X_tr25.columns.get_loc(c) for c in selected_categorical]
lgbm25.fit(X_tr25, y_tr, categorical_feature=cat_idx_25)
top25_auc = round(float(roc_auc_score(y_te25 := y_te, lgbm25.predict_proba(X_te25)[:, 1])), 4)
print(f"Top-{TOP_K} AUC: {top25_auc}  (Δ vs full {full_auc - top25_auc:+.4f})")

# ---------------------------------------------------------------------------
# 6. Save artefact
# ---------------------------------------------------------------------------
out_path = Path(__file__).resolve().parent / "results" / "top25_features.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
artefact = {
    "candidate_count": len(ALL_CANDIDATES),
    "selected_count": TOP_K,
    "full_candidate_auc": full_auc,
    "top25_auc": top25_auc,
    "selected_features": selected_features,
    "selected_numeric": selected_numeric,
    "selected_categorical": selected_categorical,
    "full_ranking": importance_df.to_dict(orient="records"),
}
with open(out_path, "w") as f:
    json.dump(artefact, f, indent=2)
print(f"\n✓ Saved to {out_path}")
print(f"\nSelected numeric ({len(selected_numeric)}): {selected_numeric}")
print(f"Selected categorical ({len(selected_categorical)}): {selected_categorical}")
