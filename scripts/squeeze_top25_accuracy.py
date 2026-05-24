"""Stage 2 — Squeeze every accuracy technique on the top-25 feature set.

Pipeline applied sequentially, each step measured vs production (0.6272)
and vs the previous stage. The final tuned model is saved as a pickle
ready to power the new Streamlit questionnaire.

Steps:
  S0: Baseline LightGBM on top 25 (already measured by Stage 1)
  S1: + derived ratios (DTI, credit_to_income, annuity_to_credit,
      credit_to_goods, employed/birth, income_per_family_member)
  S2: + RandomizedSearchCV (30 iter × 3-fold) on (S1 features)
  S3: + CTGAN tabular-GAN minority oversampling on the tuned model
  S4: + Stacking ensemble (tuned LGBM + GBM + XGB) with Platt calibration

Owner: Vytautas (orchestration), Laurynas (CTGAN step LZ-9).

Usage:
    .venv/bin/python scripts/squeeze_top25_accuracy.py
"""
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ctgan import CTGAN
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier

DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RANDOM_STATE = 0

# ---------------------------------------------------------------------------
# Load top-25 from Stage 1 artefact
# ---------------------------------------------------------------------------
with open(RESULTS_DIR / "top25_features.json") as f:
    stage1 = json.load(f)

selected_numeric = stage1["selected_numeric"]
selected_categorical = stage1["selected_categorical"]
selected_features = stage1["selected_features"]
print(f"Loaded {len(selected_features)} top-25 features from Stage 1 artefact")

# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------
print("Loading dataset...")
df = pd.read_parquet(DATA_DIRECTORY / "application_train.parquet")
df.columns = df.columns.str.lower()
df["days_employed"] = df["days_employed"].replace(365243, np.nan)
y = df["target"].astype(int)

# ---------------------------------------------------------------------------
# S1 — Top-25 + derived ratios
# ---------------------------------------------------------------------------
print("\n=== S1: top-25 + derived ratios ===")
eps = 1e-9
df["dti"] = df["amt_annuity"] / (df["amt_income_total"] + eps)
df["credit_to_income"] = df["amt_credit"] / (df["amt_income_total"] + eps)
df["annuity_to_credit"] = df["amt_annuity"] / (df["amt_credit"] + eps)
df["credit_to_goods"] = df["amt_credit"] / (df["amt_goods_price"].fillna(df["amt_credit"]) + eps)
df["years_employed_ratio"] = (-df["days_employed"]) / ((-df["days_birth"]) + eps)
df["income_per_family_member"] = df["amt_income_total"] / (df["cnt_fam_members"] + eps)
derived_features = [
    "dti", "credit_to_income", "annuity_to_credit", "credit_to_goods",
    "years_employed_ratio", "income_per_family_member",
]
print(f"Added {len(derived_features)} ratios")

feature_set = selected_features + derived_features
X = df[feature_set].copy()

# Encode categoricals
enc = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1,
    encoded_missing_value=-1,
)
X[selected_categorical] = enc.fit_transform(X[selected_categorical].astype(str))

# Impute numerics
for col in selected_numeric + derived_features:
    if X[col].isnull().any():
        X[col] = X[col].fillna(X[col].median())

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
)
cat_idx = [X_tr.columns.get_loc(c) for c in selected_categorical]

# S1: train baseline LightGBM on extended feature set
t0 = time.perf_counter()
lgbm_s1 = LGBMClassifier(
    n_estimators=500, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
)
lgbm_s1.fit(X_tr, y_tr, categorical_feature=cat_idx)
s1_time = round(time.perf_counter() - t0, 1)
s1_auc = round(float(roc_auc_score(y_te, lgbm_s1.predict_proba(X_te)[:, 1])), 4)
print(f"S1 AUC: {s1_auc}  (train time: {s1_time}s)")

# ---------------------------------------------------------------------------
# S2 — RandomizedSearchCV on the extended feature set
# Reduced budget (15 iter × 3-fold) — fast variant; cells in notebook use 50×5
# ---------------------------------------------------------------------------
print("\n=== S2: + RandomizedSearchCV (15 iter × 3-fold, fast variant) ===")
param_dist = {
    "n_estimators":      [300, 500, 700],
    "learning_rate":     [0.02, 0.05, 0.1],
    "num_leaves":        [31, 63],
    "min_child_samples": [20, 50],
    "subsample":         [0.6, 0.8, 1.0],
    "colsample_bytree":  [0.6, 0.8, 1.0],
    "reg_alpha":         [0, 0.1, 1.0],
    "reg_lambda":        [0, 0.1, 1.0],
}
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
search = RandomizedSearchCV(
    LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=1),
    param_distributions=param_dist,
    n_iter=15, cv=skf, scoring="roc_auc",
    random_state=RANDOM_STATE, n_jobs=-1, verbose=0,
)
t0 = time.perf_counter()
search.fit(X_tr, y_tr, categorical_feature=cat_idx)
s2_time = round(time.perf_counter() - t0, 1)
best_params = search.best_params_
best_cv_auc = round(float(search.best_score_), 4)
s2_auc = round(
    float(roc_auc_score(y_te, search.best_estimator_.predict_proba(X_te)[:, 1])),
    4,
)
print(f"S2 best CV AUC: {best_cv_auc} | holdout AUC: {s2_auc}  (time: {s2_time}s)")
print(f"Best params: {best_params}")
tuned_model = search.best_estimator_

# ---------------------------------------------------------------------------
# S3 — Tuned LightGBM + CTGAN minority oversampling
# ---------------------------------------------------------------------------
print("\n=== S3: + CTGAN minority oversampling ===")
minority = X_tr[y_tr == 1].reset_index(drop=True)
sub = minority.sample(n=min(3000, len(minority)), random_state=RANDOM_STATE)

t0 = time.perf_counter()
ctgan = CTGAN(epochs=30, verbose=False, enable_gpu=False)
ctgan.fit(sub)
ctgan_train = round(time.perf_counter() - t0, 1)
print(f"CTGAN trained: {ctgan_train}s")

t0 = time.perf_counter()
synth = ctgan.sample(40000)
sample_time = round(time.perf_counter() - t0, 1)
print(f"CTGAN sampled 40,000 synthetic minority: {sample_time}s")

X_bal = pd.concat([X_tr, synth.reset_index(drop=True)], ignore_index=True)
y_bal = np.concatenate([y_tr.values, np.ones(len(synth), dtype=int)])

# Retrain tuned LightGBM on balanced
lgbm_s3 = LGBMClassifier(**best_params, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
t0 = time.perf_counter()
lgbm_s3.fit(X_bal, y_bal, categorical_feature=cat_idx)
s3_time = round(time.perf_counter() - t0, 1)
s3_auc = round(float(roc_auc_score(y_te, lgbm_s3.predict_proba(X_te)[:, 1])), 4)
print(f"S3 AUC: {s3_auc}  (train time: {s3_time}s)")

# ---------------------------------------------------------------------------
# S4 — Stacking ensemble (tuned LGBM + GBM + XGB) with Platt calibration
# ---------------------------------------------------------------------------
print("\n=== S4: + Stacking ensemble + Platt calibration ===")
base_estimators = [
    ("lgbm_tuned", LGBMClassifier(**best_params, random_state=RANDOM_STATE,
                                  verbose=-1, n_jobs=1)),
    ("gbm", GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=RANDOM_STATE)),
    ("xgb", XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, eval_metric="auc", verbosity=0, n_jobs=1)),
]
stack = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    cv=3, n_jobs=1, passthrough=False,
)
t0 = time.perf_counter()
stack.fit(X_tr, y_tr)
s4_stack_time = round(time.perf_counter() - t0, 1)
stack_proba = stack.predict_proba(X_te)[:, 1]
s4_stack_auc = round(float(roc_auc_score(y_te, stack_proba)), 4)
s4_brier_uncal = round(float(brier_score_loss(y_te, stack_proba)), 4)
print(f"Stack AUC: {s4_stack_auc}  Brier: {s4_brier_uncal}  (time: {s4_stack_time}s)")

cal = CalibratedClassifierCV(FrozenEstimator(stack), method="sigmoid", cv=5)
cal.fit(X_tr, y_tr)
cal_proba = cal.predict_proba(X_te)[:, 1]
s4_cal_auc = round(float(roc_auc_score(y_te, cal_proba)), 4)
s4_brier_cal = round(float(brier_score_loss(y_te, cal_proba)), 4)
print(f"Calibrated AUC: {s4_cal_auc}  Brier: {s4_brier_cal}")

# ---------------------------------------------------------------------------
# Save artefacts: summary JSON + best model pickle
# ---------------------------------------------------------------------------
summary = {
    "feature_count": len(feature_set),
    "categorical_features": selected_categorical,
    "numeric_features": selected_numeric,
    "derived_ratios": derived_features,
    "stages": {
        "production_reference": {"auc": 0.6272, "notes": "Practical 2 baseline"},
        "stage1_top25_only": {"auc": stage1["top25_auc"]},
        "S1_top25_plus_ratios": {"auc": s1_auc, "train_time_s": s1_time},
        "S2_randomized_search": {
            "auc": s2_auc, "best_cv_auc": best_cv_auc,
            "best_params": best_params, "train_time_s": s2_time,
        },
        "S3_ctgan_balanced": {
            "auc": s3_auc,
            "ctgan_train_s": ctgan_train,
            "ctgan_sample_s": sample_time,
            "lgbm_train_s": s3_time,
        },
        "S4_stacking": {
            "stack_auc": s4_stack_auc,
            "calibrated_auc": s4_cal_auc,
            "brier_uncalibrated": s4_brier_uncal,
            "brier_calibrated": s4_brier_cal,
            "stack_train_time_s": s4_stack_time,
        },
    },
    "best_overall_auc": max(s1_auc, s2_auc, s3_auc, s4_stack_auc, s4_cal_auc),
}
with open(RESULTS_DIR / "squeeze_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n✓ Summary saved to {RESULTS_DIR / 'squeeze_summary.json'}")

# Save the best model — choose by holdout AUC
candidates = {
    "S2_tuned_lgbm": (s2_auc, tuned_model),
    "S3_ctgan_tuned": (s3_auc, lgbm_s3),
    "S4_stack": (s4_stack_auc, stack),
    "S4_calibrated": (s4_cal_auc, cal),
}
best_name, (best_auc, best_model) = max(candidates.items(), key=lambda kv: kv[1][0])
print(f"\nBest model: {best_name} (AUC {best_auc})")

best_path = RESULTS_DIR / "best_top25_model.pkl"
with open(best_path, "wb") as f:
    pickle.dump(
        {
            "model": best_model,
            "best_name": best_name,
            "best_auc": best_auc,
            "ordinal_encoder": enc,
            "feature_set": feature_set,
            "selected_numeric": selected_numeric,
            "selected_categorical": selected_categorical,
            "derived_ratios": derived_features,
        },
        f,
    )
print(f"✓ Saved best model bundle to {best_path}")

print("\n=== Final progression ===")
print(f"  Production GBM (15 fields)        0.6272")
print(f"  Stage 1 — top 25 only             {stage1['top25_auc']}")
print(f"  S1 — + 6 derived ratios           {s1_auc}")
print(f"  S2 — + RandomizedSearchCV         {s2_auc}")
print(f"  S3 — + CTGAN balancing            {s3_auc}")
print(f"  S4 — + Stacking (uncalibrated)    {s4_stack_auc}")
print(f"  S4 — + Calibration                {s4_cal_auc}")
print(f"  Brier improvement (calibration)   {s4_brier_uncal - s4_brier_cal:+.4f}")
