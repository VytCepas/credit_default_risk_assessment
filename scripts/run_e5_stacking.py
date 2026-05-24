"""E5 — Stacking ensemble (GBM + LightGBM + XGBoost) + Platt calibration.

Reproduces the Practical 3 §7 E5 experiment standalone. Uses smaller
trees and n_jobs=1 to avoid joblib over-subscription that was observed
on the original config. The notebook cell uses the production-grade
config (more trees, n_jobs=-1) — run that on a dedicated machine.

Owner: Vytautas Čepas (Sprint 4 issues #48 + #49 + #52).

Usage:
    .venv/bin/python scripts/run_e5_stacking.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "data"
RANDOM_STATE = 0

app = pd.read_parquet(DATA_DIRECTORY / "application_train.parquet")
app.columns = app.columns.str.lower()
app["days_employed"] = app["days_employed"].replace(365243, np.nan)

eps = 1e-9
app["dti"] = app["amt_annuity"] / (app["amt_income_total"] + eps)
app["credit_to_income"] = app["amt_credit"] / (app["amt_income_total"] + eps)
app["annuity_to_credit"] = app["amt_annuity"] / (app["amt_credit"] + eps)
app["years_employed_ratio"] = (-app["days_employed"]) / ((-app["days_birth"]) + eps)
app["income_per_family_member"] = (
    app["amt_income_total"] / (app["cnt_fam_members"] + eps)
)

features = [
    "cnt_children", "amt_income_total", "amt_credit", "amt_annuity",
    "cnt_fam_members", "days_birth", "days_employed",
    "dti", "credit_to_income", "annuity_to_credit",
    "years_employed_ratio", "income_per_family_member",
]
X = app[features].copy().fillna(app[features].median(numeric_only=True))
y = app["target"].astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
)

# Reduced trees + n_jobs=1 to avoid joblib over-subscription seen in
# previous environment. Stacking still demonstrates the technique.
base_estimators = [
    ("gbm", GradientBoostingClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=RANDOM_STATE)),
    ("lgbm", LGBMClassifier(
        n_estimators=100, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, verbose=-1, n_jobs=1)),
    ("xgb", XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, eval_metric="auc", verbosity=0,
        n_jobs=1)),
]

stack = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    cv=3, n_jobs=1, passthrough=False,
)

print(f"Training stack ({len(base_estimators)} base models, cv=3, n_jobs=1)...")
t0 = time.perf_counter()
stack.fit(X_tr, y_tr)
stack_time = round(time.perf_counter() - t0, 1)
proba = stack.predict_proba(X_te)[:, 1]
stack_auc = round(float(roc_auc_score(y_te, proba)), 4)
brier_uncal = round(float(brier_score_loss(y_te, proba)), 4)
print(f"  Stack: AUC={stack_auc}, Brier={brier_uncal}, time={stack_time}s")

# Save stack-only result now so a calibration failure doesn't lose the data.
partial = {
    "experiment": "E5 Stacking (no calibration yet)",
    "base_models": ["gbm(50)", "lgbm(100)", "xgb(100)"],
    "features": len(features),
    "stack_train_time_s": stack_time,
    "stack_roc_auc": stack_auc,
    "brier_uncalibrated": brier_uncal,
}
partial_path = Path(__file__).resolve().parent / "results" / "e5_partial.json"
partial_path.parent.mkdir(parents=True, exist_ok=True)
with open(partial_path, "w") as f:
    json.dump(partial, f, indent=2)
print(f"  ✓ Stack-only result saved to {partial_path}")

print("Calibrating with Platt scaling (sigmoid)...")
# sklearn ≥ 1.6 deprecated cv="prefit"; use FrozenEstimator wrapper instead.
from sklearn.frozen import FrozenEstimator
cal = CalibratedClassifierCV(FrozenEstimator(stack), method="sigmoid", cv=5)
cal.fit(X_tr, y_tr)
cal_proba = cal.predict_proba(X_te)[:, 1]
cal_brier = round(float(brier_score_loss(y_te, cal_proba)), 4)
cal_auc = round(float(roc_auc_score(y_te, cal_proba)), 4)
print(f"  Calibrated: AUC={cal_auc}, Brier={cal_brier}")

result = {
    "experiment": "E5 Stacking + Platt calibration (fast config)",
    "base_models": ["gbm(50)", "lgbm(100)", "xgb(100)"],
    "features": len(features),
    "stack_train_time_s": stack_time,
    "stack_roc_auc": stack_auc,
    "brier_uncalibrated": brier_uncal,
    "calibrated_roc_auc": cal_auc,
    "brier_calibrated": cal_brier,
    "brier_improvement": round(brier_uncal - cal_brier, 4),
}
print(json.dumps(result, indent=2))
out_path = Path(__file__).resolve().parent / "results" / "e5_result.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\n✓ Saved to {out_path}")
