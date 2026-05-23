"""E4 — CTGAN tabular GAN minority-class oversampling (fast config).

Reproduces the Practical 3 §7 E4 experiment standalone. Uses a reduced
config (epochs=20, 2000 minority samples, sample 30K synthetic) to stay
under ~5 minutes CPU. The notebook cell uses the production-grade
epochs=50 / full-balance config — run that on a dedicated machine
for the final report numbers.

Owner: Laurynas Žalaga (Sprint 4 LZ-9).

Usage:
    .venv/bin/python scripts/run_e4_ctgan.py
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from ctgan import CTGAN
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

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

# Fast config: 2000 minority samples × 20 epochs, sample 30K synthetic
minority = X_tr[y_tr == 1].reset_index(drop=True)
sub = minority.sample(n=min(2000, len(minority)), random_state=RANDOM_STATE)

print(f"Training CTGAN on {len(sub)} samples (epochs=20)...")
t0 = time.perf_counter()
ctgan = CTGAN(epochs=20, verbose=False, enable_gpu=False)
ctgan.fit(sub)
ctgan_time = round(time.perf_counter() - t0, 1)
print(f"  CTGAN trained in {ctgan_time}s")

n_to_gen = 30000
print(f"Sampling {n_to_gen} synthetic minority records...")
t0 = time.perf_counter()
synth = ctgan.sample(n_to_gen)
sample_time = round(time.perf_counter() - t0, 1)
print(f"  Sampled in {sample_time}s")

X_bal = pd.concat([X_tr, synth.reset_index(drop=True)], ignore_index=True)
y_bal = np.concatenate([y_tr.values, np.ones(n_to_gen, dtype=int)])
print(f"  Balanced shape: {X_bal.shape}, positive rate: {y_bal.mean():.3f}")

print("Training LightGBM on GAN-augmented data...")
t0 = time.perf_counter()
lgbm = LGBMClassifier(
    n_estimators=300, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, verbose=-1,
)
lgbm.fit(X_bal, y_bal)
lgbm_time = round(time.perf_counter() - t0, 1)
auc = round(float(roc_auc_score(y_te, lgbm.predict_proba(X_te)[:, 1])), 4)

result = {
    "experiment": "E4 CTGAN-balanced LightGBM (fast config)",
    "config": "CTGAN(epochs=20) on 2000 minority samples; 30K synthetic; LightGBM defaults",
    "features": len(features),
    "ctgan_train_time_s": ctgan_time,
    "sample_time_s": sample_time,
    "lgbm_train_time_s": lgbm_time,
    "synthetic_minority_added": n_to_gen,
    "roc_auc": auc,
}
print(json.dumps(result, indent=2))
out_path = Path(__file__).resolve().parent / "results" / "e4_result.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
print(f"\n✓ Saved to {out_path}")
