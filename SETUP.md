# Setup Guide

This document covers (1) running the shipped app, (2) re-training the model
end-to-end from the raw Kaggle parquets, and (3) deploying to Streamlit Cloud.

For an architectural overview see [`docs/architecture.md`](docs/architecture.md).
For contribution workflow see [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## 1. Prerequisites

- **Python 3.12** (the CI pipeline pins this version)
- **pip** ≥ 23 (for resolving the `lightgbm` / `xgboost` / `ctgan` deps)
- ~3 GB free disk for the cached Kaggle parquets if you plan to re-train

---

## 2. Install

```bash
git clone https://github.com/VytCepas/credit_default_risk_assessment.git
cd credit_default_risk_assessment

python -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

The repo ships with the **trained artefacts** committed under `src/assets/`,
so step 3 ("Run the app") works immediately without a training pass.

---

## 3. Run the app

```bash
streamlit run app.py
# → http://localhost:8501
```

The single flow is **Landing → 25-question form → Result page with 6 insight
tabs** (overview, counter-factuals, cohort, affordability, time projections,
behavioural traits).

---

## 4. (Optional) Re-train from the raw dataset

### 4a. Download the data

The dataset is **not** in the repo (Kaggle competition terms). Download from
[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk/data)
and place the parquet conversions under `data/`:

```
data/
├── application_train.parquet      # required
└── application_test.parquet       # optional, used for sanity checks
```

(If you only have the CSVs, convert with `pd.read_csv(...).to_parquet(...)` once.)

### 4b. Train the production risk model (Standard+ tier)

The production artefact is produced by a two-stage pipeline:

```bash
# Stage 1 — select the 25 self-reportable features by LightGBM gain importance
python scripts/select_top25_features.py
# → writes scripts/results/top25_features.json

# Stage 2 — squeeze AUC with 6 engineered ratios + RandomizedSearchCV
python scripts/squeeze_top25_accuracy.py
# → writes src/assets/top25_risk_model.pkl
#   and scripts/results/squeeze_summary.json
```

Expected ROC-AUC on the holdout split: **~0.7146** (Practical 3 §7.4).

### 4c. Precompute insight lookups

```bash
python scripts/precompute_insights.py
# → scripts/results/cohort_distributions.json
# → scripts/results/industry_region_benchmarks.json
```

These are loaded by `models/insights.py` for cohort percentile (P-03) and
industry/region benchmark (P-04). The app degrades gracefully if either file
is missing.

### 4d. (Optional) Re-train the behavioural-traits model

```bash
python models/behavioral_traits_model.py
# → src/assets/behavioral_traits_model.pkl
```

### 4e. (Optional) Reproduce expansion experiments

```bash
python scripts/run_e4_ctgan.py        # CTGAN minority-class balancing
python scripts/run_e5_stacking.py     # GBM + LightGBM + XGBoost stack + Platt calibration
# Results land in scripts/results/e4_result.json, e5_result.json
```

---

## 5. Risk categories shown in the UI

| Score (0–1000) | Tier badge |
|----------------|------------|
| 0–299          | Low Risk   |
| 300–599        | Medium Risk |
| 600–1000       | High Risk  |

The score is `int(default_probability × 1000)`; the decision threshold is
tuned by `scripts/squeeze_top25_accuracy.py` and stored inside the
`top25_risk_model.pkl` bundle.

---

## 6. Deployment — Streamlit Community Cloud

Main file path: `app.py`. Python version: `3.12`. Dependencies are read from
`requirements.txt`. Trained artefacts (`src/assets/*.pkl`,
`scripts/results/*.json`) are committed to the repo, so the deploy needs no
external download step.

Note that `data/*.parquet` is **gitignored** and not required at runtime —
the app only loads the pickled model bundle.

---

## 7. Tests

```bash
pytest tests/ -v --cov=models --cov=src --cov-report=term-missing
```

17 tests across two suites:

- `tests/test_top25_predictor.py` — bundle loads, output shapes, score/tier mapping
- `tests/test_insights.py` — counter-factuals, cohort percentile, recommended max loan, affordability sandbox

---

## 8. Marimo notebooks

```bash
.venv/bin/marimo edit marimo/risk_default_analysis.py
.venv/bin/marimo edit marimo/top25_squeeze.py
```

See [`marimo/README.md`](marimo/README.md) for the rationale (reactive
execution, `.py` source format, agent-friendly diffs).
