# Modeling Pipeline

## Production pipeline at a glance

```
application_train.parquet (307,511 × 122)
   │
   ▼
scripts/select_top25_features.py     ← Stage 1: pick 25 self-reportable features by LightGBM gain
   │     writes scripts/results/top25_features.json
   ▼
scripts/squeeze_top25_accuracy.py    ← Stage 2: add 6 ratios + RandomizedSearchCV tuning
   │     writes src/assets/top25_risk_model.pkl
   │     writes scripts/results/squeeze_summary.json
   ▼
src/assets/top25_risk_model.pkl   ──►  models/top25_predictor.Top25Predictor
   │
   ▼
app.py (Streamlit)
```

Result: **ROC-AUC 0.7146** on the holdout split (Practical 3 §7.4).

## How we got there — the expansion experiments

A series of measured experiments establishes both the lower bound (what we
collect from the user) and the unconstrained ceiling on this dataset. All
numbers below are recorded in Practical 3 §7.

| Experiment | Configuration | ROC-AUC | Δ vs old 15-field baseline (0.6272) |
|---|---|---|---|
| **Old production GBM** | 15 questionnaire features, GBM, threshold 0.37 | 0.6272 | — |
| E2a — questionnaire + 5 ratios | 12 numeric + 5 derived, LightGBM defaults | 0.6846 | +0.0574 |
| E1 — unconstrained baseline | 104 numeric features incl. EXT_SOURCE_*, LightGBM | 0.7589 | +0.1317 |
| E2b — unconstrained + ratios + `ext_2*3` | 111 numeric features | 0.7658 | +0.1386 |
| E3 — RandomizedSearchCV on E2a | tuned LightGBM, 20×3 CV fast variant | 0.6877 holdout | +0.0605 |
| E4 — CTGAN-balanced LightGBM | E2a + 30K synthetic minority rows | 0.6882 | +0.0610 |
| E5 — stacking + Platt calibration | GBM + LightGBM + XGBoost on E2a, LR meta | 0.6848 | +0.0576 |
| **Top-25 squeeze (current production)** | 25 self-reportable + 6 ratios + RandomizedSearchCV | **0.7146** | **+0.0874** |

### Takeaways for defence

1. **+0.057 AUC from zero-data-cost ratio features** (E2a). Derivations from columns we already collect.
2. **+0.132 AUC from removing the questionnaire constraint** (E1 vs old prod). Confirms the product requirement, not the algorithm, is the binding cap.
3. **Within the 12-feature constraint, every technique we tried lands at ~0.685 ± 0.005** — marginal-return ceiling at that feature set.
4. **Expanding the form to 25 self-reportable fields + 6 ratios lifts us to 0.7146** — within 0.04 of the Kaggle median, breaking above the application-only LR baseline (~0.70), without using supplementary tables.

## Feature engineering

The 6 derived ratios in production (computed server-side, invisible to the user):

| Ratio | Formula |
|---|---|
| `dti` | `loan_annuity / total_income` |
| `credit_to_income` | `credit_amount / total_income` |
| `annuity_to_credit` | `loan_annuity / credit_amount` |
| `credit_to_goods` | `credit_amount / goods_price` |
| `years_employed_ratio` | `years_employed / age_in_years` |
| `income_per_family_member` | `total_income / num_family_members` |

These are computed by `models.top25_predictor.Top25Predictor` at inference,
not stored.

## Hyperparameters (production)

Best `RandomizedSearchCV` result from `scripts/squeeze_top25_accuracy.py`:

```python
LGBMClassifier(
    n_estimators=700,
    learning_rate=0.02,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_alpha=0,
    reg_lambda=1.0,
    objective="binary",
    random_state=42,
)
```

## Class imbalance

The Home Credit dataset is ~8 % positives. Two strategies tried:

- **SMOTETomek** inside `imblearn.Pipeline` — used in the older 15-field GBM. Leakage-safe because resampling lives inside the pipeline.
- **CTGAN minority oversampling** (E4) — Xu et al. NeurIPS 2019. AUC 0.6882 vs 0.6846 baseline; marginal gain doesn't justify the dependency cost in production. Kept as a research artefact.

The current production model **does not resample** — RandomizedSearchCV +
`class_weight` and the LightGBM `objective="binary"` proved sufficient at
the 25-feature scale.

## What's deferred (Sprint 4+)

- **Bureau / `previous_application` aggregations** (predicted +0.04–0.06 AUC). Needs Kaggle credentials in the build env.
- **EXT_SOURCE_* consented bureau pull** (predicted +0.04–0.06 AUC). Gated on legal review.
- **Stacking with more diverse base learners** + denoising-autoencoder embeddings (top-1% Kaggle technique). Deferred indefinitely.

See [Roadmap](Roadmap) for sprint sequencing.

## Reproducing the numbers

```bash
# from repo root, with .venv activated
python scripts/select_top25_features.py     # Stage 1
python scripts/squeeze_top25_accuracy.py    # Stage 2 → writes new top25_risk_model.pkl

# Optional — reproduce expansion experiments
python scripts/run_e4_ctgan.py
python scripts/run_e5_stacking.py
```

For the narrative version, open `notebooks/risk_default_analysis.ipynb` or
the reactive `marimo/risk_default_analysis.py`.

## Sources

- Practical 3 report §6 (Kaggle benchmark) and §7 (expansion experiments).
- 1st-place Kaggle write-up: <https://www.kaggle.com/competitions/home-credit-default-risk/writeups/home-aloan-1st-place-solution>
- Aguiar public kernel: <https://github.com/js-aguiar/home-credit-default-competition>
- CTGAN paper: <https://arxiv.org/abs/1907.00503>
