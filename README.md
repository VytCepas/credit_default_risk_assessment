# Home Credit Default Risk Assessment

[![CI](https://github.com/VytCepas/credit_default_risk_assessment/actions/workflows/ci.yml/badge.svg)](https://github.com/VytCepas/credit_default_risk_assessment/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Coverage gate](https://img.shields.io/badge/coverage-%E2%89%A575%25-brightgreen.svg)](.github/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Streamlit application that estimates loan default risk from a short 25-field questionnaire, explains the main contributing factors with SHAP, and pairs the score with a behavioral borrower profile.

Trained on the [Kaggle Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) dataset.

## Highlights

- **Questionnaire-driven**: 25 user-answerable inputs — no bureau pulls or document scraping required at inference time.
- **Explainable**: SHAP per-prediction attribution surfaces the top contributors behind each score.
- **Two-model output**: a default-risk score (0–1000) plus a complementary behavioral-traits profile.
- **Production model**: LightGBM (with GBM kept as a comparison baseline); CTGAN minority-class augmentation and stacking experiments live under `scripts/`.
- **CI-gated**: lint + 28-test pytest suite + ≥75 % coverage threshold enforced on every push.

## Tech stack

`Python 3.12` · `Streamlit` · `LightGBM` · `scikit-learn` · `SHAP` · `pandas` / `pyarrow` · `imbalanced-learn` · `CTGAN` · `Plotly` · `marimo` · `pytest` · GitHub Actions

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train models (requires data/application_train.parquet — see SETUP.md)
python models/risk_model.py
python models/behavioral_traits_model.py    # optional

streamlit run app.py
```

The app opens at <http://localhost:8501>. Full install, data-download, and deployment notes live in [SETUP.md](SETUP.md).

## Model performance

The production model is deliberately constrained to **25 questionnaire-collectible features** so the entire input can be answered by an applicant without external data.

| Configuration | ROC-AUC | Context |
|---|---|---|
| Production (25-field questionnaire) | ~0.63 | Product constraint: every input is user-answerable |
| Unconstrained `application_train` baseline | ~0.74 | All raw application features, no bureau joins |
| Kaggle median submission | ~0.75 | Public leaderboard reference |
| Aguiar "LightGBM with Simple Features" kernel | ~0.791 | Most-forked public seed |
| 1st place ("Home Aloan") | ~0.806 | LightGBM + XGBoost + CatBoost stacked ensemble |

The ~0.12 AUC gap between the production model and the Kaggle median is the deliberate cost of the questionnaire-only product requirement, not technical debt. Gap analysis and roadmap to close it are in [`project_docs/practical_3_report.md`](project_docs/practical_3_report.md) §6–§7.

## Risk categories

| Score | Category |
|-------|----------|
| 0–299 | Low Risk |
| 300–599 | Medium Risk |
| 600–1000 | High Risk |

## Project layout

```
.
├── app.py                      Streamlit entry point
├── models/                     Training pipelines + inference wrappers
├── src/
│   ├── assets/                 Trained model artefacts (.pkl)
│   ├── components/             Streamlit UI components
│   └── predictors/             Cached loaders wrapping models/
├── notebooks/                  Jupyter analysis (authoritative)
├── marimo/                     Reactive notebook ports (Epic 9)
├── scripts/                    One-shot ETL, experiment, precompute scripts
├── tests/                      pytest suites (unit + integration + SHAP + perf)
├── docs/architecture.md        Code-layout deep-dive
└── project_docs/               Practical briefs, reports, ADRs, meeting notes
```

For a fuller walkthrough see [`docs/architecture.md`](docs/architecture.md).

## Testing

```bash
pip install pytest pytest-cov
pytest --cov=src --cov=models --cov-fail-under=75
```

The suite covers the top-25 predictor, SHAP validation, prediction insights (P-01…P-09), an end-to-end integration flow, and a performance smoke test. CI runs the same commands on every push and pull request — see [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

## Documentation

- [SETUP.md](SETUP.md) — installation, data download, deployment
- [docs/architecture.md](docs/architecture.md) — where each piece of code lives and why
- [project_docs/](project_docs/) — practical briefs and reports (TA-1, TA-2, TA-3), ADRs, meeting notes

## Team

Vytautas Čepas · Laurynas Žalaga — Vilnius Tech academic project (3 practical milestones).
