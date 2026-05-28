# Home Credit Default Risk Assessment

[![CI](https://github.com/VytCepas/credit_default_risk_assessment/actions/workflows/ci.yml/badge.svg)](https://github.com/VytCepas/credit_default_risk_assessment/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-academic-lightgrey)

A Streamlit application that scores loan applicants for default risk in under
two seconds, explains the score with SHAP, and surfaces nine user-facing
insights — counter-factuals, cohort percentile, recommended max loan,
loan-affordability sandbox, and more — built on the public
[Kaggle Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk)
dataset.

The system is the term project for the *AI Product Development* course
(Vilnius University). It is defended in three milestones (TA-1/2/3) and
shipped as a single Standard+ flow that the user can run end-to-end on a
laptop, no cloud, no GPU.

---

## What's in the box

| Capability | Where |
|---|---|
| **Standard+ application form** — 22 user-typed questions (+ 2 auto-fill) covering personal, employment, loan, financial, assets, residence | `src/components/questionnaire_top25.py` |
| **Risk model** — LightGBM on 25 self-reportable features + 6 engineered ratios, RandomizedSearchCV-tuned, ROC-AUC **0.7146** | `src/assets/top25_risk_model.pkl` · `models/top25_predictor.py` |
| **9 user-facing insights** (P-01…P-09 — counter-factuals, cohort percentile, industry benchmark, affordability sandbox, recommended max loan, time-to-improvement, approval-process time, risk decomposition) | `models/insights.py` |
| **Behavioural-traits profile** as a complementary tab | `models/behavioral_traits_model.py` · `src/components/behavioral_traits.py` |
| **SHAP explanations** for every prediction | `models/insights.py` + Streamlit result tabs |
| **CI pipeline** (lint + 28 tests + coverage XML) | `.github/workflows/ci.yml` |
| **Reactive notebooks** — full E1–E5 experiment chain + a what-if playground | `marimo/` |
| **Authoritative analysis notebook** | `notebooks/risk_default_analysis.ipynb` |

---

## Model performance at a glance

The production model is **deliberately constrained** to features an applicant
can self-report in a web form. Headline numbers (Practical 3 §7):

| Model | Inputs | ROC-AUC | Reference |
|---|---|---|---|
| **Production (Standard+ tier)** | 25 self-reported fields + 6 ratios, tuned LightGBM | **0.7146** 🏆 | `scripts/results/squeeze_summary.json` |
| Earlier baseline (15 fields, GBM) | 15 fields | 0.6272 | retired |
| Unconstrained baseline (same dataset) | ~104 numeric features incl. EXT_SOURCE_* | 0.7589 | E1 |
| Aguiar public kernel | application + bureau aggregations | ~0.791 | top-100 reference |
| Kaggle median submission | full feature stack | ~0.75 | leaderboard |
| Kaggle 1st place | LightGBM/XGBoost/CatBoost stack | ~0.806 | leaderboard |

We sit **+0.087 AUC above** the original 15-field flow, **above** the
application-only logistic-regression baseline (~0.70), and **within 0.04** of
the Kaggle median — without using any supplementary tables. The remaining gap
to leaderboard territory is bureau-aggregation work (Sprint 4 / Epic 8).

---

## Quick start

```bash
# 1. Clone and create a Python 3.12 virtualenv
git clone https://github.com/VytCepas/credit_default_risk_assessment.git
cd credit_default_risk_assessment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py
# → opens at http://localhost:8501
```

Trained artefacts (`src/assets/top25_risk_model.pkl`,
`src/assets/behavioral_traits_model.pkl`) ship with the repo, so the app boots
without a training step. To **re-train** from the raw Kaggle parquets, see
[SETUP.md](SETUP.md).

### Optional: reactive notebooks

```bash
.venv/bin/marimo edit marimo/risk_default_analysis.py   # full experiment chain
.venv/bin/marimo edit marimo/top25_squeeze.py           # what-if playground
```

---

## Repository layout

```
.
├── app.py                       Streamlit entry point — single Standard+ flow
├── models/                      ML model classes (no Streamlit imports)
│   ├── top25_predictor.py        25-field Standard+ inference wrapper
│   ├── insights.py               9 user-facing prediction surfaces (P-01…P-09)
│   └── behavioral_traits_model.py
├── src/
│   ├── assets/                  Trained .pkl artefacts + images
│   ├── components/              Streamlit UI components (forms, results)
│   └── predictors/              Streamlit-side cached model loaders
├── notebooks/
│   └── risk_default_analysis.ipynb     Authoritative Jupyter notebook
├── marimo/                      Reactive notebooks (Epic 9)
├── scripts/                     One-shot training / experiment / precompute scripts
│   ├── select_top25_features.py        Stage-1 feature selection
│   ├── squeeze_top25_accuracy.py       Stage-2 tuning → produces production model
│   ├── precompute_insights.py          Cohort + industry benchmark lookups
│   ├── run_e4_ctgan.py / run_e5_stacking.py
│   └── results/                        JSON measurement artefacts
├── tests/                       17 pytest tests (top25 predictor + insights)
├── data/                        Cached Kaggle parquets (gitignored)
├── docs/architecture.md         Module layout and code-organisation guide
└── project_docs/
    ├── adr/                     Architecture Decision Records
    ├── wiki/                    Source for the GitHub Wiki
    ├── practical_{1,2,3}_report.md     TA-1/2/3 milestone reports
    └── Meetings                 Meeting minutes
```

See [`docs/architecture.md`](docs/architecture.md) for the rationale behind the
`models/` vs `src/predictors/` split and how a Streamlit page flows.

---

## CI / testing

Every push runs [GitHub Actions](.github/workflows/ci.yml):

1. **Lint** — `flake8 src/ models/ tests/` (line length 100, ignores E203/W503).
2. **Tests** — `pytest tests/ -v --cov=models --cov=src` (artifact: `coverage.xml`).

Coverage is *measured* but not yet *enforced* — see Sprint 3 retrospective
[P3](project_docs/practical_3_report.md#52-problems-identified-4-teamwork-problems-per-brief).

Run locally:

```bash
flake8 src/ models/ tests/ --max-line-length=100 --extend-ignore=E203,W503
pytest tests/ -v --cov=models --cov=src --cov-report=term-missing
```

---

## Documentation map

| Doc | Audience | What's inside |
|---|---|---|
| [README.md](README.md) | first visitor | This page |
| [SETUP.md](SETUP.md) | someone running the app or retraining | venv, data download, training, deployment |
| [CONTRIBUTING.md](CONTRIBUTING.md) | a contributor | branch/PR/commit/CI conventions |
| [CHANGELOG.md](CHANGELOG.md) | milestone reviewer | Sprint 1–3 highlights |
| [docs/architecture.md](docs/architecture.md) | someone reading code | module split, asset table, page flow |
| [GitHub Wiki](https://github.com/VytCepas/credit_default_risk_assessment/wiki) | extended reference | architecture, modeling pipeline, insights catalogue, risk register, glossary |
| [project_docs/adr/](project_docs/adr/) | architectural reviewer | ADR 0001 (tiered questionnaire); future ADRs |
| [project_docs/practical_3_report.md](project_docs/practical_3_report.md) | TA-3 examiner | estimation, CI, risk matrix, retrospective, Kaggle benchmark, expansion experiments |

---

## Team

| Member | Role | Contributions |
|---|---|---|
| Vytautas Čepas ([@VytCepas](https://github.com/VytCepas)) | Lead engineer | Modeling, app, CI, docs |
| Laurynas Žalaga ([@Gitlaurynas](https://github.com/Gitlaurynas)) | Team member | Behavioural traits design, retrospective sign-off (pending) |

> Single-contributor reality is tracked openly as risk **R-V1** in the
> [risk register](project_docs/practical_3_report.md#41-vytautas--3-risks).

---

## License & data

Source code is released for academic review of the term project. The
**Home Credit Default Risk dataset** is © Home Credit Group and is governed by
the [Kaggle competition terms](https://www.kaggle.com/competitions/home-credit-default-risk/rules);
it is **not** redistributed in this repository — download it yourself from
Kaggle and place the parquets under `data/` (see [SETUP.md](SETUP.md)).
