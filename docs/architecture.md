# Project Architecture

Updated: 2026-05-25

This document explains where each piece of code lives and why. If you're new to the repo, start here. For a higher-level intro see [README.md](../README.md); for setup steps see [SETUP.md](../SETUP.md).

## Top-level layout

```
credit_default_risk_assessment/
├── app.py                     ← Streamlit entry point — single Standard+ flow
├── models/                    ← ML model classes & training pipelines
│   ├── top25_predictor.py        — 25-field Standard+ tier inference wrapper
│   ├── insights.py               — ADR 0002 prediction surfaces (P-01…P-09)
│   └── behavioral_traits_model.py — Behavioural-traits classifier training
├── src/
│   ├── assets/                ← Trained artefacts (.pkl), pictures
│   ├── components/            ← Streamlit UI components (forms, results panes)
│   │   ├── questionnaire_top25.py  — 25-field Standard+ form
│   │   ├── results.py              — Result display widgets (kept for reuse)
│   │   └── behavioral_traits.py    — Behavioural-traits result component
│   └── predictors/            ← Streamlit-side cached loaders that wrap models/
│       └── behavioral_predictor.py
├── notebooks/
│   └── risk_default_analysis.ipynb    ← Jupyter notebook (authoritative)
├── marimo/                            ← Marimo reactive notebooks (Epic 9)
│   ├── README.md
│   ├── risk_default_analysis.py       — Port of the Jupyter notebook
│   └── top25_squeeze.py               — Reactive playground for the production model
├── scripts/                   ← One-shot ETL / experiment / precompute scripts
│   ├── select_top25_features.py
│   ├── squeeze_top25_accuracy.py
│   ├── precompute_insights.py
│   ├── run_e4_ctgan.py
│   ├── run_e5_stacking.py
│   └── results/               ← JSON artefacts produced by the scripts above
├── tests/                     ← pytest suites
│   ├── test_top25_predictor.py
│   └── test_insights.py
├── data/                      ← Cached Kaggle parquets (+ pictures, gitignored)
└── project_docs/              ← Reports, briefs, ADRs (markdown)
```

> **Note:** the legacy 15-field flow (`risk_model.py`, `questionnaire.py`,
> `risk_predictor.py`, the legacy tests, and the `risk_model.pkl` artefact)
> was removed during the Practical 3 UX rework. The Standard+ tier is now
> the only flow the app exposes. See [CHANGELOG.md](../CHANGELOG.md) for the
> per-milestone history.

## The `models/` vs `src/predictors/` split

This is the one structural decision worth understanding.

- **`models/`** holds **pure model code**: training pipelines, transformer classes,
  inference wrappers that depend only on scikit-learn / LightGBM / XGBoost / ctgan,
  and the insights catalogue. Nothing in `models/` imports Streamlit. You can use
  these classes from a notebook, a script, a future REST API, or anywhere else.

- **`src/predictors/`** holds **Streamlit-side adapters**: `@st.cache_resource`
  loaders that find the artefacts in `src/assets/`, instantiate the model classes
  from `models/`, and expose a small predict/explain surface to `app.py`.

The rule is one-way: `src/predictors/` may import from `models/`; `models/` never
imports from `src/`. Keeps the ML core deployable independently of Streamlit.

## The `src/assets/` artefacts

| File | Trained by | Used by |
|------|------------|---------|
| `top25_risk_model.pkl` | `scripts/squeeze_top25_accuracy.py` | 25-field Standard+ flow (production) |
| `behavioral_traits_model.pkl` | `models/behavioral_traits_model.py` | Behavioural-traits tab |

Each `.pkl` is a self-contained bundle (model + encoder + feature list when
applicable). Replacing a `.pkl` is the deployment step; the code doesn't have
to change.

## How a Streamlit page typically flows

1. `app.py::show_*_page()` is invoked by the router (`current_page`).
2. The page imports its **form component** from `src/components/` (renders inputs,
   returns a dict on submit).
3. The page imports its **predictor wrapper** from `src/predictors/` (loads
   the model via `st.cache_resource`).
4. The page calls `predictor.predict(form_dict)`.
5. The page renders the prediction + any `models/insights.py` surfaces.

## ADRs

Architecture-shaping decisions go in `project_docs/adr/`:

- **[ADR 0001](../project_docs/adr/0001_tiered_questionnaire.md)** — Tiered Questionnaire Strategy (Quick / Standard+ / Extended +
  derived layer + bureau pull track).
- **[ADR 0002](../project_docs/adr/0002_insights_surfaces.md)** — Additional Prediction Surfaces (P-01…P-10 user-facing insights).

Future ADRs should follow the same numbered template.

## CI / tests

The CI workflow (`.github/workflows/ci.yml`) runs two phases on every push:

1. `flake8 src/ models/ tests/` (line length 100, ignores E203/W503 to play
   nicely with Black).
2. `pytest tests/ -v --cov=models --cov=src --cov-report=xml`.

`app.py` is intentionally *not* in the lint scope (long Plotly/markup lines).
