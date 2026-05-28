# Architecture

This page mirrors and extends [`docs/architecture.md`](https://github.com/VytCepas/credit_default_risk_assessment/blob/main/docs/architecture.md). The repo doc is authoritative; this page is the searchable, hyperlinked tour.

## Top-level layout

```
credit_default_risk_assessment/
├── app.py                     Streamlit entry point — single Standard+ flow
├── models/                    ML model classes (no Streamlit imports)
│   ├── top25_predictor.py        Standard+ inference wrapper
│   ├── insights.py               9 user-facing prediction surfaces
│   └── behavioral_traits_model.py
├── src/
│   ├── assets/                Trained .pkl artefacts
│   ├── components/            Streamlit UI components (forms, results)
│   └── predictors/            Streamlit-side cached model loaders
├── notebooks/                 Jupyter (authoritative analysis)
├── marimo/                    Reactive notebooks (Epic 9)
├── scripts/                   One-shot ETL / experiment / precompute
│   └── results/               JSON measurement artefacts
├── tests/                     pytest suites
├── data/                      Cached Kaggle parquets (gitignored)
├── docs/architecture.md       This page's source
└── project_docs/              Reports, briefs, ADRs, wiki source
```

## The `models/` vs `src/predictors/` split

This is the one structural decision worth understanding.

- **`models/`** — pure model code. Training pipelines, transformer classes,
  inference wrappers that depend only on scikit-learn / LightGBM / XGBoost /
  ctgan, and the insights catalogue. **Nothing here imports Streamlit.**
  You can use these classes from a notebook, a script, a future REST API,
  or anywhere else.

- **`src/predictors/`** — Streamlit-side adapters. `@st.cache_resource`
  loaders that find the artefacts in `src/assets/`, instantiate the model
  classes from `models/`, and expose a small predict/explain surface to
  `app.py`.

The rule is one-way: `src/predictors/` may import from `models/`; `models/`
never imports from `src/`. Keeps the ML core deployable independently of
Streamlit.

## Trained artefacts

| File | Trained by | Used by |
|------|------------|---------|
| `src/assets/top25_risk_model.pkl` | `scripts/squeeze_top25_accuracy.py` | 25-field Standard+ flow (production) |
| `src/assets/behavioral_traits_model.pkl` | `models/behavioral_traits_model.py` | Behavioural-traits tab |
| `scripts/results/cohort_distributions.json` | `scripts/precompute_insights.py` | Cohort percentile insight (P-03) |
| `scripts/results/industry_region_benchmarks.json` | `scripts/precompute_insights.py` | Industry/region benchmark (P-04) |

Each `.pkl` is a self-contained bundle (model + encoder + feature list when
applicable). Replacing a `.pkl` is the deployment step; the code doesn't have
to change.

## How a Streamlit page flows

1. `app.py::show_*_page()` is invoked by the router (`current_page`).
2. The page imports its **form component** from `src/components/` (renders inputs, returns a dict on submit).
3. The page imports its **predictor wrapper** from `src/predictors/` (loads the model via `st.cache_resource`).
4. The page calls `predictor.predict(form_dict)`.
5. The page renders the prediction + any `models/insights.py` surfaces.

## CI scope

[`.github/workflows/ci.yml`](https://github.com/VytCepas/credit_default_risk_assessment/blob/main/.github/workflows/ci.yml) runs two phases on every push and on PRs against `main`:

1. **Lint** — `flake8 src/ models/ tests/` (line length 100, ignores E203/W503).
2. **Tests** — `pytest tests/ -v --cov=models --cov=src` with `coverage.xml` uploaded as the `coverage-report` artifact.

`app.py` is intentionally *not* in the lint scope (long Plotly/markup lines).
See [CI and Testing](CI-and-Testing) for the test inventory.
