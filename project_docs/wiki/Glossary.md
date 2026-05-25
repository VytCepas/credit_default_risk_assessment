# Glossary

## Domain

| Term | Meaning |
|---|---|
| **AUC / ROC-AUC** | Area under the receiver operating characteristic curve. 0.5 = random, 1.0 = perfect ranking. We optimise this. |
| **Brier score** | Mean squared error of probability predictions. Lower = better-calibrated probabilities. ROC-AUC measures *ranking*; Brier measures *probability quality*. |
| **Calibration** | Adjusting probability outputs so that "30 %" predictions actually default ~30 % of the time. Platt scaling, isotonic regression. |
| **DTI** | Debt-to-income ratio. `loan_annuity / total_income`. |
| **EXT_SOURCE_1/2/3** | Anonymised external credit scores from the Home Credit data. Highly predictive but not self-reportable. |
| **Home Credit** | The lender that released the Kaggle dataset. Czech-headquartered consumer lender. |
| **PD** | Probability of default. The thing the model estimates. |
| **SHAP** | SHapley Additive exPlanations. Per-prediction feature attribution. We use `shap.TreeExplainer` on the LightGBM. |
| **SMOTE / SMOTETomek** | Synthetic Minority Oversampling TEchnique. Interpolates new positive-class rows. SMOTETomek adds a cleanup step (Tomek links) that removes ambiguous samples. |
| **Tier (1/2/3)** | The Quick / Standard+ / Extended questionnaire variants from [ADR 0001](https://github.com/VytCepas/credit_default_risk_assessment/blob/main/project_docs/adr/0001_tiered_questionnaire.md). |

## Project / process

| Term | Meaning |
|---|---|
| **ADR** | Architecture Decision Record. See [ADR Index](ADR-Index). |
| **CTGAN** | Conditional Tabular GAN (Xu et al. NeurIPS 2019). Used in experiment E4 to model the joint distribution of minority-class rows. |
| **DI / FR / NFR / US** | Backlog item types in our planning vocabulary — Design/Implementation, Functional Requirement, Non-Functional Requirement, User Story. |
| **E1 … E5** | The numbered model-expansion experiments. See [Modeling Pipeline](Modeling-Pipeline). |
| **Epic** | Multi-sprint workstream (E1 Data, E2 Features, …, E9 Marimo). |
| **MoSCoW** | Must / Should / Could / Won't priority labels. |
| **P-01 … P-09** | The nine user-facing prediction surfaces. See [Insights Catalogue](Insights-Catalogue). |
| **SP** | Story Points — relative effort estimate. Fibonacci scale (1, 2, 3, 5, 8, 13). |
| **Squeeze model** | The Stage-2 production model produced by `scripts/squeeze_top25_accuracy.py` (25 features + 6 ratios + RandomizedSearchCV → 0.7146 ROC-AUC). |
| **Standard+ tier** | The default — and currently only — UI flow. 22 user-visible questions → 25 model features. |
| **TA-1 / TA-2 / TA-3** | The three course defence milestones (2026-05-14 / -05-21 / -05-28). |

## Tooling

| Term | Meaning |
|---|---|
| **flake8** | Python linter we run in CI Phase 1. |
| **GitHub Actions** | Our CI runner. |
| **imblearn.Pipeline** | The scikit-learn-compatible pipeline class that supports resamplers in intermediate steps without leakage. |
| **LightGBM** | Histogram-based gradient boosting library. Production estimator. |
| **marimo** | Reactive Python notebook (`marimo edit`, `marimo run`) stored as `.py`. Epic 9. |
| **pytest-cov** | Coverage plugin for pytest. Emits `coverage.xml` for upload. |
| **Streamlit** | The UI framework. |
| **XGBoost** | Alternative GBDT library. Used in E5 stacking experiment. |
