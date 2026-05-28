# Changelog

All notable changes since project kickoff. Dates use the project's `currentDate`
calendar (academic 2026 cycle).

## [TA-3 — Sprint 3] — 2026-05-28

### Added
- **Standard+ tier** (25 self-reportable fields + 6 ratios, LightGBM, RandomizedSearchCV-tuned). Production ROC-AUC **0.7146**.
  - `models/top25_predictor.py`, `src/components/questionnaire_top25.py`, `src/assets/top25_risk_model.pkl`
- **9 user-facing insights** (P-01…P-09 from ADR 0002) — counter-factuals, approval probability, cohort percentile, industry/region benchmark, affordability sandbox, recommended max loan, time-to-improvement, approval-time, risk decomposition.
- **Behavioural-traits tab** integrated into the result page.
- **Marimo notebooks** (Epic 9 partial) — `marimo/risk_default_analysis.py` (port) + `marimo/top25_squeeze.py` (what-if playground).
- **ADR 0001** — Tiered Questionnaire Strategy (Quick / Standard+ / Extended + derived layer).
- **Model expansion experiments** E1–E5 measured and recorded in Practical 3 §7.
- **Kaggle benchmark reference** with gap analysis (Practical 3 §6).
- **Risk register** (6 risks, P×I matrix) + **Sprint 3 retrospective**.
- **CI pipeline** — `.github/workflows/ci.yml` with lint + test phases.
- **17 unit/integration tests** across `tests/test_top25_predictor.py` and `tests/test_insights.py`.
- **CONTRIBUTING.md**, **CHANGELOG.md**, README rewrite, SETUP.md refresh.

### Changed
- App refactored to a **single Standard+ flow** (landing → 25-question form → result page with 6 insight tabs).
- `requirements.txt` adds `xgboost`, `ctgan`, `marimo`.

### Removed
- Legacy 15-field flow: `models/risk_model.py`, `src/components/questionnaire.py`, `src/predictors/risk_predictor.py`, `src/assets/risk_model.pkl`, `tests/test_predictor.py`, `tests/test_preprocessing.py`.

## [TA-2 — Sprint 2] — 2026-05-21

### Added
- GBM training pipeline (`models/risk_model.py`, since retired).
- SMOTETomek resampling with leakage-safe pipeline composition.
- Threshold optimisation (F1-max).
- SHAP TreeExplainer integration.
- 15-field questionnaire UI.
- Behavioural-traits model (Laurynas).
- GBM vs LightGBM comparison + 5-fold stratified CV (PR #64).

## [TA-1 — Sprint 1] — 2026-05-14

### Added
- Repo skeleton and project specification.
- EDA notebook on `application_train`.
- Initial backlog (16 items) and GitHub Issues.
- Feasibility analysis using the Kaggle Home Credit dataset.

---

For per-PR detail run `git log --oneline --merges` or browse
[GitHub PRs](https://github.com/VytCepas/credit_default_risk_assessment/pulls?q=is%3Apr+is%3Aclosed).
