"""ML model classes, training pipelines, and analytical helpers.

Modules:

- :mod:`models.risk_model` — legacy 15-field production pipeline
  (``RiskModel``, ``QuestionnaireToFeatures``, ``DataPreprocessor``).
- :mod:`models.top25_predictor` — Stage-2 squeeze model wrapper for the
  25-field Standard+ tier (ADR 0001).
- :mod:`models.insights` — user-facing prediction surfaces from ADR 0002
  (counter-factuals, cohort percentile, loan-affordability, etc.).
- :mod:`models.behavioral_traits_model` — behavioural-traits classifier
  training pipeline.

Streamlit cache wrappers around these live in :mod:`src.predictors`.
"""
