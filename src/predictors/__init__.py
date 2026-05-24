"""Streamlit-facing thin wrappers around the ML model classes in ``models/``.

This package contains *prediction loaders* — functions that:

- locate trained model artefacts under ``src/assets/``
- load them once and cache via :func:`streamlit.cache_resource`
- expose a small predict/explain surface used by ``app.py``

Pure model code (training pipelines, ``QuestionnaireToFeatures``,
``Top25Predictor``, the insights catalogue, behavioural-traits training)
lives in the project-root ``models/`` package — predictors here import
from there.
"""
