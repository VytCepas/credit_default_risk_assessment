# Home Credit Default Risk Assessment — Wiki

> Source of this wiki lives in the repo at [`project_docs/wiki/`](https://github.com/VytCepas/credit_default_risk_assessment/tree/main/project_docs/wiki). Edit there, then sync (see [Contributing](../blob/main/CONTRIBUTING.md#wiki)).

Welcome. This is the extended reference for the credit default risk
assessment project — a Streamlit + LightGBM + SHAP product built on the
Kaggle Home Credit dataset, delivered as the term project for the *AI Product
Development* course at Vilnius University.

The [README](https://github.com/VytCepas/credit_default_risk_assessment#readme)
is the front door; this wiki is the depth.

## Where to start

| If you want to… | Open |
|---|---|
| Understand what was built and why | [Project Overview](Project-Overview) |
| See how the code is organised | [Architecture](Architecture) |
| Understand how the model is trained | [Modeling Pipeline](Modeling-Pipeline) |
| Know which 25 fields the user fills | [Standard+ Questionnaire](Standard-Plus-Questionnaire) |
| See the 9 user-facing insights | [Insights Catalogue](Insights-Catalogue) |
| Run / extend CI or write tests | [CI and Testing](CI-and-Testing) |
| Check what risks we track and how | [Risk Register](Risk-Register) |
| Know what's coming next | [Roadmap](Roadmap) |
| Read decision records | [ADR Index](ADR-Index) |
| Look up a term | [Glossary](Glossary) |

## Headline numbers

| Model | ROC-AUC | Status |
|---|---|---|
| **Production (Standard+ tier)** | **0.7146** | shipping |
| Earlier baseline (15 fields, GBM) | 0.6272 | retired |
| Unconstrained baseline (E1, ~104 features) | 0.7589 | research |
| Kaggle median | ~0.75 | reference |
| Kaggle 1st place | ~0.806 | reference |

See [Modeling Pipeline](Modeling-Pipeline) for the full E1–E5 expansion
chain and the gap-analysis story.

## Milestones

| Milestone | Defence | Report |
|---|---|---|
| TA-1 — Problem & feasibility | 2026-05-14 | [practical_1_report.md](https://github.com/VytCepas/credit_default_risk_assessment/blob/main/project_docs/practical_1_report.md) |
| TA-2 — Model development | 2026-05-21 | [practical_2_report.md](https://github.com/VytCepas/credit_default_risk_assessment/blob/main/project_docs/practical_2_report.md) |
| TA-3 — System integration + CI/CD + risk | 2026-05-28 | [practical_3_report.md](https://github.com/VytCepas/credit_default_risk_assessment/blob/main/project_docs/practical_3_report.md) |
