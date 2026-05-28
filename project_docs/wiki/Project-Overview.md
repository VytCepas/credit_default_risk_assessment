# Project Overview

## Problem

Credit default risk: given an applicant's profile, estimate the probability
they will default on a loan. The Kaggle Home Credit Default Risk dataset
(307,511 applicants × 122 columns, ~8 % default rate) is the public benchmark
we build against.

The product question that shapes our design: **can we collect enough signal
through a self-service questionnaire to be useful to either a loan officer or
an applicant in under 5 minutes?**

## Users

| Persona | What they want |
|---|---|
| **Applicant** (self-service) | "Will my loan be approved? What can I improve?" |
| **Loan officer** | "Quick sanity check on a candidate before pulling the bureau report." |
| **Course examiner** (TA-3) | "Can you justify your modeling, CI, and risk posture?" |

The MVP focuses on the applicant and the examiner. Loan-officer ergonomics
(Tier 3 of [ADR 0001](https://github.com/VytCepas/credit_default_risk_assessment/blob/main/project_docs/adr/0001_tiered_questionnaire.md)) is deferred.

## Solution shape

A Streamlit single-page application with one flow:

```
Landing  →  25-question form  →  Result page (tier badge + 6 insight tabs)
```

The result page surfaces:

- **Risk tier** — Low / Medium / High (score 0–1000)
- **Counter-factual** — "if you nudged X by Y, the score would drop by Z" (P-01)
- **Approval probability** with bootstrap confidence interval (P-02)
- **Cohort percentile** — "you're at the 67th percentile of applicants aged 30–40 earning €25–35k" (P-03)
- **Industry/region benchmark** (P-04)
- **Affordability sandbox** — slider over credit amount → live score (P-05)
- **Recommended max loan** via binary search (P-06)
- **Time-to-improvement** projection (P-07)
- **Approval-process-time** estimate (P-08)
- **Risk decomposition** Plotly pie by feature group (P-09)
- **Behavioural-traits profile** (Laurynas's complementary model)

## Technology choices

| Layer | Tool | Why |
|---|---|---|
| UI | Streamlit 1.50 | Fastest path from a Python model to a deployable UI; zero JS |
| Charts | Plotly 6 | Interactive without a frontend toolchain |
| ML | LightGBM 4.6 | Histogram-based GBDT, fast CPU training, top-Kaggle-solution choice |
| Explainability | SHAP 0.45 | Field standard for tree models; tab on every result |
| Imbalance | imbalanced-learn (SMOTETomek) | Tackles ~8 % positive rate without leakage when used inside `imblearn.Pipeline` |
| Tuning | RandomizedSearchCV | Cheaper than grid; reaches good params in 20–50 iter |
| Tabular GAN | CTGAN 0.10 | Experimental — see [Modeling Pipeline](Modeling-Pipeline#e4--ctgan) |
| Reactive notebooks | marimo 0.20 | Reactive cells + `.py` source format that diffs cleanly |
| CI | GitHub Actions | Free for public repos; native to GitHub |

## Constraints we accepted

1. **Self-reportable features only** (production model). EXT_SOURCE_* and bureau aggregates exist in the dataset but the user can't answer them — out of scope until a consented bureau-pull integration ships.
2. **CPU only**, local laptop. No paid cloud GPU spend. Kaggle's 30 GPU-h/week is the fallback if we ever need it.
3. **Two-person team** (one active). Tooling and process are sized accordingly — no Jira, no Kubernetes, no MLflow yet.

## Non-goals

- Real-money lending decisions. The system is a *prototype*; the UI explicitly frames the risk score as indicative, not underwriting.
- Beating Kaggle's leaderboard. We track the gap (see [Modeling Pipeline](Modeling-Pipeline)) but the binding constraint is the questionnaire, not the algorithm.
- Multi-language. English UI only.

## Team

| Member | Role | Notes |
|---|---|---|
| Vytautas Čepas ([@VytCepas](https://github.com/VytCepas)) | Lead engineer | Modeling, app, CI, docs |
| Laurynas Žalaga ([@Gitlaurynas](https://github.com/Gitlaurynas)) | Team member | Behavioural traits design; retrospective sign-off pending |

Single-contributor reality is tracked as risk **R-V1** — see [Risk Register](Risk-Register).
