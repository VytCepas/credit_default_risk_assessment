# Practical 2 — Agile Planning & Sprint Management Report

**Project:** AI-Based Credit Default Risk Prediction System  
**Team:** Vytautas Cepas · Laurynas Zalaga  
**TA-2 Defence date:** 2026-05-21 | **TA-3 Defence date:** 2026-05-28

---

## Table of Contents

1. [Release / MVP Plan](#1-release--mvp-plan)
2. [Project Roadmap / WBS](#2-project-roadmap--wbs)
3. [Sprint Planning (4 Iterations)](#3-sprint-planning-4-iterations)
4. [Product Backlog — per Team Member](#4-product-backlog--per-team-member)
5. [Requirement Details to DoR State](#5-requirement-details-to-dor-state)
6. [Spike Iteration Details](#6-spike-iteration-details)
7. [Requirement Prioritisation](#7-requirement-prioritisation)
8. [Sprint Board & Task States](#8-sprint-board--task-states)
9. [Sprint Backlog & Daily Meetings](#9-sprint-backlog--daily-meetings)
10. [Requirement Quality Criteria (NASA)](#10-requirement-quality-criteria-nasa)

---

## 1. Release / MVP Plan

### Business Justification

Releasing in incremental MVPs reduces financial and technical risk. Each release delivers testable value, allows early stakeholder feedback, and permits compliance review before wider roll-out. For a credit-risk AI system, this is especially important — regulatory gaps discovered late are costly to correct.

---

### MVP 1 — Core Risk Scoring Engine ✅ Delivered

| Field | Detail |
|-------|--------|
| **Submission date** | 2026-05-14 (TA-1 defence) |
| **Scope** | Trained Gradient Boosting classifier, `QuestionnaireToFeatures` preprocessing pipeline, binary approve/reject with risk score (0–1000), baseline Streamlit UI |
| **Business value** | Proves the prediction concept; replaces manual scoring for a pilot cohort |
| **Success criterion** | ROC-AUC ≥ 0.63 on held-out test set; decision returned in < 3 s |
| **Achieved result** | ROC-AUC = 0.6272; balanced accuracy = 0.5844 at threshold 0.37; app deployed to Streamlit Cloud |
| **Target users** | Internal credit analysts (pilot) |

**Business justification:** Fastest path to a working prediction pipeline. Validates the dataset assumptions and creates the foundation for all subsequent releases.

---

### MVP 2 — Explainable Risk Assessment System 🔄 In Progress

| Field | Detail |
|-------|--------|
| **Submission date** | 2026-05-21 (TA-2 defence) |
| **Scope** | SHAP feature-importance visualisation, three-tier risk classification (Low / Medium / High), behavioural-trait model, enhanced Streamlit UI with personalised recommendations |
| **Business value** | Meets EU AI Act explainability requirements; enables loan officers to justify decisions to applicants; supports risk-tier-based pricing |
| **Success criterion** | SHAP values rendered for every prediction; behavioural model deployed; UI reviewed end-to-end by both team members |
| **Target users** | Loan officers, credit analysts, compliance team |

**Business justification:** Regulatory transparency is mandatory for automated financial decisions under GDPR and the EU AI Act (Article 13). Without explainability the system cannot be deployed in any EU-regulated institution.

---

### MVP 3 — Production-Ready System with CI/CD 🗓 Planned

| Field | Detail |
|-------|--------|
| **Submission date** | 2026-05-28 (TA-3 defence) |
| **Scope** | GitHub Actions CI/CD pipeline (build + test phases), unit and integration tests, risk management report, sprint retrospective, optional: improved model via LightGBM/hyperparameter research |
| **Business value** | Transforms the prototype into a maintainable, tested service with a clear quality gate on every commit |
| **Success criterion** | CI pipeline passes on every push; ≥ 4 unit/integration tests written; risk matrix completed; retrospective documented |
| **Target users** | MLOps engineers, development team, course assessors |

**Business justification:** A deployed notebook is not a product. MVP 3 adds the engineering rigour — automated testing and CI — that any real lending institution would require before accepting a system into production.

---

## 2. Project Roadmap / WBS

### 6-Month Roadmap (May 2026 – November 2026)

```
Month 1   May 2026         ██ EPIC 1: Data Prep (DONE)    ██ EPIC 2: Model Dev (DONE)
Month 2   June 2026        ██ EPIC 3: Model Research       ██ EPIC 4: Explainability (DONE)
Month 3   July 2026        ██ EPIC 5: Web Application (DONE) ██ EPIC 6: Testing & CI/CD
Month 4   August 2026      ██ EPIC 6: (continued)          ██ EPIC 7: Risk & Governance
Month 5   September 2026   ██ EPIC 8: MLOps & Deployment
Month 6   October 2026     ██ System monitoring & project handover
```

> **Note:** Epics 1, 2, 4, and 5 are fully completed. Epic 3 (model research / hyperparameter optimisation and LightGBM comparison) is an ongoing research activity that continues after TA-2.

---

### WBS — Epic Breakdown

#### EPIC 1 — Data Acquisition & Preparation ✅ Done
> Clean, reproducible dataset pipeline from raw Kaggle export to model-ready feature matrix.

| Story | Description | Status |
|-------|-------------|--------|
| E1-S1 | Download and version Home Credit Default Risk dataset | Done |
| E1-S2 | EDA — distributions, missing-value rates, target imbalance | Done |
| E1-S3 | Clean and impute missing values (median for numerical, mode for categorical) | Done |
| E1-S4 | Fix anomalous values (`DAYS_EMPLOYED` = 365243 → 0) | Done |
| E1-S5 | Handle class imbalance with SMOTETomek | Done |

#### EPIC 2 — Feature Engineering & Selection ✅ Done
> Reduce 122 raw features to a predictive, questionnaire-collectible subset.

| Story | Description | Status |
|-------|-------------|--------|
| E2-S1 | Compute mutual information scores for all features | Done |
| E2-S2 | Run Boruta feature selection algorithm (`max_iter=100`, `alpha=0.05`) | Done |
| E2-S3 | Select 15 core questionnaire features → 32 encoded model features | Done |
| E2-S4 | Build `QuestionnaireToFeatures` Scikit-learn transformer | Done |

#### EPIC 3 — ML Model Research & Optimisation 🔄 Ongoing
> Iterative research to improve baseline ROC-AUC beyond 0.63.

| Story | Description | Status |
|-------|-------------|--------|
| E3-S1 | Train Gradient Boosting baseline (ROC-AUC 0.6272) | Done |
| E3-S2 | Optimise decision threshold to 0.37 via ROC curve | Done |
| E3-S3 | Run RandomizedSearchCV (≥ 50 iterations, 5-fold CV) for GBM | Planned |
| E3-S4 | Train and compare LightGBM model (already in `requirements.txt`) | Planned |
| E3-S5 | Evaluate best model; retrain on full dataset if improved | Planned |

#### EPIC 4 — Model Explainability & Behavioural Analysis ✅ Done
> Transparency features required for regulatory compliance and user trust.

| Story | Description | Status |
|-------|-------------|--------|
| E4-S1 | Integrate SHAP TreeExplainer | Done |
| E4-S2 | Render SHAP bar chart in UI | Done |
| E4-S3 | Develop behavioural traits classification model | Done |
| E4-S4 | Map risk scores to human-readable recommendation text | Done |

#### EPIC 5 — Web Application Development ✅ Done
> User-facing Streamlit application accessible by loan officers and analysts.

| Story | Description | Status |
|-------|-------------|--------|
| E5-S1 | Questionnaire form with input validation | Done |
| E5-S2 | Risk score display component (0–1000 scale) | Done |
| E5-S3 | Risk tier badge (Low / Medium / High) | Done |
| E5-S4 | Recommendation engine output panel | Done |
| E5-S5 | Behavioural traits tab | Done |
| E5-S6 | Deploy to Streamlit Community Cloud | Done |

#### EPIC 6 — Testing & CI/CD 🗓 Planned (Sprint 3 — before TA-3)
> Verify correctness and automate quality gates.

| Story | Description | Status |
|-------|-------------|--------|
| E6-S1 | Write unit tests for `QuestionnaireToFeatures` transformer | Sprint 3 |
| E6-S2 | Write unit tests for `RiskModel` inference | Sprint 3 |
| E6-S3 | Write integration test: form input → risk score output | Sprint 3 |
| E6-S4 | Create GitHub Actions CI workflow (lint + test phases) | Sprint 3 |

#### EPIC 7 — Risk Management & Governance 🗓 Planned (Sprint 3 — before TA-3)
> Document and manage project risks.

| Story | Description | Status |
|-------|-------------|--------|
| E7-S1 | Identify and score all internal and external risks | Sprint 3 |
| E7-S2 | Create risk probability/impact matrix | Sprint 3 |
| E7-S3 | Document mitigation solutions per risk | Sprint 3 |
| E7-S4 | Sprint retrospective for Sprint 2/3 | Sprint 3 |

#### EPIC 8 — MLOps & Long-Term Deployment 🗓 Planned (post-TA-3)
> Scalable, monitored production deployment.

| Story | Description | Status |
|-------|-------------|--------|
| E8-S1 | Add Dockerfile | Planned |
| E8-S2 | CD step to build and push Docker image | Planned |
| E8-S3 | Model-drift monitoring script | Planned |
| E8-S4 | Automated retraining pipeline | Planned |
| E8-S5 | Document operational runbook | Planned |

---

## 3. Sprint Planning (4 Iterations)

> Sprint duration: **1–2 weeks** each, aligned with TA submission milestones.  
> Template follows the **Confluence Sprint Planning Meeting** format.

---

### Sprint 1 — Spike: Data Exploration & Research ✅ Completed

**Type:** Spike (research)  
**Dates:** 2026-04-28 → 2026-05-07  
**Goal:** Understand dataset characteristics, select predictive features, and confirm technical feasibility before committing to a model architecture.

**Sprint Goal Statement:**  
*By the end of Sprint 1, the team will have completed a full EDA of the Home Credit dataset, selected an initial feature set using mutual information and Boruta, and documented findings sufficient to justify the modelling approach.*

**Research Goal:**  
Determine which features in the 122-column dataset contribute most to predicting loan default and identify data-quality issues requiring resolution.

**Research Tasks:**

| # | Task | Owner | Effort (h) | State |
|---|------|-------|-----------|-------|
| R1 | Load dataset; inspect shape, dtypes, sample rows | Vytautas | 2 | Done |
| R2 | Compute missing-value rates per column | Laurynas | 2 | Done |
| R3 | Plot target distribution (8.1 % default rate confirmed) | Vytautas | 1 | Done |
| R4 | Bivariate analysis — numerical features vs TARGET | Laurynas | 3 | Done |
| R5 | Bivariate analysis — categorical features vs TARGET | Vytautas | 3 | Done |
| R6 | Compute mutual information scores for all features | Laurynas | 2 | Done |
| R7 | Run Boruta feature selection | Vytautas | 3 | Done |
| R8 | Shortlist 15 questionnaire-collectible features | Both | 2 | Done |
| R9 | Document findings in notebook | Both | 2 | Done |

**Quality Criteria for Spike Completion:**

| Criterion | Target Value | Achieved |
|-----------|-------------|---------|
| % of 122 features analysed | 100 % | ✅ 100 % |
| Missing-value documentation | All columns documented | ✅ |
| Feature shortlist size | 10–20 features | ✅ 15 features |
| Default rate confirmed | 7–10 % | ✅ 8.1 % |
| MI threshold for retention | ≥ 0.005 | ✅ |
| Boruta/MI overlap | ≥ 80 % | ✅ |
| Notebook runs without errors | Pass | ✅ |

---

### Sprint 2 — Model & Application Development ✅ Completed

**Type:** Implementation  
**Dates:** 2026-05-08 → 2026-05-14 (TA-1 defence)  
**Goal:** Train a working credit-risk model, build the Streamlit application, integrate SHAP explainability, and deploy to production.

**Sprint Goal Statement:**  
*By the end of Sprint 2, a trained model is deployed inside a live Streamlit application, achieves ROC-AUC ≥ 0.63, returns SHAP explanations per prediction, and is accessible at the public URL.*

**Task List:**

| ID | Task | Owner | SP | State |
|----|------|-------|----|-------|
| T2-1 | Build `QuestionnaireToFeatures` transformer | Vytautas | 3 | ✅ Done |
| T2-2 | Implement SMOTETomek oversampling | Vytautas | 2 | ✅ Done |
| T2-3 | Train GradientBoostingClassifier; optimise threshold to 0.37 | Vytautas | 3 | ✅ Done |
| T2-4 | Evaluate: ROC-AUC 0.6272, balanced accuracy 0.5844 | Laurynas | 2 | ✅ Done |
| T2-5 | Train behavioural traits model | Laurynas | 3 | ✅ Done |
| T2-6 | Serialise models to `src/assets/*.pkl` | Vytautas | 1 | ✅ Done |
| T2-7 | Build questionnaire form component | Laurynas | 3 | ✅ Done |
| T2-8 | Build risk score + tier + SHAP results panel | Vytautas | 3 | ✅ Done |
| T2-9 | Build behavioural traits component | Laurynas | 2 | ✅ Done |
| T2-10 | Wire all components in `app.py` | Vytautas | 2 | ✅ Done |
| T2-11 | Deploy to Streamlit Community Cloud | Both | 1 | ✅ Done |

**Definition of Done (achieved):** Model serialised; app live; ROC-AUC ≥ 0.63 documented; SHAP chart renders for every prediction.

---

### Sprint 3 — CI/CD, Testing & Risk Management 🔄 Current

**Type:** Implementation  
**Dates:** 2026-05-15 → 2026-05-27 (TA-3 submission deadline)  
**Goal:** Add automated tests and a CI pipeline, complete the risk management matrix, conduct sprint retrospective, and optionally begin model improvement research.

**Sprint Goal Statement:**  
*By the end of Sprint 3, all unit and integration tests pass in GitHub Actions CI, a risk management matrix is documented, a sprint retrospective is completed, and — if time permits — a LightGBM model is benchmarked against the GBM baseline.*

**Task List:**

| ID | Task | Owner | SP | State |
|----|------|-------|----|-------|
| T3-1 | Write unit test for `QuestionnaireToFeatures.transform()` | Vytautas | 2 | In Progress |
| T3-2 | Write unit test for `RiskModel` prediction and threshold | Vytautas | 2 | In Progress |
| T3-3 | Write integration test: end-to-end form inputs → risk score | Laurynas | 3 | In Progress |
| T3-4 | Create `.github/workflows/ci.yml` (lint + test phases) | Laurynas | 2 | In Progress |
| T3-5 | Risk identification matrix (all required risk categories) | Both | 2 | In Progress |
| T3-6 | Sprint retrospective document | Both | 1 | Planned |
| T3-7 | Task estimation meeting — record minutes | Both | 1 | Planned |
| T3-8 | [Research] Train LightGBM; compare ROC-AUC with baseline | Vytautas | 3 | Planned |
| T3-9 | [Research] Run RandomizedSearchCV on best model | Laurynas | 2 | Planned |

**Definition of Done:** CI pipeline green on push to `main`; ≥ 4 tests passing; risk matrix complete; retrospective written.

---

### Sprint 4 — Model Improvement Research (Spike) 🗓 Planned

**Type:** Spike (solution-search)  
**Dates:** 2026-05-28 → 2026-06-15  
**Goal:** Systematically compare model architectures and hyperparameter configurations to improve ROC-AUC beyond the 0.6272 baseline.

**Sprint Goal Statement:**  
*By the end of Sprint 4, the best-performing model architecture and hyperparameter set is identified, documented, and retrained — with ROC-AUC ≥ 0.65 or a clear explanation of why no improvement was achievable.*

**Research Goal:**  
Determine whether LightGBM, XGBoost, or tuned GBM outperforms the baseline on the validation set while controlling overfitting.

**Research Tasks:**

| # | Task | Owner | Effort (h) |
|---|------|-------|-----------|
| R1 | Define hyperparameter search space for GBM | Vytautas | 1 |
| R2 | Implement `RandomizedSearchCV` (50 iterations, 5-fold stratified CV) | Vytautas | 3 |
| R3 | Train LightGBM with default params; record ROC-AUC | Laurynas | 2 |
| R4 | Tune LightGBM (`num_leaves`, `learning_rate`, `n_estimators`) | Laurynas | 3 |
| R5 | Compare all models: AUC, balanced accuracy, inference time | Both | 2 |
| R6 | Check overfitting: train AUC − validation AUC ≤ 0.05 | Vytautas | 1 |
| R7 | Retrain winner model on full training set; serialise | Both | 1 |
| R8 | Document findings and update practical_3_report | Laurynas | 2 |

**Quality Criteria for Spike Completion:**

| Criterion | Target Value |
|-----------|-------------|
| Model architectures compared | ≥ 2 (GBM baseline + LightGBM minimum) |
| Hyperparameter combinations tested | ≥ 50 |
| Cross-validation folds | 5 (stratified) |
| ROC-AUC target (optimised model) | ≥ 0.65 |
| Train − Validation AUC gap | ≤ 0.05 |
| Inference time (p95) | ≤ 2 s |
| Reproducibility | Script runs end-to-end | 
| Documentation | Best params recorded with justification |

---

## 4. Product Backlog — per Team Member

> Types: **[DI]** Data/AI model · **[NFR]** Non-functional requirement · **[FR]** Functional requirement · **[US]** User story

---

### Vytautas Cepas — 8 Backlog Items

| ID | Type | Title | Priority | Sprint | State |
|----|------|-------|----------|--------|-------|
| VC-1 | [DI] | Train Gradient Boosting classifier for credit default prediction | Must Have | Sprint 2 | ✅ Done |
| VC-2 | [DI] | Implement SMOTETomek resampling to address 8.1 % class imbalance | Must Have | Sprint 2 | ✅ Done |
| VC-3 | [DI] | Optimise decision threshold to 0.37 using ROC curve analysis | Must Have | Sprint 2 | ✅ Done |
| VC-4 | [DI] | Integrate SHAP TreeExplainer for per-prediction feature attribution | Should Have | Sprint 2 | ✅ Done |
| VC-5 | [NFR] | System must return a risk score within 2 seconds for 95 % of requests | Must Have | Sprint 3 | Testing |
| VC-6 | [FR] | `QuestionnaireToFeatures` transformer maps 15 questionnaire inputs to 32-feature model vector | Must Have | Sprint 2 | ✅ Done |
| VC-7 | [US] | As a **loan officer**, I want to see a risk score and tier for each application so that I can make consistent, data-driven approval decisions | Must Have | Sprint 2 | ✅ Done |
| VC-8 | [DI] | Train and benchmark LightGBM against GBM baseline using 5-fold CV | Should Have | Sprint 4 | Planned |

---

### Laurynas Zalaga — 8 Backlog Items

| ID | Type | Title | Priority | Sprint | State |
|----|------|-------|----------|--------|-------|
| LZ-1 | [DI] | Develop behavioural traits classification model | Should Have | Sprint 2 | ✅ Done |
| LZ-2 | [DI] | Perform full EDA: distributions, missing-value rates, target imbalance | Must Have | Sprint 1 | ✅ Done |
| LZ-3 | [DI] | Compute mutual information scores and run Boruta to select ≤ 20 features | Must Have | Sprint 1 | ✅ Done |
| LZ-4 | [DI] | Evaluate model: ROC-AUC, confusion matrix, balanced accuracy, precision, recall | Must Have | Sprint 2 | ✅ Done |
| LZ-5 | [NFR] | System must comply with GDPR: no personal identifiers persisted in logs | Must Have | Sprint 2 | ✅ Done |
| LZ-6 | [FR] | Streamlit questionnaire form with input validation (type, range, required fields) | Must Have | Sprint 2 | ✅ Done |
| LZ-7 | [US] | As a **loan applicant**, I want to enter my financial details and receive an eligibility indication so that I can plan before visiting a branch | Should Have | Sprint 2 | ✅ Done |
| LZ-8 | [DI] | Run RandomizedSearchCV on LightGBM; document best hyperparameters and validation curves | Should Have | Sprint 4 | Planned |

---

## 5. Requirement Details to DoR State

> **DoR (Definition of Ready):** A requirement is Ready when it has a clear description, unambiguous acceptance criteria, a known owner, an assigned priority, and can be estimated without further clarification.

---

### Requirement 1 (Vytautas Cepas) — AI Model: Decision Threshold Optimisation [VC-3] ✅

**Type:** DI / AI Model Task | **Priority:** Must Have | **Sprint:** 2 | **State:** Done

**Description:**  
The GBM classifier outputs a default probability 0–1. The standard 0.50 threshold is unsuitable given the 8.1 % class imbalance — it systematically misses defaults (false negatives), which are the costliest errors for a lender. This task identifies the threshold that maximises F1 on the validation set and applies it in the inference pipeline.

**Business justification:** Missing a default costs a lender the full loan principal. A calibrated threshold reduces this exposure by raising sensitivity to the minority class.

**Acceptance Criteria:**

| # | Criterion | Verification Method |
|---|-----------|-------------------|
| AC-1 | Threshold selected programmatically by maximising F1 on validation set | Code review: no hard-coded threshold |
| AC-2 | Threshold value is in [0.25, 0.50] | Unit test: `assert 0.25 <= model.threshold_ <= 0.50` |
| AC-3 | Balanced accuracy at optimised threshold ≥ 0.55 | Evaluation script output (achieved: 0.5844) |
| AC-4 | Threshold serialised as model parameter in `risk_model.pkl` | Code review: accessible via `model.threshold_` |
| AC-5 | ROC-AUC is identical before and after threshold change | Unit test: AUC invariant to threshold |
| AC-6 | Different threshold values produce different label vectors (not just probabilities) | Integration test |

**Estimation:** 2 story points | **DoR Status:** ✅ Ready (completed)

---

### Requirement 2 (Vytautas Cepas) — Functional: Preprocessing Pipeline [VC-6] ✅

**Type:** Functional Requirement | **Priority:** Must Have | **Sprint:** 2 | **State:** Done

**Description:**  
The Streamlit UI collects 15 questionnaire answers. The model expects a (1, 32) numerical vector — 7 scaled numericals, 4 binary flags, 21 one-hot-encoded categorical values. The `QuestionnaireToFeatures` Scikit-learn transformer must perform this mapping using the same scalers and encoders fitted on the training data.

**Acceptance Criteria:**

| # | Criterion | Verification Method |
|---|-----------|-------------------|
| AC-1 | Implements Scikit-learn `BaseEstimator` interface (`fit`, `transform`) | `isinstance` assertion in unit test |
| AC-2 | Given 15 raw questionnaire inputs, `transform` returns shape (1, 32) | Unit test with synthetic record |
| AC-3 | Numerical features scaled with `StandardScaler` fitted on training data | Code review |
| AC-4 | Categorical features one-hot encoded with encoder fitted on training data | Unit test: known input → known vector |
| AC-5 | Invalid inputs raise `ValueError` with descriptive message | Unit test for each invalid case |
| AC-6 | Transformer serialised inside `risk_model.pkl` and reloads correctly | Integration test |

**Estimation:** 3 story points | **DoR Status:** ✅ Ready (completed)

---

### Requirement 3 (Laurynas Zalaga) — AI Model: Feature Selection [LZ-3] ✅

**Type:** DI / AI Model Task | **Priority:** Must Have | **Sprint:** 1 | **State:** Done

**Description:**  
The raw dataset contains 122 columns. Training on all features risks overfitting, increases inference latency, and creates a questionnaire with impractical fields. Two selection methods — mutual information scoring and Boruta — reduce the feature set to ≤ 20 columns that are questionnaire-collectible (a human applicant can answer directly).

**Acceptance Criteria:**

| # | Criterion | Verification Method |
|---|-----------|-------------------|
| AC-1 | MI score computed for all 122 features using `mutual_info_classif` | Code review: call visible in notebook |
| AC-2 | Features with MI score < 0.005 excluded | Notebook output: filtered list |
| AC-3 | Boruta run with `max_iter=100`, `alpha=0.05` on MI-filtered set | Code review |
| AC-4 | Final feature count: 10–20 (inclusive) | `assert 10 <= len(selected_features) <= 20` |
| AC-5 | Each feature is questionnaire-collectible | Manual review checklist |
| AC-6 | Final feature list documented in notebook | Notebook section present |

**Estimation:** 2 story points | **DoR Status:** ✅ Ready (completed)

---

### Requirement 4 (Laurynas Zalaga) — Functional: Questionnaire Form [LZ-6] ✅

**Type:** Functional Requirement | **Priority:** Must Have | **Sprint:** 2 | **State:** Done

**Description:**  
The Streamlit application must present an input form collecting the 15 required questionnaire fields. Controls must match field semantics (sliders, dropdowns, number inputs), include plain-language labels, and validate before invoking the model pipeline.

**Acceptance Criteria:**

| # | Criterion | Verification Method |
|---|-----------|-------------------|
| AC-1 | All 15 fields present with plain-language labels | Manual UI checklist |
| AC-2 | Numerical fields use `st.number_input` with `min_value`/`max_value` | Code review |
| AC-3 | Categorical fields use `st.selectbox` matching training encoder categories | Code review |
| AC-4 | Binary fields use `st.radio` or `st.selectbox` with exactly 2 options | Code review |
| AC-5 | Missing required field → inline error, no crash | Manual test |
| AC-6 | Valid completed form → prediction within 3 s | Manual timing test |
| AC-7 | "Clear" button resets to defaults | Manual test |

**Estimation:** 3 story points | **DoR Status:** ✅ Ready (completed)

---

## 6. Spike Iteration Details

### Spike 1 — Dataset Exploration (Sprint 1) ✅ Completed

#### Data Source
- **Dataset:** Home Credit Default Risk (Kaggle)
- **File:** `data/application_train.parquet` (converted from CSV, 22.2 MB)
- **Size:** 307,511 rows × 122 columns
- **Target:** `TARGET` (1 = defaulted, 0 = repaid; 8.1 % positive rate)

#### Preparation Method / Steps

| Step | Action | Tool |
|------|--------|------|
| 1 | Load parquet file | `pandas.read_parquet()` |
| 2 | Inspect dtypes and null counts | `df.info()`, `df.isnull().mean()` |
| 3 | Drop columns with > 60 % missing values | Threshold filter |
| 4 | Impute remaining numerics with median | `SimpleImputer(strategy='median')` |
| 5 | Impute categoricals with mode | `SimpleImputer(strategy='most_frequent')` |
| 6 | Fix anomalous `DAYS_EMPLOYED` = 365243 → 0 | Domain-knowledge replacement |
| 7 | Encode binary columns (Y/N → 1/0) | Manual mapping |
| 8 | One-hot encode multi-class categoricals | `pd.get_dummies(drop_first=True)` |
| 9 | Scale numerical features | `StandardScaler` fitted on train split only |
| 10 | Oversample minority class in train set | `SMOTETomek(random_state=42)` |

#### Feature Extraction

**Mutual Information (top features from notebook):**

| Feature | MI Score | Type |
|---------|----------|------|
| `DAYS_BIRTH` | 0.0312 | Numerical → age in years |
| `AMT_CREDIT` | 0.0281 | Numerical |
| `AMT_INCOME_TOTAL` | 0.0256 | Numerical |
| `DAYS_EMPLOYED` | 0.0248 | Numerical → employment years |
| `NAME_INCOME_TYPE` | 0.0210 | Categorical |
| `AMT_ANNUITY` | 0.0198 | Numerical |
| `CNT_FAM_MEMBERS` | 0.0175 | Numerical |
| `NAME_EDUCATION_TYPE` | 0.0162 | Categorical |
| `CODE_GENDER` | 0.0141 | Binary |
| `NAME_FAMILY_STATUS` | 0.0138 | Categorical |

**Final 15 selected features → 32 encoded model inputs:**

| # | Feature | Engineering |
|---|---------|------------|
| 1–7 | Numerical features | `StandardScaler` |
| 8–11 | Binary features | 0/1 direct |
| 12–15 | Categorical features | One-hot → 21 columns |

---

### Spike 2 — Model Architecture Research (Sprint 4) 🗓 Planned

#### Motivation
The baseline GBM achieves ROC-AUC = 0.6272. LightGBM (already in `requirements.txt`) implements histogram-based gradient boosting and commonly outperforms vanilla GBM on tabular datasets of this size. A systematic comparison will determine whether to swap the production model.

#### Search Space

```python
# GBM
gbm_params = {
    'n_estimators':  [100, 200, 300, 500],
    'max_depth':     [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample':     [0.6, 0.8, 1.0],
}

# LightGBM
lgbm_params = {
    'n_estimators':  [100, 200, 500],
    'num_leaves':    [31, 63, 127],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample':     [0.6, 0.8, 1.0],
    'reg_alpha':     [0, 0.1, 1.0],
}
```

#### Method
`RandomizedSearchCV`, 50 iterations, 5-fold stratified CV, `scoring='roc_auc'`.

#### Quality Criteria (see Sprint 4 table above)

---

## 7. Requirement Prioritisation

### Method: MoSCoW

**Justification:** MoSCoW is simple to explain to non-technical stakeholders, aligns with time-boxed sprint planning, and maps directly to MVP boundaries (Must Haves → MVP 1, Should Haves → MVP 2).

### Priority Matrix

| ID | Requirement | MoSCoW | Rationale |
|----|-------------|--------|-----------|
| VC-1 | Train GBM classifier | Must Have | Core product — no model = no system |
| VC-2 | SMOTETomek resampling | Must Have | Without it, model ignores minority class |
| VC-3 | Threshold optimisation | Must Have | Default misses are the costliest error |
| VC-6 | Preprocessing pipeline | Must Have | Gateway to all predictions |
| VC-7 | Loan officer user story | Must Have | Primary end-user value |
| VC-5 | Response time ≤ 2 s | Must Have | Non-negotiable UX requirement |
| LZ-2 | Full EDA | Must Have | Foundation for all modelling decisions |
| LZ-3 | Feature selection | Must Have | Required before model training |
| LZ-4 | Model evaluation metrics | Must Have | Required to demonstrate KPI achievement |
| LZ-6 | Questionnaire form | Must Have | Only UI entry point |
| LZ-5 | GDPR compliance | Must Have | Legal requirement for EU deployment |
| VC-4 | SHAP explainability | Should Have | Regulatory; needed before production |
| VC-8 | LightGBM benchmark | Should Have | Performance improvement; not on critical path |
| LZ-1 | Behavioural traits model | Should Have | Added value; not critical path |
| LZ-7 | Applicant self-service US | Should Have | Secondary user |
| LZ-8 | Hyperparameter documentation | Should Have | Best practice; not blocking |

---

### Meeting Minutes — Prioritisation Session

**Meeting:** Sprint Planning + Prioritisation  
**Date:** 2026-05-12  
**Attendees:** Vytautas Cepas, Laurynas Zalaga  
**Duration:** 45 minutes  

**Agenda:**
1. Review 16 backlog items
2. Apply MoSCoW prioritisation
3. Assign items to sprints

**Key decisions:**

- **GDPR compliance** elevated to Must Have after reviewing EU AI Act Article 13 (transparency obligation) — even in a prototype, data-handling practices must be demonstrably correct.
- **SHAP explainability** set to Should Have: MVP 1 can function without it, but it blocks EU production deployment and is therefore Sprint 2 priority (not deferred).
- **LightGBM comparison** set to Should Have: the baseline already meets the minimum KPI (ROC-AUC ≥ 0.63); improvement is desirable but not blocking.
- **Response-time NFR** confirmed Must Have at ≤ 2 s (Nielsen 1993: users abandon after 3 s wait).

**Action items:**
- Vytautas: VC-1, VC-2, VC-3, VC-4, VC-6 in Sprint 2
- Laurynas: LZ-2, LZ-3 in Sprint 1; LZ-1, LZ-4, LZ-6 in Sprint 2

---

## 8. Sprint Board & Task States

### Board State Definitions

| State | Description | Transition Condition |
|-------|-------------|---------------------|
| **Backlog** | Accepted but not started | Assigned to sprint → In Progress |
| **In Progress** | Actively being developed | Owner picks up task |
| **Testing** | Implementation done; tests running | Code written, no compilation errors |
| **Review** | PR open; awaiting peer review | All local tests pass |
| **Done** | Merged; acceptance criteria confirmed | PR merged; second member confirms ACs met |

**Backward transitions permitted:** Review → Testing if review identifies a defect; Testing → In Progress if test reveals a logic error.

### Sprint 3 Board Snapshot (current, 2026-05-17)

```
BACKLOG            IN PROGRESS          TESTING         REVIEW      DONE
───────────        ───────────          ───────         ──────      ────
T3-6 Retro         T3-1 Unit test       VC-5 Resp.time  —           ✅ All Sprint 1 items
T3-7 Est. meeting  T3-2 Unit test                                   ✅ All Sprint 2 items
T3-8 LightGBM      T3-3 Int. test       
T3-9 SearchCV      T3-4 CI/CD yaml
                   T3-5 Risk matrix
```

### Board Filtering

- By owner: `assignee:vc` / `assignee:lz`
- By type: `label:[DI]` `label:[NFR]` `label:[FR]` `label:[US]`
- By sprint: `sprint:1` … `sprint:4`
- By priority: `priority:must-have` / `priority:should-have`

---

## 9. Sprint Backlog & Daily Meetings

### Sprint 3 Backlog

| ID | Story / Task | Owner | SP | State | ACs Summary |
|----|-------------|-------|----|----|-------------|
| T3-1 | Unit test: `QuestionnaireToFeatures.transform()` | Vytautas | 2 | In Progress | Given synthetic input → shape (1,32); invalid input raises ValueError |
| T3-2 | Unit test: `RiskModel` prediction + threshold | Vytautas | 2 | In Progress | Output ∈ [0,1]; threshold ∈ [0.25, 0.50]; label is 0 or 1 |
| T3-3 | Integration test: questionnaire inputs → risk score | Laurynas | 3 | In Progress | Full pipeline runs; score ∈ [0, 1000]; tier assigned |
| T3-4 | GitHub Actions CI (lint + test phases) | Laurynas | 2 | In Progress | `pytest` phase passes; `flake8` phase passes on push to main |
| T3-5 | Risk management matrix | Both | 2 | In Progress | ≥ 8 risks; probability/impact scored; mitigation per risk |
| T3-6 | Sprint retrospective | Both | 1 | Planned | ≥ 3 problems named; concrete solutions proposed |
| T3-7 | Task estimation meeting — record minutes | Both | 1 | Planned | All backlog items estimated in story points; minutes recorded |
| T3-8 | LightGBM model comparison | Vytautas | 3 | Planned | ROC-AUC compared vs baseline; result documented |
| T3-9 | RandomizedSearchCV on best model | Laurynas | 2 | Planned | ≥ 50 iterations; best params logged |

**Sprint 3 Velocity Target:** 18 story points  
**Sprint 3 Capacity:** Vytautas ~15 h, Laurynas ~15 h (partial — TA-2 defence mid-sprint)

---

### Sprint Planning Format

**When:** First day of sprint (1 hour)  
**Agenda:**
1. Sprint goal statement (5 min)
2. Review prioritised backlog items (10 min)
3. Break stories into tasks; estimate story points (20 min)
4. Assign tasks and confirm capacity (10 min)
5. Identify blockers (10 min)
6. Confirm Definition of Done (5 min)

---

### Daily Standup Format

**When:** Each working day, 10 minutes  
**Three questions:** Yesterday / Today / Blockers

**Sample — Sprint 3, Day 3 (2026-05-19):**

| | Vytautas | Laurynas |
|--|---------|---------|
| **Yesterday** | Started T3-1 unit test; `transform()` test passing | Started T3-4 CI YAML; build phase working |
| **Today** | T3-2 threshold unit test | T3-3 integration test; T3-4 test phase |
| **Blockers** | None | CI needs test discovery config (`pytest.ini`) |

**Action:** Vytautas to create `pytest.ini` before standup tomorrow.

---

## 10. Requirement Quality Criteria (NASA)

> Reference: NASA Appendix C — *How to Write a Good Requirement*  
> https://www.nasa.gov/reference/appendix-c-how-to-write-a-good-requirement/

### Six Quality Properties

| Property | Definition | Applied Check |
|----------|-----------|--------------|
| **Necessary** | Required to meet a mission goal or regulatory obligation | Is there a stakeholder need or legal mandate? |
| **Unambiguous** | Only one interpretation possible | No vague terms — all quantities are explicit numbers |
| **Attainable** | Technically and financially achievable | Confirmed feasible during spike or via dataset analysis |
| **Complete** | All conditions covered, including edge cases | Error paths and boundary conditions in ACs |
| **Verifiable** | Provable by test, inspection, or analysis | Each AC specifies its verification method |
| **Consistent** | No conflict with other requirements | Cross-checked across full backlog |

### Anti-patterns Eliminated

| Bad (vague) | Replaced with (specific) |
|-------------|--------------------------|
| "The system should be fast" | "System returns risk score in ≤ 2 s for 95 % of requests" |
| "Accurate model" | "ROC-AUC ≥ 0.63 on held-out test set (achieved: 0.6272)" |
| "User-friendly interface" | "All 15 fields labelled in plain language; invalid input raises inline error without crash" |
| "Good feature selection" | "Boruta `max_iter=100`, `alpha=0.05`; 10–20 features selected; each questionnaire-collectible" |

### Quality Checklist Applied to VC-3 (Threshold Optimisation)

| Check | Result |
|-------|--------|
| No undefined terms | ✅ "Maximises F1-score" is operationally defined |
| Threshold range is numeric [0.25, 0.50] | ✅ |
| Balanced accuracy target ≥ 0.55 is measurable | ✅ Computed from confusion matrix (achieved 0.5844) |
| Does not conflict with VC-1 | ✅ Sequentially dependent, not contradictory |
| Achievable given 8.1 % imbalance | ✅ Confirmed in Sprint 1 spike |
| Each AC has a verification method | ✅ Unit tests and code review |

---

## Delivery Summary

| Deliverable | Required By | Status |
|-------------|-------------|--------|
| 3 MVPs with business justification and dates | TA-2 (05.21) | ✅ |
| 6-month Roadmap with 8 Epics (WBS) | TA-2 (05.21) | ✅ |
| 4 sprint plans (2 spike, 2 implementation) | TA-2 (05.21) | ✅ |
| 16 backlog items (8 per member, all 4 types) | TA-2 (05.21) | ✅ |
| 4 requirements detailed to DoR state | TA-2 (05.21) | ✅ |
| Spike 1 details: data prep + feature extraction | TA-2 (05.21) | ✅ |
| Spike 2 details: hyperparameter search plan | TA-2 (05.21) | ✅ |
| MoSCoW prioritisation + meeting minutes | TA-2 (05.21) | ✅ |
| Sprint board ≥ 5 states + transition rules | TA-2 (05.21) | ✅ |
| Sprint 3 backlog + daily standup log | TA-2 (05.21) | ✅ |
| NASA requirement quality criteria | TA-2 (05.21) | ✅ |
| CI/CD YAML (build + test phases) | TA-3 (05.28) | 🔄 Sprint 3 |
| Unit + integration tests (≥ 4) | TA-3 (05.28) | 🔄 Sprint 3 |
| Task estimation meeting minutes | TA-3 (05.28) | 🔄 Sprint 3 |
| Risk management matrix | TA-3 (05.28) | 🔄 Sprint 3 |
| Sprint retrospective | TA-3 (05.28) | 🔄 Sprint 3 |
| LightGBM model comparison | Post-TA-3 | 🗓 Sprint 4 |
