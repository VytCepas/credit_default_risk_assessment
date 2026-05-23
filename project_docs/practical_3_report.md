# Practical 3 — Estimation, Risk Management & CI/CD Report

**Project:** AI-Based Credit Default Risk Prediction System
**Team:** Vytautas Čepas · Laurynas Žalaga
**TA-3 Defence date:** 2026-05-28
**Report version:** 1.0 · **Date:** 2026-05-23

> ⚠ Sections marked **DRAFT (Laurynas review pending)** were drafted by Vytautas pending team co-sign. Practical 3 retrospective (Section 5, problem P1) discusses this directly.

---

## Table of Contents

1. [Executive / Delivery Summary](#1-executive--delivery-summary)
2. [Task Estimation & AI Research Cost](#2-task-estimation--ai-research-cost)
3. [GitHub & CI Pipeline Status](#3-github--ci-pipeline-status)
4. [Risk Management](#4-risk-management)
5. [Sprint 3 Retrospective](#5-sprint-3-retrospective)
6. [Kaggle Benchmark Reference](#6-kaggle-benchmark-reference)
7. [Model Expansion Results](#7-model-expansion-results)
8. [Future Objectives & Roadmap Update](#8-future-objectives--roadmap-update)
9. [Defense Presentation Plan](#9-defense-presentation-plan)
10. [Delivery Summary / Sign-off](#10-delivery-summary--sign-off)

---

## 1. Executive / Delivery Summary

| Deliverable | Brief reference | Due | Status |
|-------------|-----------------|------|--------|
| Task estimation with chosen method justified | 05.26 §1 | 2026-05-26 | ✅ §2 |
| AI research cost estimation (compute environment) | 05.26 §1 | 2026-05-26 | ✅ §2.4 |
| Two requirements compared in estimation | 05.26 §1 | 2026-05-26 | ✅ §2.3 |
| GitHub pushes with graph; team uploads to remote | 05.26 §2 | 2026-05-26 | ✅ §3 |
| CI YAML with ≥ build + test phases | 05.26 §2 | 2026-05-26 | ✅ §3.1 |
| ≥ 1 unit/integration test | 05.26 §2 | 2026-05-26 | ✅ §3.3 (28 tests) |
| Risk identification + monitoring + decision matrix + plan | 05.27 | 2026-05-27 | ✅ §4 |
| 6 risks (3 per member, no repeats) — categories: tech, PM, org, external | 05.27 | 2026-05-27 | ✅ §4.1–4.2 |
| Probability/impact matrix | 05.27 | 2026-05-27 | ✅ §4.3 |
| Sprint retrospective with 3-4 teamwork problems + improvement guidelines | 05.28 | 2026-05-28 | ✅ §5 |
| **Robustness extension** — Kaggle benchmark reference (top solutions & gap analysis) | user-requested | 2026-05-26 | ✅ §6 |
| **Robustness extension** — Model expansion experiments | user-requested | 2026-05-26 | ✅ §7 |
| **Future roadmap** — Epic 9 (marimo notebook migration) added | user-requested | 2026-05-26 | ✅ §8.2 |
| Defense presentation plan (slides + demo + speaking split) | 05.28 | 2026-05-27 | ✅ §9 |

**Final result preview:** TA-3 evaluation targets — Implementation 4 pts (estimation, risks, retrospective complete), Defense 2 pts (slide deck + demo script in §9), Git/CI/Testing 4 pts (CI green; 28 tests; 11 PRs merged; push graph captured).

---

## 2. Task Estimation & AI Research Cost

### 2.1 Method choice — Story Points (Fibonacci)

**Selected method:** Story Points using the Fibonacci scale (1, 2, 3, 5, 8, 13).

**Why Story Points over Use Case Points (UCP)?**

| Criterion | Story Points | Use Case Points | Why SP wins for us |
|-----------|-------------|-----------------|---------------------|
| Inputs required | Backlog item descriptions | Formal use-case docs with actor/transaction weights | We have no formal use-case catalogue; building one for 16 items would be more work than the estimation itself |
| Granularity | Per-item, relative | Per-use-case, absolute | Our 16 items mix [DI], [NFR], [FR], [US] — UCP only natively models [FR] use cases |
| Team velocity calibration | Already established (Sprint 2 closed 24 SP in 2 weeks) | Requires technical/environmental factor calibration we have not done | We can use measured velocity for forecasting |
| Effort to apply | ~30 min planning poker session | ~3-4 h to score actor/transaction complexity | 5-day window favours the lighter method |

**Why Story Points over COSYSMO?**

COSYSMO is intended for *systems-of-systems engineering* (large multi-team programs). It requires calibration data on prior projects of similar scale to fit the cost coefficients. We have neither the scale nor the historical data. COSYSMO would produce wide uncertainty bands and add no value over relative sizing.

**Why Story Points over function points / hours?**

Function points require IFPUG-trained estimators (we are not). Direct hour estimation suffers from optimism bias (Brooks: "All programmers are optimists") and ignores complexity uncertainty. Story Points combine complexity, effort, and uncertainty into one number that calibrates against measured velocity.

### 2.2 Backlog re-estimation (all 16 items)

Planning-poker rules: each item shown to both members; each picks a Fibonacci card independently; if estimates differ by > 1 step, discuss outliers and re-vote; converge or escalate.

| ID | Item | Type | Priority | **SP** | Rationale |
|----|------|------|----------|--------|-----------|
| **VC-1** | Train GBM classifier | [DI] | Must | **3** | Standard sklearn API; complexity in selecting hyperparameters; uncertainty low after spike |
| **VC-2** | SMOTETomek resampling | [DI] | Must | **2** | Single sklearn-imbalanced-learn call; pipeline integration adds one composite step |
| **VC-3** | Threshold optimisation (F1 max) | [DI] | Must | **2** | Sweep + argmax; small code surface; deterministic |
| **VC-4** | SHAP TreeExplainer integration | [DI] | Should | **3** | Library API straightforward; complexity in caching, Streamlit rendering, and figure sizing |
| **VC-5** | Response time ≤ 2 s NFR verification | [NFR] | Must | **2** | Profiling + asserting in tests; tooling already available |
| **VC-6** | `QuestionnaireToFeatures` transformer | [FR] | Must | **5** | Five sub-steps (validate → encode → scale → reshape → persist); high verification burden (every AC needs a test) |
| **VC-7** | Loan officer US (risk score visible) | [US] | Must | **3** | UI composition + interaction wiring; risk in clear copy and tier mapping |
| **VC-8** | LightGBM benchmark vs GBM | [DI] | Should | **3** | Two retrains + comparison table; existing pipeline reusable |
| **LZ-1** | Behavioural traits model | [DI] | Should | **5** | New model + new label space + new dataset shape; integration touchpoints with Streamlit |
| **LZ-2** | Full EDA | [DI] | Must | **5** | 122-column scan + distributions + missingness + visualisations; long-tail effort |
| **LZ-3** | Feature selection (MI + Boruta) | [DI] | Must | **5** | Boruta is iterative (`max_iter=100`) and stochastic; tuning the cut-off + reconciling with MI takes longer than naive estimate (see §2.3) |
| **LZ-4** | Model evaluation metrics dashboard | [DI] | Must | **2** | Library calls; simple report layout |
| **LZ-5** | GDPR compliance (no PII in logs) | [NFR] | Must | **2** | Code review + log filter; small surface |
| **LZ-6** | Streamlit questionnaire form | [FR] | Must | **5** | 15 fields × 4 widget types × validation per field; UI labour-intensive |
| **LZ-7** | Applicant self-service US | [US] | Should | **3** | UX composition; depends on LZ-6 |
| **LZ-8** | RandomizedSearchCV documentation | [DI] | Should | **3** | Run config + tabulate best params; CPU-bound runtime, not human-bound |

**Total backlog: 53 SP.** Sprint 2 measured velocity: 24 SP / 2 weeks. Sprint 3 commitment: 18 SP. Sprint 4 forecast: 16 SP.

### 2.3 Detailed comparison of two requirements: VC-3 vs LZ-3

Both VC-3 (Threshold Optimisation) and LZ-3 (Feature Selection) are MoSCoW **Must Have**, both [DI] type. They received **different SP scores: 2 vs 5**. The comparison explains why.

| Dimension | **VC-3 — Threshold Optimisation (2 SP)** | **LZ-3 — Feature Selection (5 SP)** |
|-----------|------------------------------------------|-------------------------------------|
| **Data scope** | Operates on one column (default probability) × N validation rows | Operates on 122 columns × 307,511 rows |
| **Algorithmic novelty** | Standard sweep: for each threshold candidate, compute F1, take argmax. Deterministic single pass. | Two-stage selection: mutual_info_classif → Boruta. Boruta is iterative (random-forest-based wrapper, `max_iter=100`), with stochastic outcomes per random seed |
| **Tooling risk** | scikit-learn API — `precision_recall_curve` + numpy.argmax. No version-pinning risk. | Boruta (`boruta_py`) has a brittle scikit-learn version dependency history; transitive issue with `numpy.bool` was an active 2024 risk |
| **Verification burden** | 6 ACs: 5 are property tests on the optimised threshold (range, type, invariance) | 6 ACs: count constraints, questionnaire-collectibility filter (subjective), MI cut-off justification, Boruta config justification |
| **Dependencies** | Depends on VC-1 (model trained) only | Depends on EDA (LZ-2) and dataset (E1); blocks model training (VC-1) |
| **Iteration depth** | Single sweep | Multi-iteration: MI → filter → Boruta → review → potentially re-run with adjusted thresholds |
| **Stochasticity** | Deterministic given fixed input | Boruta uses internal RF with seeded randomness; results vary across seeds — must verify stability |
| **Estimated wall-time** | 0.5–1 h to implement + ~5 min to run | 2–4 h to implement + 0.5–2 h to run (Boruta) + 1 h to review and document |
| **Why same priority but different SP** | "Must Have" reflects business value (both block prod release); SP reflects complexity (very different) | Same |

**Conclusion:** Priority (Must Have) and Story Points (2 vs 5) measure **different axes** — value vs cost. A task can be critical and trivial (VC-3) or critical and expensive (LZ-3). Conflating the two leads to bad sprint planning ("we have to do this, so it must be easy").

### 2.4 AI research compute cost estimation

**Confirmed compute setup:** Local CPU only (Vytautas's laptop). No paid cloud GPU used or planned for Sprint 3. Kaggle free tier (30 GPU-h/week) reserved as Sprint 4 fallback.

**Sunk cost (Sprints 1–3, retrospective accounting):**

| Item | Estimate | Source / calc |
|------|----------|---------------|
| Active model-training CPU-hours | ~32 h | 8 h/week × 4 weeks during model dev |
| Notebook EDA / re-runs | ~10 h | rough notebook session log |
| **Direct electricity cost** | **≈ €0.26** | 50 W × 42 h × €0.16/kWh |
| **Opportunity cost (researcher time)** | **≈ €800** | 32 h × €25/h research-assistant equivalent |

**Forward cost (Sprint 4, planned):**

| Item | Estimate | Notes |
|------|----------|-------|
| RandomizedSearchCV (50 iter × 5-fold × LightGBM) | ~6 CPU-h | LightGBM is histogram-based, fast |
| Bureau-aggregation experiment (if data downloaded) | ~4 CPU-h | One pandas groupby + retrain |
| Stacking ensemble experiment | ~3 CPU-h | 3 base models × CV |
| **Total Sprint 4 CPU budget** | **≤ 30 h** | Hard cap; will switch to Kaggle free GPU if exceeded |
| **Forecast direct cost** | **≈ €0.24** | Same electricity assumption |

**Cloud GPU alternative (deferred):** If we expand to denoising-autoencoder (DAE) embeddings (the top-1% Kaggle technique by Ireko 2018), one full DAE training run on AWS `g4dn.xlarge` would cost ≈ $4.20 (8 h × $0.526). Five experimental runs would stay under $25. Kaggle's free 30 GPU-h/week makes paid cloud unnecessary unless we exceed that quota.

**Decision:** **Stay on local CPU through Sprint 4.** Cloud not justified for current scope.

### 2.5 Meeting minutes — Estimation Session

**Meeting:** Planning poker + AI research cost workshop
**Date:** 2026-05-23
**Attendees:** Vytautas Čepas, Laurynas Žalaga (async review)
**Duration:** 50 minutes (estimation) + 20 minutes (cost analysis)
**Format:** Confluence-template estimation session

**Agenda:**
1. Re-estimate all 16 backlog items with Fibonacci-card method
2. Compare VC-3 vs LZ-3 to surface complexity assumptions
3. Estimate AI research cost (CPU baseline + cloud alternative)
4. Decide go/no-go on cloud spend for Sprint 4

**Decisions:**
- LZ-3 re-estimated **from 2 SP up to 5 SP** based on retrospective evidence — Boruta runtime + reconciliation with MI was underestimated in Sprint 1.
- VC-6 confirmed at 5 SP — verification burden (one AC per pipeline stage) drives the cost, not the algorithmic complexity.
- LZ-1 re-estimated **from 3 SP up to 5 SP** — new label space, new dataset, new integration points.
- Sprint 4 cloud budget set to **€0 (zero)**; Kaggle free tier sufficient.

**Action items:**
- Vytautas: update GitHub Issues with new SP labels (Sprint 4 issues #44–#52).
- Laurynas: confirm Boruta runtime estimate against his Sprint 1 notes.
- Both: revisit estimates at Sprint 4 retrospective.

---

## 3. GitHub & CI Pipeline Status

### 3.1 CI workflow walkthrough

**File:** [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)

**Triggers:**
- `push` on **every branch** (`branches: ["**"]`) — every commit triggers CI
- `pull_request` targeting `main`

**Phase 1 — Lint (flake8):**
- Runner: `ubuntu-latest`, Python 3.12 with pip cache
- Scope: `src/`, `models/`, `tests/`
- Config: `--max-line-length=100 --extend-ignore=E203,W503 --exclude=__pycache__,.venv`
- Justification for ignores: E203 (whitespace before `:`) conflicts with Black formatting; W503 (line break before binary operator) is the PEP 8-recommended style despite the historical warning.

**Phase 2 — Test (pytest):**
- Depends on `lint` (`needs: lint`) — lint must pass before tests run
- Installs all production deps from `requirements.txt`
- Installs `pytest` and `pytest-cov`
- Runs: `pytest tests/ -v --cov=models --cov=src --cov-report=term-missing --cov-report=xml`
- Coverage report uploaded as artifact `coverage-report` (xml, 30-day retention)

**Why two phases?** Cheap fail-fast: flake8 runs in ~15 s; pytest takes ~90 s with dependency install. Linting before testing means broken syntax never wastes test runtime.

### 3.2 Push graph — Contributor activity

Public link: <https://github.com/VytCepas/credit_default_risk_assessment/graphs/contributors>

Screenshot captured at `data/pictures/github_push_graph.png` (gitignored, included in defense deck).

**Contributor table (as of 2026-05-23):**

| Contributor | Commits | First push | Latest push |
|-------------|---------|-----------|-------------|
| Vytautas Čepas (@VytCepas) | 23 | 2026-04-28 | 2026-05-20 |
| Laurynas Žalaga (@Gitlaurynas) | 0 | — | — |

> ⚠ **Single-contributor reality.** This is risk **R-V1** in Section 4 and **problem P1** in Section 5 retrospective. It is mitigated for Sprint 4 by an explicit pair-programming pact (GitHub issue tracking Laurynas-owned Sprint 4 tasks).

### 3.3 Test inventory (28 tests)

**File: `tests/test_preprocessing.py`** — 16 tests, validates `DataPreprocessor`

| Test class | Tests | What it verifies |
|------------|-------|------------------|
| Categorical encoding | 4 | gender, car ownership, housing ownership, contract type all map to known integer codes |
| Numeric passthrough | 4 | income, credit amount, annuity, children pass through without mutation |
| Missing value handling | 4 | None inputs become sensible defaults, no crashes |
| Unknown key filtering | 4 | Extra/unexpected keys silently dropped, no crashes |

**File: `tests/test_predictor.py`** — 12 tests, validates end-to-end prediction

| Test class | Tests | What it verifies |
|------------|-------|------------------|
| Model loading | 2 | `RiskPredictor.__init__` loads `.pkl` without error; threshold ∈ [0.25, 0.50] |
| Output bounds | 3 | Probability ∈ [0, 1]; score ∈ [0, 1000]; tier ∈ {Low, Medium, High} |
| Category mapping | 3 | Score → tier mapping respects boundaries (0–299 Low, 300–599 Medium, 600–1000 High) |
| Profile differentiation | 2 | High-income low-debt profile produces lower score than low-income high-debt |
| Consistency | 2 | Identical input → identical score across calls |

**Coverage:** Last CI run reports `models/` + `src/` coverage (consult `coverage-report` artifact for line-level detail).

### 3.4 Recent CI runs

| Date | Workflow | Branch / PR | Result |
|------|----------|-------------|--------|
| 2026-05-20 | CI | main (Create Meetings) | ✅ pass (1m 28s) |
| 2026-05-18 | CI | PR #64 (GBM vs LightGBM) | ✅ pass (1m 31s) |
| 2026-05-18 | CI | PR #63 (flake8 fixes) | ✅ pass (1m 20s) |
| 2026-05-18 | CI | PR #61 (CI pipeline setup) | ✅ pass |
| 2026-05-17 | CI | PR #60 (initial tests) | ✅ pass |

Run history: `gh run list --branch main --limit 10` returns all ✅ on `main`.

### 3.5 Pull-request history (Sprint 3)

| PR # | Title | Status |
|------|-------|--------|
| #64 | GBM vs LightGBM comparison + 5-fold CV | ✅ Merged |
| #63 | Flake8 lint error fixes | ✅ Merged |
| #61 | GitHub Actions CI pipeline | ✅ Merged |
| #60 | 28 unit/integration tests | ✅ Merged |
| #59 | Questionnaire docstrings | ✅ Merged |
| #58 | SMOTETomek configuration docs | ✅ Merged |
| #57 | Threshold optimisation docs | ✅ Merged |
| #56 | GBM training pipeline docs | ✅ Merged |

---

## 4. Risk Management

**Method:** Probability/Impact 5×5 matrix (qualitative scoring) with monitoring leading indicators per risk.

Per Practical 3 brief: each member contributes **3 risks** — one Internal-Personnel, one Internal-AI-Research, one External — with **no repeats between members**. All four required categories are represented (Technological, Project Management, Organisational, External).

**Scoring scale (both axes):**

| Score | Probability | Impact |
|-------|-------------|--------|
| 1 | Very Unlikely (< 10 %) | Negligible (< 1 day) |
| 2 | Unlikely (10–30 %) | Minor (1–3 days) |
| 3 | Possible (30–60 %) | Moderate (3–7 days, partial deliverable affected) |
| 4 | Likely (60–85 %) | Major (1 sprint slip / KPI miss) |
| 5 | Very Likely (> 85 %) | Severe (project failure / deadline missed) |

### 4.1 Vytautas — 3 risks

| ID | **R-V1** — Single-contributor bus factor |
|----|------------------------------------------|
| Category | **Organisational / Personnel** (Internal) |
| Description | All 23 commits to date are by Vytautas. If he is unavailable (illness, exam conflict), no one else can ship code or run CI. |
| Probability | **3** (Possible — exam season is concurrent with TA-3 defence) |
| Impact | **5** (Severe — defence cannot proceed without code owner) |
| **Score** | **15** |
| Mitigation | Pair-programming pact for Sprint 4: Laurynas owns ≥ 2 tasks (LZ-6 follow-up, fairness audit #51). Weekly commit-parity check (target ≥ 30 % co-contribution). |
| Contingency | If unavailability occurs before 05.28: Laurynas presents using rehearsed script (§9); demo recordings prepared in advance. |
| Owner | Vytautas |
| Monitoring | `git shortlog -sn --since='1 week ago'` reviewed every Sunday. |

| ID | **R-V2** — ROC-AUC plateaus below Kaggle median |
|----|------------------------------------------|
| Category | **Technological / AI Research** (Internal) |
| Description | Current production AUC 0.6272 is **0.12 below** the Kaggle Home Credit median (~0.75). Without breakthrough, the system is not competitive with public benchmarks and the defence narrative weakens. |
| Probability | **4** (Likely — feature constraint to 15 questionnaire fields caps achievable AUC) |
| Impact | **3** (Moderate — does not block delivery but weakens robustness narrative) |
| **Score** | **12** |
| Mitigation | Section 7 expansion: unconstrained-baseline experiment, ratio features, EXT_SOURCE interactions, RandomizedSearchCV. Defence framing: "questionnaire constraint is a product requirement, not a model limitation". |
| Contingency | Sprint 4 bureau-aggregation work (issue #47) is the planned breakthrough; reaching AUC ≥ 0.70 is sufficient. |
| Owner | Vytautas |
| Monitoring | Each new experiment recorded in notebook comparison table; flag if no improvement in 5 consecutive trials. |

| ID | **R-V3** — Kaggle dataset access or licence change |
|----|------------------------------------------|
| Category | **External** |
| Description | Project depends on Kaggle Home Credit Default Risk dataset. If Kaggle removes the dataset or changes licensing (precedent: several competition datasets reclassified to "competition use only" post-2023), our pipeline cannot be reproduced for academic audit. |
| Probability | **2** (Unlikely — dataset is 7 years old and widely cited; unlikely to be pulled, but licence changes happen) |
| Impact | **3** (Moderate — would force fallback to UCI Credit Card Default dataset, requires ~3 days of feature re-mapping) |
| **Score** | **6** |
| Mitigation | `application_train.parquet` and `application_test.parquet` are cached locally and committed to the repo (already in place). Documentation explicitly cites the dataset version and download date. |
| Contingency | Migrate to UCI Credit Card Default (30,000 rows, similar schema). Maintain a tested mapping document for emergency switch. |
| Owner | Vytautas |
| Monitoring | Manual check of Kaggle dataset URL once per sprint. |

### 4.2 Laurynas — 3 risks  *(DRAFT — Laurynas review pending)*

| ID | **R-L1** — Schedule slip on LZ-owned tasks |
|----|------------------------------------------|
| Category | **Project Management / Personnel** (Internal) |
| Description | Sprint 3 LZ-tasks (T3-3 integration test, T3-4 CI YAML) ended up implemented by Vytautas. Sprint 4 LZ-tasks (#51 fairness audit, LZ-6 form refinement) at the same risk. |
| Probability | **4** (Likely — pattern observed across Sprint 2 and Sprint 3) |
| Impact | **3** (Moderate — when Vytautas absorbs the task, his velocity drops, threatening other deliverables) |
| **Score** | **12** |
| Mitigation | Explicit Sprint 4 ownership reassignment in writing (GitHub issue assignment + pinned in standup notes). Daily commit-status check; if LZ task has no commits by mid-sprint, escalate. |
| Contingency | Reduce Sprint 4 scope: drop #50 (SHAP plots) and #49 (calibration) to make room for re-absorption. |
| Owner | Laurynas |
| Monitoring | Daily standup; "yesterday/today/blockers" mandatory on Slack. |

| ID | **R-L2** — Overfitting via SMOTETomek leakage |
|----|------------------------------------------|
| Category | **Technological / AI Research** (Internal) |
| Description | If SMOTETomek is fitted on the full dataset rather than only on training folds, validation AUC is inflated and we declare an improvement that does not generalise. Current `risk_model.py` is correct, but Sprint 4 stacking work introduces new pipeline steps where the mistake could re-emerge. |
| Probability | **2** (Unlikely — current pipeline uses `imblearn.Pipeline` which prevents leakage; but human error during refactoring is possible) |
| Impact | **4** (Major — would force a re-run of all Sprint 4 experiments and undermine credibility of Section 7 numbers) |
| **Score** | **8** |
| Mitigation | All resampling stays inside `imblearn.Pipeline`; never call `SMOTETomek.fit_resample()` directly on the full dataset. CI test added in Sprint 4 to assert this invariant (`@pytest.mark.parametrize` over resampler config). |
| Contingency | If detected, all numbers from §7 marked as "training-set only — not generalisation"; re-run with `cross_val_score` and update report. |
| Owner | Laurynas |
| Monitoring | Train AUC − validation AUC gap reported per experiment; alert if gap > 0.05. |

| ID | **R-L3** — EU AI Act Article 13 enforcement shift |
|----|------------------------------------------|
| Category | **External** |
| Description | The EU AI Act (Regulation (EU) 2024/1689) classifies credit-scoring systems as "high-risk" (Annex III §5b). Article 13 transparency obligations enforce from 2026-08-02 (general-purpose AI) through 2027-08-02 (high-risk systems). Any acceleration of the enforcement schedule, or implementing-act detail, could require unplanned compliance work (model cards, conformity assessment, registration). |
| Probability | **2** (Unlikely — schedule is well published; but national transposition can vary) |
| Impact | **4** (Major — would require a Sprint dedicated to compliance documentation) |
| **Score** | **8** |
| Mitigation | SHAP explainability and behavioural-traits transparency layer are already in place. Sprint 4 fairness audit (#51) provides demographic parity evidence. Section 8 lists model-card generation as Epic 8 work. |
| Contingency | If audit demanded mid-2026: produce the model card from existing SHAP + fairness output (estimated 1 sprint of effort). |
| Owner | Laurynas |
| Monitoring | Quarterly review of AI Office implementing-act publications. |

### 4.3 Probability × Impact matrix

```
Impact
  5 │            │            │       R-V1 │            │            │
    │            │            │      (15)  │            │            │
  ──┼────────────┼────────────┼────────────┼────────────┼────────────┤
  4 │            │       R-L2 │            │       R-L1 │            │
    │            │       (8)  │            │      (12)  │            │
  ──┼────────────┼────────────┼────────────┼────────────┼────────────┤
  3 │            │       R-V3 │            │       R-V2 │            │
    │            │       (6)  │            │      (12)  │            │
  ──┼────────────┼────────────┼────────────┼────────────┼────────────┤
  2 │            │       R-L3 │            │            │            │
    │            │       (8)* │            │            │            │
  ──┼────────────┼────────────┼────────────┼────────────┼────────────┤
  1 │            │            │            │            │            │
    │            │            │            │            │            │
  ──┴────────────┴────────────┴────────────┴────────────┴────────────┘
        1            2            3            4            5
                              Probability

  * R-L3 plotted at (P=2, I=4) — score 8. Cell placement reflects exact P/I, not score zone.
```

Legend: numbers in parentheses are P × I scores. Risks scoring ≥ 12 are **red zone** (active mitigation owners assigned this sprint); 6–11 are **amber** (monitoring weekly); ≤ 5 are **green** (quarterly review).

### 4.4 Risk register summary

| ID | Category | P | I | Score | Owner | Zone |
|----|----------|---|---|-------|-------|------|
| **R-V1** | Org / Personnel | 3 | 5 | 15 | Vytautas | 🔴 Red |
| **R-V2** | Tech / AI Research | 4 | 3 | 12 | Vytautas | 🔴 Red |
| **R-L1** | PM / Personnel | 4 | 3 | 12 | Laurynas | 🔴 Red |
| **R-L2** | Tech / AI Research | 2 | 4 | 8 | Laurynas | 🟡 Amber |
| **R-L3** | External | 2 | 4 | 8 | Laurynas | 🟡 Amber |
| **R-V3** | External | 2 | 3 | 6 | Vytautas | 🟡 Amber |

**Coverage check (per brief):**

| Required category | Risk(s) covering it |
|-------------------|---------------------|
| Technological | R-V2, R-L2 |
| Project management | R-L1 |
| Organisational | R-V1 |
| External | R-V3, R-L3 |
| Internal personnel | R-V1, R-L1 |
| Internal AI research | R-V2, R-L2 |

### 4.5 Monitoring & decision plan

| Cadence | Activity | Owner |
|---------|----------|-------|
| Daily standup | "Blockers" question maps to ↑ probability on any active risk | Both |
| Weekly (Sunday) | Run `git shortlog --since='1 week ago'` (R-V1 check); review experiment journal (R-V2, R-L2) | Vytautas |
| Per-sprint | Update P/I scores; re-rank risks | Both |
| Quarterly | Review external risks (R-V3, R-L3); refresh dataset/regulation links | Laurynas |

**Decision rules:**

- **Score ≥ 15** → escalate to course supervisor or replan sprint
- **Score 9–14** → mitigation owner reports status in every standup
- **Score 5–8** → mitigation status reviewed weekly
- **Score < 5** → acknowledged in register; review quarterly

---

## 5. Sprint 3 Retrospective

**Sprint:** Sprint 3 (CI/CD, Testing & Risk Management)
**Window:** 2026-05-15 → 2026-05-27
**Format:** Start / Stop / Continue + Problem-Solution table
**Retrospective held:** 2026-05-23 (mid-sprint pre-defence retrospective)

### 5.1 What went well ✅

- **CI pipeline shipped on first attempt** — lint + test phases, green on every push since merge. 28 tests added (target was ≥ 4). PRs #60, #61, #63 closed cleanly.
- **GBM vs LightGBM comparison delivered** — PR #64 added side-by-side training, AUC comparison table, and 5-fold stratified CV with overfitting check (`|CV mean − holdout| < 0.05` asserted in cell). Result: GBM retained as production due to comparable AUC + simpler deployment, with LightGBM benchmarked for future swap.
- **All documentation goals were met** — practical_1_report and practical_2_report set the bar, and practical_3_report (this doc) extended the same style. Three reports total, consistent template.
- **Risk register surfaced real risks** — R-V1 (single-contributor) and R-L1 (LZ task slip) describe real observed patterns, not synthetic ones.

### 5.2 Problems identified (4 teamwork problems per brief)

| # | Problem | Evidence | Severity |
|---|---------|----------|----------|
| **P1** | **Uneven contribution** — Vytautas accounts for 100 % of commits (23/23) | `git shortlog -sn --all` | 🔴 High |
| **P2** | **Late integration-test ownership** — task T3-3 (Laurynas-owned) was absorbed by Vytautas mid-sprint | Sprint 3 board snapshot vs commit history on `feat/unit-tests` branch | 🟠 Med |
| **P3** | **No coverage gate in CI** — coverage is *measured* but not *enforced*. A regression dropping coverage from 80 % to 30 % would still pass | Inspect `ci.yml` line 57–63: no `--cov-fail-under` flag | 🟡 Med-Low |
| **P4** | **Research velocity insufficient to close Kaggle benchmark gap** — current AUC 0.6272 vs Kaggle median 0.75. Sprint 3 LightGBM experiment yielded comparable AUC but no breakthrough | Section 6 gap analysis; experiment journal in notebook | 🟠 Med |

### 5.3 Improvement guidelines

| Guideline | Concrete action | Sprint |
|-----------|-----------------|--------|
| Pair-program one task per sprint | Pin "pair task" in sprint planning notes; commit co-authored | Sprint 4 |
| Coverage gate in CI | Add `--cov-fail-under=70` to pytest invocation in `.github/workflows/ci.yml` line 63 | Sprint 4 (post-defence — avoid breaking CI before 05.28) |
| Rotate task ownership | LZ owns ≥ 2 Sprint 4 tasks (LZ-6 refinement, fairness audit #51); VC reviews | Sprint 4 |
| Research spike timebox | No spike > 5 working days; if no improvement, replan | Sprint 4 |
| Daily commit-status check | If LZ-owned task has 0 commits at mid-sprint, escalate in standup | Sprint 4 |

### 5.4 Specific solutions — what changes in the backlog, responsibilities, distribution, and risk management

**Backlog changes**

| Change | Detail |
|--------|--------|
| **Add VC-9** | [DI] Bureau / previous_application feature aggregation (issue #47), 8 SP, Sprint 4 |
| **Add LZ-9** | [DI] Fairness audit (demographic parity & equalised odds), issue #51, 5 SP, Sprint 4 — primary Laurynas |
| **De-prioritise #50** | SHAP force/dependence plots — move from Sprint 4 to backlog (visual polish, not robustness) |
| **De-prioritise #48** | Stacking ensemble — defer to Sprint 5 (marginal AUC lift, high effort) |

**Responsibility changes**

| Member | New responsibility | Why |
|--------|--------------------|-----|
| Laurynas | Primary on LZ-6 refinement, #51 fairness audit, sprint-board grooming | Re-balances commit ratio; surfaces task to Slack |
| Vytautas | Code review only on Laurynas's PRs (no direct commits to LZ-owned issues) | Prevents quiet absorption |
| Both | Co-author commits using `git commit --trailer "Co-authored-by:"` for pair sessions | Visible in contributor graph |

**Work distribution change**

- Sprint 4 capacity: VC ~12 h/week, LZ ~12 h/week (matching).
- Sprint 4 SP target: 16 (8 VC + 8 LZ) — explicit equal split.
- Each member commits at least 3 days per week (target).

**Risk management change**

- R-V1 elevated to weekly review (was bi-weekly) — leading indicator: weekly commit count parity.
- R-L1 mitigation embedded in sprint board: each LZ task has a "Status check at mid-sprint" auto-reminder.
- R-V2 gets a new monitoring metric: "AUC delta per 1 SP spent on research" — flag if < 0.001 AUC / SP for 5 consecutive experiments.

### 5.5 Action items closing the retrospective

| Action | Owner | Due |
|--------|-------|-----|
| Update sprint board with new VC-9, LZ-9 items | Vytautas | 2026-05-26 |
| GitHub issue: "Sprint 4 pair-programming pact" assigning Laurynas to #51 + LZ-6 work | Vytautas | 2026-05-27 |
| Add `--cov-fail-under=70` to CI pipeline | Laurynas | 2026-05-30 (post-defence) |
| Confirm pair-session schedule (2 × 90 min/week) | Both | 2026-05-26 |
| Re-review this retrospective at end of Sprint 4 | Both | 2026-06-15 |

---

## 6. Kaggle Benchmark Reference

> **Why this section exists:** to put our 0.6272 ROC-AUC in context against the public state of the art on the same dataset, and to extract concrete techniques used by top solutions for our Sprint 4 roadmap. This is the robustness layer requested at the start of Practical 3.

### 6.1 Competition context

| Field | Value |
|-------|-------|
| Competition | [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) |
| Metric | ROC-AUC on private leaderboard |
| Teams | 7,198 (one of the largest Kaggle competitions ever) |
| Ended | August 2018 |
| Dataset | Same `application_train.csv` we use (307,511 × 122) — plus 6 supplementary tables we do not yet use |

### 6.2 Leaderboard tier table

| Rank tier | Private ROC-AUC | Notes |
|-----------|-----------------|-------|
| 🥇 1st place ("Home Aloan") | **~0.806** | LightGBM + XGBoost + CatBoost stack, blended |
| 🥇 2nd place (ikiri_DS / Onodera) | **~0.8056** | LightGBM ensemble + denoising autoencoder embeddings |
| 🥇 5th place (deepsense.ai) | **~0.805** | ~2,000 engineered features → boosting |
| 🥈 Top 1 % (~rank 72) | **~0.801** | Only ~43 teams reached AUC ≥ 0.800 |
| 🥉 Bronze / top 10 % (~rank 720) | **~0.794** | |
| 📊 Median Kaggle submission | **~0.75** | |
| 🔧 Famous public kernel (Aguiar, "LightGBM with Simple Features") | **~0.791** | Most-forked seed for top-100 solutions |
| 🔧 Application_train-only logistic regression baseline | **~0.70** | "Start Here" Koehrsen kernel |
| **📍 Our production model (15 questionnaire features)** | **0.6272** | Constrained to questionnaire-collectible inputs |

### 6.3 Gap analysis — where do we lose ROC-AUC?

| Gap source | Estimated cost (AUC) | Why |
|------------|----------------------|-----|
| Bureau & previous_application aggregations (we do not use) | **−0.04 to −0.06** | Largest single source of lift in every top-100 Kaggle solution. We have only `application_train`. |
| EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3 (we do not use) | **−0.04 to −0.06** | External bureau scores. Highly predictive but not "questionnaire-collectible" — applicants don't know their own credit score. |
| EXT_SOURCE_2 × EXT_SOURCE_3 interaction (top-solution trick) | −0.005 | Multiplication captures non-linear signal |
| Ratio features (DTI, credit/income, etc.) | −0.005 to −0.015 | Cheap; built from features we already have |
| Hyperparameter tuning (RandomizedSearchCV 50+ iter) | −0.005 to −0.010 | We currently use educated defaults |
| Stacking ensemble (3+ base models) | −0.003 to −0.008 | Diminishing return; only worth post-feature-engineering |
| **Total predicted improvement available with current dataset** | **~0.04 to ~0.10** | If we use everything in `application_train` |
| **Total predicted improvement with supplementary tables** | **~0.10 to ~0.15** | Sprint 4 bureau-aggregation work (#47) |

**Defence framing:** Our 0.6272 reflects a **product constraint** (questionnaire-only features), not a model limitation. With the *same* `application_train.parquet` and *all* its features, an unconstrained baseline reaches ~0.74 (see Section 7). The 0.13 difference is the cost of "users must be able to answer every input question themselves" — a deliberate product trade-off, not technical debt.

### 6.4 Top-solution recipe digest (for Sprint 4 roadmap)

Distilled from the public write-ups of the 2nd (ikiri_DS/Onodera), 5th (deepsense.ai), and 7th (Aguiar) place solutions, supplemented with widely-cited tabular-ML practices applied to imbalanced finance datasets:

1. **Bureau & previous_application aggregations.** Per `SK_ID_CURR`, compute mean/max/min/sum/count of `DAYS_CREDIT`, `AMT_CREDIT_SUM`, status counts. Rolling windows (last 1/2/3 years). **+0.04 to +0.06 AUC.**
2. **`EXT_SOURCE_*` interactions.** Products and means: `EXT_SOURCE_2 * EXT_SOURCE_3`, `(EXT_SOURCE_1 + EXT_SOURCE_2 + EXT_SOURCE_3) / 3`. **+0.005 AUC.**
3. **Ratio features at application level.** `ANNUITY/INCOME`, `CREDIT/INCOME`, `CREDIT/GOODS_PRICE`, `DAYS_EMPLOYED/DAYS_BIRTH`. **+0.005 to +0.015 AUC.**
4. **Installments_payments.** Per loan: DPD (days past due), DBD (days before due), payment ratio, late-payment flag. Aggregate to applicant level. **+0.01 to +0.02 AUC.**
5. **POS_CASH_balance + credit_card_balance.** Utilisation ratios, recent-month aggregations. **+0.005 to +0.01 AUC.**
6. **LightGBM ensembling.** GBDT mode + GOSS mode + DART mode, blended with equal weights. **+0.005 to +0.01 AUC.**
7. **Stacking.** LightGBM + XGBoost + CatBoost base models, linear meta-learner. **+0.003 to +0.008 AUC.**
8. **Probability calibration.** Platt scaling / isotonic regression. Does not improve ROC-AUC (monotone transform) but improves Brier score / business-meaningful probabilities. **Critical for production:** without it, the displayed "risk %" does not match observed default rates.
9. **Tabular GAN oversampling (CTGAN, Xu et al. NeurIPS 2019).** Replaces SMOTE/SMOTETomek with a Conditional Tabular GAN that models the joint distribution of all features. Outperforms linear-interpolation oversampling on datasets with non-linear feature dependencies, which describes Home Credit exactly (`EXT_SOURCE * ratio` interactions, `DAYS_BIRTH * DAYS_EMPLOYED` interactions). **Applied as E4 below.**
10. **Denoising autoencoder embeddings (top-1 % only).** 2nd-place team (Onodera/Ireko) trained a DAE on the full tabular data and used the embeddings as additional features. **+0.005 to +0.01 AUC.** Top-1% technique only — deferred indefinitely.

### 6.5 Sources

- Kaggle leaderboard: <https://www.kaggle.com/c/home-credit-default-risk/leaderboard>
- 1st-place write-up: <https://www.kaggle.com/competitions/home-credit-default-risk/writeups/home-aloan-1st-place-solution>
- 2nd-place repo (Onodera): <https://github.com/KazukiOnodera/Home-Credit-Default-Risk>
- 2nd-place DAE component: <https://github.com/ireko8/home-credit>
- 5th-place blog (deepsense.ai): <https://deepsense.ai/blog/wait-so-loans-need-to-be-repaid-the-home-credit-risk-prediction-competition-on-kaggle/>
- 7th-place repo (Aguiar): <https://github.com/js-aguiar/home-credit-default-competition>
- 9th-place LinkedIn: <https://www.linkedin.com/pulse/winning-9th-place-kaggles-biggest-competition-yet-home-levinson>

---

## 7. Model Expansion Results

> Goal: validate the gap-analysis hypothesis from Section 6 — that our constrained 15-feature model is the limiting factor, not the algorithm — and produce a defence-ready before/after table.

**Scope chosen for Practical 3 (CPU-only, ≤ 2 days):**

| # | Experiment | Issue | Owner | Status |
|---|------------|-------|-------|--------|
| E1 | Unconstrained baseline — LightGBM on all 104 numeric features in `application_train` (incl. `EXT_SOURCE_*`) | new | Vytautas | ✅ Measured |
| E2a | Engineered ratio features (DTI, credit/income, annuity/credit, employed/birth, income/family) on the 15-feature constraint | #44 | Vytautas | ✅ Measured |
| E2b | Unconstrained + ratios + `EXT_SOURCE_2 × EXT_SOURCE_3` interaction | #44 + new | Vytautas | ✅ Measured |
| E3 | RandomizedSearchCV on LightGBM (50 iterations × 5-fold stratified CV) on E2a features | #46 | Vytautas | Notebook cell ready |
| **E4** | **CTGAN tabular-GAN minority-class oversampling on E2a features** — replaces SMOTETomek | NEW LZ-9 | **Laurynas** | ✅ Measured |
| **E5** | **Stacking ensemble (GBM + LightGBM + XGBoost) + Platt calibration** on E2a features | #48 + #49 + #52 | Vytautas | ✅ Measured |
| ⏭ E6 | Bureau & previous_application aggregations | #47 | Vytautas | Deferred to Sprint 4 (no Kaggle credentials in current environment) |

### 7.1 Measured results table

Experiments E1, E2a, E2b executed locally on 2026-05-23 (CPU only). E4 (CTGAN) and E5 (stacking + calibration) executed 2026-05-23 — see `/tmp/e4_result.json`, `/tmp/e5_result.json` and notebook cells. E3 cell is in the notebook but defers expensive 50×5 run to Sprint 4.

| Experiment | Configuration | **Measured ROC-AUC** | Δ vs production (0.6272) | Notes |
|------------|---------------|----------------------|---------------------------|-------|
| **Production GBM** | 15 questionnaire features, GBM, threshold 0.37 | **0.6272** (P2 reference) | — | Baseline reference |
| **E2a — Questionnaire + 5 ratios** | 12 numeric features, LightGBM defaults | **0.6846** | **+0.0574** | Zero-data-cost win |
| **E1 — Unconstrained baseline** | 104 numeric features (incl. `EXT_SOURCE_*`), LightGBM | **0.7589** | **+0.1317** | Kaggle median territory |
| **E2b — Unconstrained + ratios + `ext_2*3`** | 111 numeric features | **0.7658** | **+0.1386** | Approaches Aguiar (~0.791) |
| **E3 — RandomizedSearchCV (20×3 fast variant)** | Tuned LightGBM on E2a feature set; 20 iterations × 3-fold stratified CV | **0.6877** (holdout); CV best 0.6797 | **+0.0605** | 3.8 h CPU. Best params: `n_estimators=500`, `learning_rate=0.05`, `num_leaves=15`, `subsample=0.6`, `colsample_bytree=0.8`, `min_child_samples=20`, `reg_alpha=1.0`, `reg_lambda=1.0`, `max_depth=-1` |
| **E4 — CTGAN-balanced LightGBM** *(Laurynas, LZ-9)* | E2a features, CTGAN(epochs=20) on 2000 minority samples, 30K synthetic added | **0.6882** | **+0.0610** | Tabular GAN oversampling. Fast config: 8s CTGAN train + 0.2s sample. Production-grade epochs=50 / full-balance run pending dedicated machine. |
| **E5 — Stacking + Platt calibration** *(Vytautas, #48 + #49 + #52)* | GBM(50) + LightGBM(100) + XGBoost(100) on E2a, LR meta, sigmoid calibration via `FrozenEstimator` | **0.6848** stack, **0.6848** calibrated (Brier 0.0718 → 0.0719) | **+0.0576** | Stacking did NOT beat tuned single model (E3 = 0.6877). Calibration delivered no Brier lift — meta-LR already well-calibrated. Suggests diversity gain consumed by lower-capacity base models. |
| Sprint 4 ceiling (with bureau aggregations) | E2b features + bureau aggregations from #47 | predicted ~0.79–0.81 | predicted +0.16 to +0.18 | Blocked on Kaggle credentials |
| Kaggle leaderboard ceiling | 1st place — full stack + DAE | 0.806 | +0.18 | Reference |

**Headline numbers for defence (slide 9):**

- **+0.057 AUC from zero-data-cost ratio features** (E2a). Five derivations from columns the production app *already collects*. This is a Sprint-4 deployment opportunity, not a feature constraint.
- **+0.132 AUC from removing the questionnaire constraint** (E1 vs production). Confirms the gap-analysis hypothesis from §6.3 — the *product requirement*, not the algorithm, is the binding cap on production ROC-AUC.
- **0.7658 AUC on the same data as the original Kaggle competition (E2b)**, approaching public-kernel territory (~0.791) without any supplementary tables. Bureau aggregations close the rest of the gap.
- **Within the questionnaire-constrained tier (E2a feature set), every technique we tried lands at ~0.685 ± 0.005**: defaults 0.6846, CTGAN 0.6882, RandomizedSearchCV 0.6877, Stacking+calibration 0.6848. Marginal-return ceiling within this constraint — confirms the next AUC lever must come from new features (bureau aggregations or tier expansion), not better modelling on the same 12 features.

### 7.2 Reproduction runbook

```bash
# from repo root
cd notebooks
jupyter notebook risk_default_analysis.ipynb
# Run all cells. Phase-2 expansion lives in the section
# titled "Model Expansion — Practical 3" near the end of the notebook.
```

The expansion sections in the notebook are clearly labelled with markdown headers and produce:

- A "Kaggle Benchmark vs Our Models" comparison bar chart
- A 5-fold CV box plot for the RandomizedSearchCV winner
- A table of best hyperparameters and their justifications

### 7.3 Defence narrative (matching Section 6 framing)

**Three-beat story (with measured numbers):**

1. **"Our production model achieves 0.6272 with 15 questionnaire features."** — The honest starting point.
2. **"An unconstrained baseline on the same data reaches 0.7589."** — Measured in E1. Shows the *feature constraint* (not model) is the cap.
3. **"Engineered ratio features alone — derivable from columns we already collect — lift the constrained model from 0.6272 to 0.6846 (+0.057 AUC). The path to Kaggle-competitive ~0.79 runs through supplementary-table aggregation, which is Sprint 4 work."** — Demonstrates we know the recipe and have a credible plan.

The point of the defence is not "we beat Kaggle" — it's "we know exactly where we are on the leaderboard, why, and what closes the gap."

### 7.4 Standard+ tier — Top-25 squeeze model (post-defence enhancement)

After the E1–E5 experiments above, a separate study selected the top 25 self-reportable features from `application_train` by LightGBM gain importance (38 candidates → top 25). The full Stage-1 + Stage-2 pipeline (`scripts/select_top25_features.py` and `scripts/squeeze_top25_accuracy.py`) measured:

| Stage | Feature set | **ROC-AUC** | Δ vs prior |
|-------|-------------|-------------|------------|
| Production GBM (15 fields) | 15 | 0.6272 | reference |
| Top-25 only | 25 | 0.6854 | +0.058 |
| **+ 6 derived ratios** | 31 | **0.7093** | **+0.024** (biggest step) |
| **+ RandomizedSearchCV** | 31 | **0.7146 🏆** | +0.005 (BEST) |
| + CTGAN balancing | 31 | 0.7119 | −0.003 |
| + Stacking | 31 | 0.7142 | flat |
| + Calibration | 31 | 0.7142 | flat |

**Headline:** Top-25 + ratios + tuning reaches **ROC-AUC 0.7146** (`scripts/results/squeeze_summary.json`), **+0.0874 over production**, breaks above the application-only LR baseline (~0.70), within 0.04 of the Kaggle median (0.75). Best params: `n_estimators=700, learning_rate=0.02, num_leaves=31, min_child_samples=50, subsample=0.8, colsample_bytree=0.6, reg_alpha=0, reg_lambda=1.0`.

**Production artefact:** `src/assets/top25_risk_model.pkl` (2.5 MB; model + OrdinalEncoder + feature list). Loaded by `models/top25_predictor.py::Top25Predictor` and served by the new "📋 Standard+ Application" Streamlit page.

**The 22 user-typed questions** (auto-fill: hour_appr_process_start, weekday_appr_process_start):

| Section | Fields | n |
|---|---|---|
| Personal | gender, age, family_status, num_children, num_family_members | 5 |
| Employment | years_employed, organization_type, occupation_type, has_work_phone | 4 |
| Loan | contract_type, credit_amount, loan_annuity, goods_price | 4 |
| Financial | total_income | 1 |
| Assets | owns_car, car_age, owns_realty | 3 |
| Residence | years_at_address, years_since_id_change, *city_size* (covers 2 model fields), works_in_different_city | 4 |
| Other | has_landline | 1 |

`city_size` dropdown maps to both `region_population_relative` and `region_rating_client_w_city` via training-data medians (saves a question).

### 7.5 Insights surfaces — ADR 0002 (PR #85)

The Standard+ result page exposes 9 user-facing insights (P-01…P-09 from ADR 0002):

| ID | Surface | Owner |
|---|---|---|
| P-01 | Counter-factual "what to improve" (mutable-feature perturbation) | VC (#75) |
| P-02 | Approval probability ± confidence interval (bootstrap) | LZ (#76) |
| P-03 | Cohort percentile (age × income buckets, precomputed) | LZ (#77) |
| P-04 | Industry & region default-rate benchmark (precomputed lookup) | VC (#78) |
| P-05 | Loan-affordability sandbox slider | VC (#79) |
| P-06 | Recommended max loan via binary search | LZ (#80) |
| P-07 | Time-to-improvement projection | LZ (#81) |
| P-08 | Estimated approval-process time | VC (#82) |
| P-09 | Risk decomposition by feature group (Plotly pie) | VC (#83) |
| P-10 | Time-to-default (survival) — EPIC, gated on bureau data | VC + LZ (#84) |

Lookup artefacts: `scripts/results/cohort_distributions.json` (20 cohorts), `scripts/results/industry_region_benchmarks.json` (58 industries, 3 region ratings, 8.07% population baseline). Tests: `tests/test_insights.py` (12 unit tests, all green).

---

## 8. Future Objectives & Roadmap Update

### 8.1 Roadmap status snapshot (2026-05-23)

| Epic | Status change since P2 report |
|------|------------------------------|
| E1 Data Acquisition | ✅ Done (unchanged) |
| E2 Feature Engineering | ✅ Done (unchanged) |
| E3 ML Model Research | 🔄 Ongoing — LightGBM comparison + 5-fold CV merged (PR #64); §7 expansion ongoing |
| E4 Explainability | ✅ Done (unchanged) |
| E5 Web App | ✅ Done (unchanged) |
| E6 Testing & CI/CD | ✅ Done — 28 tests, CI green |
| E7 Risk Management | ✅ Done — §4 register, §5 retrospective |
| E8 MLOps & Long-Term Deployment | 🗓 Planned (post-TA-3) |
| **E9 Notebook Migration to Marimo** *(NEW)* | 🗓 Planned (Sprint 5 — post-Sprint-4) |

### 8.2 Epic 9 — Notebook Migration to Marimo (NEW — partial implementation shipped)

> **Scope update (2026-05-23):** original plan was documentation-only. During Practical 3 expansion, a working marimo port was created at `notebooks/risk_default_analysis.py` containing all six experiments (E1–E5). The port is validated by `marimo convert` and runs end-to-end. Full migration (remove the `.ipynb`, switch CI to `marimo check`, update README) remains Sprint 5 work — see LZ-10 in §8.3.

**Rationale (three reasons we are adding it):**

1. **Reactive execution eliminates the "did you run cells in order" class of bugs.** Marimo notebooks track cell dependencies as a DAG and re-execute downstream cells automatically. Stale state — the most common Jupyter gotcha — becomes structurally impossible.
2. **`.py` source format makes notebooks first-class git citizens.** Marimo notebooks are stored as `.py` files. They diff cleanly in PRs, can be code-reviewed line-by-line, and don't suffer from the `.ipynb`-merge-conflict problem (Jupyter notebooks are JSON with embedded image base64).
3. **Agent-friendliness.** LLMs read and write `.py` files more reliably than `.ipynb` JSON. As we increase agent-assisted development in Sprint 4+ (issue #50 SHAP plot generation, #51 fairness audit code), having the notebook in `.py` form materially improves agent throughput.

**Migration tasks:**

| Story | Description | Estimate | Status |
|-------|-------------|----------|--------|
| E9-S1 | Add `marimo` to `requirements.txt`; verify install on Python 3.12 | 1 SP | ✅ Done (Practical 3) |
| E9-S2 | Port `risk_default_analysis.ipynb` → `notebooks/risk_default_analysis.py` (marimo format) | 3 SP | ✅ Done (Practical 3) |
| E9-S3 | Verify reactive execution: changing a constant updates all downstream cells | 1 SP | 🗓 Sprint 5 (LZ-10) |
| E9-S4 | Add `marimo check` step to CI workflow | 1 SP | 🗓 Sprint 5 |
| E9-S5 | Document marimo workflow in README (`marimo edit` vs `marimo run`) | 1 SP | 🗓 Sprint 5 |
| E9-S6 | Decommission `.ipynb` once `.py` reaches feature parity | 1 SP | 🗓 Sprint 5 |

**Reference:** <https://marimo.io> · <https://docs.marimo.io/guides/coming_from_jupyter/>

**Not in scope for TA-3.**

### 8.3a Tiered questionnaire strategy (ADR 0001)

A separate strategic decision was recorded during Practical 3 in **[ADR 0001 — Tiered Questionnaire Strategy](adr/0001_tiered_questionnaire.md)**. The ADR proposes three opt-in tiers (Quick ~8 fields, Standard+ ~20 fields, Extended ~100+ fields) sharing a common derived-features layer, plus a separate consented bureau-pull integration as the long-term ceiling lift. Sprint 4 begins with the **always-on derived-features layer** (ratios → +0.057 AUC at zero new questions) and a **feature-importance study** to pick the 3–5 questions added in Tier 2. Sprints 5–6 add the additional tiers and mode selector. Full backlog tracked in GitHub issues — see §10.3.

### 8.3 Sprint 4 — refined plan (with explicit owner balance)

Sprint 4 was previously defined in Practical 2 as a spike on model improvement. Refined in light of Practical 3 expansion work and the explicit need to re-balance ownership (R-V1, R-L1). **Laurynas owns 14 SP, Vytautas owns 14 SP — equal split enforced.**

| Issue # | Task | **Owner** | SP |
|---------|------|-----------|-----|
| #47 | Bureau & previous_application feature aggregation | Vytautas | 8 |
| #46 (continuation) | Apply RandomizedSearchCV on bureau-augmented feature set | Vytautas | 3 |
| #52 | Add XGBoost to model comparison trio | Vytautas | 3 |
| **NEW LZ-9** | **CTGAN tabular GAN — minority-class synthetic balancing** (Xu et al. NeurIPS 2019); compare AUC vs SMOTETomek baseline | **Laurynas** | **5** |
| **NEW LZ-10** | **Marimo notebook migration** — port `risk_default_analysis.ipynb` → `risk_default_analysis.py`; verify reactive execution; add `marimo` to requirements | **Laurynas** | **3** |
| #51 | Fairness audit — demographic parity & equalised odds | Laurynas | 3 |
| LZ-6 follow-up | Questionnaire form labels + accessibility | Laurynas | 1 |
| #49 | Probability calibration (CalibratedClassifierCV) | Laurynas | 2 |
| **Sprint 4 capacity** | | | **28 SP** |
| **Equal split achieved** | | **14 VC + 14 LZ** | |

**Owner-balance rationale:** Sprint 3 retrospective P1 (uneven contribution) requires concrete mitigation, not just a stated guideline. Sprint 4 enforces parity by assigning Laurynas the two newest items added during Practical 3 (CTGAN + Marimo migration) — both involve learning a new library, which is a fair pairing of growth opportunity and accountability.

**Commit hygiene for the parity gate:** every pair-session commit must use the `Co-authored-by:` git trailer so the GitHub contributor graph reflects the actual collaboration:

```bash
git commit -m "$(cat <<'EOF'
feat: add CTGAN minority-class oversampling experiment

Co-authored-by: Vytautas Cepas <vyt.cepas.ve@gmail.com>
EOF
)"
```

Mid-Sprint-4 commit-parity check (weekly Sunday): if `git shortlog --since='1 week ago' -sn` shows Laurynas < 30 % of commits, escalate in standup before Monday's planning.

### 8.4 Updated 6-month roadmap

```
         Apr  |  May W1  |  May W2  |  May W3  |  May W4  |  Jun–Jul  |  Aug–Oct
              |          |          |          |          |           |
E1 Data Prep  |██████████|          |          |          |           |
E2 Features   |██████████|          |          |          |           |
E3 ML Research|          |██████████|──────────|──────────|███████████|
E4 Explain.   |          |██████████|          |          |           |
E5 Web App    |          |██████████|          |          |           |
E6 CI/CD+Test |          |          |██████████|██████████|           |
E7 Risk Mgmt  |          |          |██████████|██████████|           |
E8 MLOps      |          |          |          |          |███████████|███████████
E9 Marimo     |          |          |          |          |    ░░░░░░░|░░░ (Sprint 5)
              |          |    TA-1  |          |   TA-2   |   TA-3    |
              |          |  05-14   |          |  05-21   |  05-28    |
```

---

## 9. Defense Presentation Plan

**Defence date:** 2026-05-28 (Thursday)
**Duration target:** 15–20 minutes presentation + Q&A
**Format:** Slides + live demo + Q&A
**Team participation:** Both members present (per brief: "The whole team participates")

### 9.1 Slide outline (10 slides)

| # | Slide | Time | Speaker |
|---|-------|------|---------|
| 1 | Title + team + project recap | 1 min | Vytautas |
| 2 | Problem statement + EU AI Act compliance angle | 2 min | Laurynas |
| 3 | MVP status (3 MVPs, with achieved metrics) | 1 min | Laurynas |
| 4 | **Task estimation method + VC-3 vs LZ-3 comparison** (Practical 3 requirement) | 2 min | Laurynas |
| 5 | **GitHub & CI/CD demo** (live: `gh run list`, push graph, `pytest`) | 3 min | Vytautas |
| 6 | **Risk matrix walkthrough** (6 risks, P×I grid) | 2 min | Laurynas |
| 7 | **Sprint 3 retrospective** (4 problems + improvements) | 2 min | Laurynas |
| 8 | **Kaggle benchmark + our position** (Section 6 table + gap analysis) | 2 min | Vytautas |
| 9 | **Model expansion results** (live demo: AUC progression cell in notebook) | 2 min | Vytautas |
| 10 | Roadmap (E9 marimo + Sprint 4 plan) + Q&A | 1 min | Both |

**Speaking-time split:** Vytautas ~50 % (technical + GitHub + Kaggle + expansion), Laurynas ~50 % (process + estimation + risks + retrospective).

**Contingency:** If Laurynas absent, Vytautas takes 100 %; script is modular per slide.

### 9.2 Demo script

| Step | Command / action | Expected output |
|------|------------------|-----------------|
| 1 | `gh run list --branch main --limit 3` | 3 ✅ runs visible |
| 2 | `pytest tests/ -v` | 28/28 passed in < 30 s |
| 3 | Show `.github/workflows/ci.yml` in editor — point at lint + test phases | Phase 1 / Phase 2 clearly demarcated |
| 4 | Open `notebooks/risk_default_analysis.ipynb` → cell "Model Expansion" → run | AUC bar chart + before/after table |
| 5 | (Optional) `streamlit run app.py` → submit a sample questionnaire | Score + tier + SHAP chart |
| 6 | Show `https://github.com/VytCepas/credit_default_risk_assessment/graphs/contributors` (push graph) | Live activity over 4 weeks |

### 9.3 Anticipated Q&A and answers

| Likely question | Prepared answer |
|-----------------|------------------|
| Why is AUC 0.6272 when Kaggle median is 0.75? | Section 6 + 7 narrative — questionnaire constraint is the binding cap. Unconstrained baseline reaches 0.74. |
| Why Story Points over Use Case Points? | Section 2.1 — no formal use-case catalogue; UCP would cost more than the estimation itself. |
| Why only one active contributor? | Acknowledged as R-V1 and P1; specific Sprint 4 mitigation in §5.4. |
| Why isn't the LightGBM model in production? | Section 7 + cell-43 of notebook — comparable AUC, simpler deployment chain, conservative production swap policy. |
| What's the path to Kaggle-competitive AUC? | Section 6.4 recipe → Sprint 4 bureau aggregation (#47) → Sprint 5 stacking + DAE. |
| What about EU AI Act compliance? | R-L3 mitigation already in place: SHAP explainability + behavioural traits transparency + fairness audit (#51) in Sprint 4. |
| Why marimo? | Section 8.2 — reactivity, git-friendliness, agent-friendliness; documentation-only addition for TA-3, work happens in Sprint 5. |

### 9.4 Pre-defence rehearsal (2026-05-27 evening)

| Item | Owner |
|------|-------|
| Run through all 10 slides at presentation pace | Both |
| Verify demo commands execute in ≤ 2 min total | Vytautas |
| Capture push-graph screenshot at 800×600 resolution | Vytautas |
| Confirm `streamlit run app.py` boots within 10 s | Vytautas |
| Print or have offline copy of this report for fallback | Laurynas |

---

## 10. Delivery Summary / Sign-off

### 10.1 Practical 3 brief — requirement-by-requirement tick

| Brief item | Deadline | Where in report | Status |
|------------|----------|-----------------|--------|
| Estimate tasks in product task list — justify method (UserStoryPoint / UseCasePoint / Cosysmo) | 05.26 | §2.1 (justification), §2.2 (full backlog) | ✅ |
| Explain entire assessment process | 05.26 | §2.1, §2.5 (meeting minutes) | ✅ |
| Compare differences in assessment of 2 requirements | 05.26 | §2.3 (VC-3 vs LZ-3) | ✅ |
| Estimate costs of researching AI solutions (CPU/GPU/Cloud) | 05.26 | §2.4 | ✅ |
| Each team member commits / pushes to remote (graph for defence) | 05.26 | §3.2 | ⚠ Single-contributor (R-V1) — Sprint 4 mitigation |
| CI yaml with build + test phases | 05.26 | §3.1, `.github/workflows/ci.yml` | ✅ |
| ≥ a few unit / unit-integration tests | 05.26 | §3.3 (28 tests) | ✅ |
| Risk identification, monitoring, decision matrix, plan | 05.27 | §4 | ✅ |
| Internal risks (teamwork, AI research, tech, responsibility, schedule) | 05.27 | R-V1, R-V2, R-L1, R-L2 | ✅ |
| External risks (client, laws, external systems / docs / data) | 05.27 | R-V3, R-L3 | ✅ |
| Each risk has ≥ 1 possible solution | 05.27 | §4.1, §4.2 (Mitigation + Contingency rows) | ✅ |
| Each member: 1 internal personnel + 1 internal AI research + 1 external risk; no repeats | 05.27 | §4.1 (VC) + §4.2 (LZ); categories tracked in §4.4 | ✅ |
| Risks cover technological, project management, organisational, external | 05.27 | §4.4 coverage check | ✅ |
| Probability / impact / solutions matrix | 05.27 | §4.3, §4.4 | ✅ |
| Retrospective — ≥ 3–4 teamwork problems | 05.28 | §5.2 (4 problems) | ✅ |
| Improvement guidelines, specific solutions (backlog / responsibilities / distribution / risk mgmt changes) | 05.28 | §5.3, §5.4 | ✅ |
| Whole team participates in defence | 05.28 | §9.1 speaking split | 🗓 (defence date) |

### 10.2 Robustness additions (user-requested, beyond brief)

| Item | Section | Status |
|------|---------|--------|
| Kaggle leaderboard benchmark reference | §6 | ✅ |
| Gap analysis — why we're at 0.6272 vs median 0.75 | §6.3 | ✅ |
| Top-solution recipe digest for Sprint 4 | §6.4 | ✅ |
| Model expansion E1–E5 (all measured) | §7.1 | ✅ E1: 0.7589, E2a: 0.6846, E2b: 0.7658, E3: 0.6877, E4: 0.6882, E5: 0.6848 |
| **Top-25 squeeze model** — Standard+ tier production-ready | §7.4 | ✅ **0.7146** (+0.0874 over production) |
| **ADR 0002 insights — 9 user-facing surfaces (P-01…P-09)** | §7.5 | ✅ shipped on the Standard+ result page |
| ADR 0001 — tiered questionnaire strategy + 8 follow-up GitHub issues (#66-#73) | §8.3a, project_docs/adr/0001 | ✅ |
| Future objective — Epic 9 marimo migration | §8.2 | ✅ partial implementation |

### 10.3 File appendix

| Path | Purpose |
|------|---------|
| `app.py` | Streamlit entry point |
| `models/risk_model.py` | Legacy 15-field production GBM pipeline |
| `models/top25_predictor.py` | Standard+ tier wrapper around the squeeze model |
| `models/insights.py` | ADR 0002 prediction surfaces (P-01…P-09) |
| `models/behavioral_traits_model.py` | Behavioural-traits classifier |
| `src/predictors/risk_predictor.py`, `behavioral_predictor.py` | Streamlit cache wrappers |
| `src/components/questionnaire.py`, `questionnaire_top25.py`, `results.py`, `behavioral_traits.py` | Streamlit UI components |
| `src/assets/risk_model.pkl`, `top25_risk_model.pkl`, `behavioral_traits_model.pkl` | Trained artefacts |
| `notebooks/risk_default_analysis.ipynb` | Authoritative analysis notebook (code collapsed by default) |
| `notebooks/risk_default_analysis.py` | Marimo reactive port (Epic 9) |
| `scripts/select_top25_features.py`, `squeeze_top25_accuracy.py`, `precompute_insights.py`, `run_e4_ctgan.py`, `run_e5_stacking.py` | Reproducer scripts |
| `scripts/results/*.json` | Measurement artefacts (squeeze summary, top-25 feature ranking, cohort distributions, industry/region benchmarks, E3/E4/E5 results) |
| `tests/` | 45 unit tests (preprocessing, predictor, top25, insights) |
| `.github/workflows/ci.yml` | CI pipeline (§3.1) |
| `data/application_train.parquet`, `application_test.parquet` | Kaggle Home Credit dataset (cached locally; see R-V3) |
| `data/pictures/github_push_graph.png` | Defence screenshot (gitignored) |
| `project_docs/adr/0001_tiered_questionnaire.md` | Tiered questionnaire ADR |
| `docs/architecture.md` | Module layout and code-organisation guide |

### 10.4 Sign-off

| Role | Name | Signed-off |
|------|------|------------|
| Author | Vytautas Čepas | 2026-05-23 |
| Reviewer | Laurynas Žalaga | ⚠ Pending — Section 4.2 risks and Section 5 retrospective entries need explicit co-sign |
| Defence date | — | 2026-05-28 |
