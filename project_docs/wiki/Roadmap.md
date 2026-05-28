# Roadmap

Snapshot source: [Practical 3 report §8](https://github.com/VytCepas/credit_default_risk_assessment/blob/main/project_docs/practical_3_report.md#8-future-objectives--roadmap-update).

## Epic status (2026-05-23)

| Epic | Status |
|---|---|
| E1 Data Acquisition | ✅ Done |
| E2 Feature Engineering | ✅ Done |
| E3 ML Model Research | 🔄 Ongoing — LightGBM, RandomizedSearchCV merged; Sprint-4 bureau aggregation pending |
| E4 Explainability | ✅ Done (SHAP, P-09 decomposition) |
| E5 Web App | ✅ Done (Standard+ single flow) |
| E6 Testing & CI/CD | ✅ Done (CI green, 17 tests) |
| E7 Risk Management | ✅ Done (register, retrospective) |
| E8 MLOps & Long-Term Deployment | 🗓 Planned (post-TA-3) |
| E9 Notebook Migration to Marimo | 🗓 Partial — two notebooks ported; full migration Sprint 5 |

## Sprint 4 — refined plan (parity-enforced)

**Equal split: Vytautas 14 SP / Laurynas 14 SP.**

| Issue # | Task | Owner | SP |
|---|---|---|---|
| #47 | Bureau & previous_application feature aggregation | Vytautas | 8 |
| #46 (cont.) | Apply RandomizedSearchCV on bureau-augmented feature set | Vytautas | 3 |
| #52 | Add XGBoost to model comparison trio | Vytautas | 3 |
| LZ-9 | CTGAN tabular GAN — minority-class synthetic balancing; compare AUC vs SMOTETomek baseline | Laurynas | 5 |
| LZ-10 | Marimo notebook migration — full port + reactive verification + `marimo check` in CI | Laurynas | 3 |
| #51 | Fairness audit — demographic parity & equalised odds | Laurynas | 3 |
| LZ-6 follow-up | Questionnaire form labels + accessibility | Laurynas | 1 |
| #49 | Probability calibration (CalibratedClassifierCV) | Laurynas | 2 |

**Owner-balance rationale:** Sprint 3 retrospective P1 (uneven contribution)
requires concrete mitigation. Sprint 4 enforces parity by assigning Laurynas
the two newest items (CTGAN + Marimo migration) — both involve learning a
new library, which is a fair pairing of growth opportunity and accountability.

## 6-month roadmap (Gantt-ish)

```
         Apr  |  May W1  |  May W2  |  May W3  |  May W4  |  Jun–Jul  |  Aug–Oct
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

## Tiered questionnaire — sequencing

From [ADR 0001](https://github.com/VytCepas/credit_default_risk_assessment/blob/main/project_docs/adr/0001_tiered_questionnaire.md):

| Phase | Sprint | Scope |
|---|---|---|
| Phase 0 — decision recorded | this ADR | document; stakeholder review |
| Phase 1 — derived features in production | Sprint 4 | ship the 5 ratios server-side (✅ already in the Top-25 model) |
| Phase 2 — feature-importance study | Sprint 4 | rank `application_train` for Tier 2 candidates |
| Phase 3 — Tier 2 questionnaire + mode selector | Sprint 5 | add 3–5 fields; build mode selector |
| Phase 4 — Tier 1 Quick mode | Sprint 5 | reduced-field path with "indicative only" copy |
| Phase 5 — Tier 3 Extended mode | Sprint 6 | loan-officer / opt-in extended form |
| Phase 6 — consented bureau integration | Epic 8 | soft credit pull → EXT_SOURCE_* into derived layer |
| Phase 7 — per-tier fairness audit | rolling | demographic-parity check per tier |
