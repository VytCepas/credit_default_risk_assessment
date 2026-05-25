# ADR Index

Architecture Decision Records live in
[`project_docs/adr/`](https://github.com/VytCepas/credit_default_risk_assessment/tree/main/project_docs/adr).
Each ADR follows the same template: Context → Decision → Consequences →
Implementation phasing → References.

| # | Title | Status | Date | Summary |
|---|---|---|---|---|
| [0001](https://github.com/VytCepas/credit_default_risk_assessment/blob/main/project_docs/adr/0001_tiered_questionnaire.md) | Tiered Questionnaire Strategy | Proposed | 2026-05-23 | Three opt-in tiers (Quick / Standard+ / Extended) sharing a derived-features layer, with a separate consented bureau-pull track for the long-term ceiling lift |
| [0002](https://github.com/VytCepas/credit_default_risk_assessment/blob/main/project_docs/adr/0002_insights_surfaces.md) | Additional Prediction Surfaces (P-01 … P-10) | Implemented (P-01…P-09); P-10 deferred | 2026-05-23 | Nine focused result-page surfaces; pure functions in `models/insights.py`; SHAP stays in log-odds units; precomputed lookups for cohort + industry benchmarks |

## Authoring a new ADR

1. Copy `0001_tiered_questionnaire.md` to `NNNN_short_title.md` (next sequential number).
2. Fill **Context** (what state are we in, what's the forcing function), **Decision** (what we picked), **Consequences** (positive + negative + open questions), **Implementation phasing**, **References**.
3. Status starts at `Proposed`. After team agreement, change to `Accepted`. If superseded, link the successor and change to `Superseded by ADR-XXXX`.
4. Open a PR — ADR reviews are about the decision, not the prose.

## When to write an ADR vs. inline doc

| Write an ADR | Inline doc / commit message |
|---|---|
| Decision that changes the **shape** of the system (interface boundary, module split, data flow) | Implementation detail of an existing decision |
| Decision that we will **regret being unable to recall in 6 months** | Decision we will rediscover by reading the code |
| Decision with stakeholders **outside** the immediate dev team | Dev-internal taste call |
| Decision that **rules out alternatives** future contributors might re-propose | Decision that's obviously correct given the constraint |

When in doubt: write the ADR. Five small ADRs cost less than one wrong
re-decision.
