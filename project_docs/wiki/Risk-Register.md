# Risk Register

Source of truth: [Practical 3 report §4](https://github.com/VytCepas/credit_default_risk_assessment/blob/main/project_docs/practical_3_report.md#4-risk-management).

This page is the searchable summary.

## Method

Probability × Impact 5×5 matrix, qualitative scoring. Each axis 1 (very low)
to 5 (very high). Score = P × I. Zones:

| Score | Zone | Cadence |
|---|---|---|
| ≥ 12 | 🔴 Red | Mitigation owner reports every standup |
| 6–11 | 🟡 Amber | Weekly review |
| ≤ 5 | 🟢 Green | Quarterly review |

## Register

| ID | Risk | Category | P | I | Score | Owner | Zone |
|---|---|---|---|---|---|---|---|
| **R-V1** | Single-contributor bus factor | Org / Personnel | 3 | 5 | **15** | Vytautas | 🔴 |
| **R-V2** | ROC-AUC plateaus below Kaggle median | Tech / AI research | 4 | 3 | **12** | Vytautas | 🔴 |
| **R-L1** | Schedule slip on LZ-owned tasks | PM / Personnel | 4 | 3 | **12** | Laurynas | 🔴 |
| **R-L2** | SMOTETomek / resampling leakage | Tech / AI research | 2 | 4 | 8 | Laurynas | 🟡 |
| **R-L3** | EU AI Act Article 13 enforcement shift | External | 2 | 4 | 8 | Laurynas | 🟡 |
| **R-V3** | Kaggle dataset access / licence change | External | 2 | 3 | 6 | Vytautas | 🟡 |

## Probability × Impact matrix

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
    │            │       (8)  │            │            │            │
  ──┼────────────┼────────────┼────────────┼────────────┼────────────┤
  1 │            │            │            │            │            │
  ──┴────────────┴────────────┴────────────┴────────────┴────────────┘
        1            2            3            4            5
                              Probability
```

## Mitigations in flight

| Risk | Mitigation | Status |
|---|---|---|
| R-V1 | Sprint 4 pair-programming pact; Laurynas owns ≥ 2 tasks (#51 fairness audit, LZ-6 form refinement); weekly `git shortlog` parity check | committed |
| R-V2 | Section 7 expansion experiments measured; Sprint 4 bureau-aggregation work (#47) targeting AUC ≥ 0.70 | committed |
| R-L1 | Explicit Sprint 4 ownership reassignment in writing (GitHub issues); daily mid-sprint commit-status check | committed |
| R-L2 | Pipeline-only resampling; CI test asserting `imblearn.Pipeline` usage (Sprint 4) | planned |
| R-L3 | SHAP explainability + behavioural-traits transparency layer already shipped; fairness audit (#51) Sprint 4 | partial |
| R-V3 | Dataset cached locally + committed; documented Kaggle version + download date; UCI Credit Card Default as fallback | done |

## Coverage check (per Practical 3 brief)

| Required category | Risk(s) covering it |
|---|---|
| Technological | R-V2, R-L2 |
| Project management | R-L1 |
| Organisational | R-V1 |
| External | R-V3, R-L3 |
| Internal personnel | R-V1, R-L1 |
| Internal AI research | R-V2, R-L2 |

## Monitoring cadence

| When | Activity | Owner |
|---|---|---|
| Daily standup | "Blockers" question maps to ↑ probability on any active risk | Both |
| Weekly (Sunday) | `git shortlog --since='1 week ago'` (R-V1); review experiment journal (R-V2, R-L2) | Vytautas |
| Per-sprint | Re-score P/I; re-rank risks | Both |
| Quarterly | Refresh external risks (R-V3, R-L3) | Laurynas |
