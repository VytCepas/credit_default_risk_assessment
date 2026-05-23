# ADR 0001 — Tiered Questionnaire Strategy

**Status:** Proposed
**Date:** 2026-05-23
**Authors:** Vytautas Čepas
**Reviewers:** Laurynas Žalaga (pending), stakeholder panel (pending)
**Supersedes:** none
**Superseded by:** none

---

## Context

The production credit-risk model achieves ROC-AUC ≈ 0.6272 with the current 15-field questionnaire (`src/components/questionnaire.py`). This is 0.12 below the Kaggle median (~0.75) and 0.18 below the competition winner (~0.806) on the *same* dataset. Practical 3 expansion experiments measured the cause concretely:

| Model | Input fields | ROC-AUC | Δ vs production |
|---|---|---|---|
| Production GBM | 15 (current questionnaire) | 0.6272 | — |
| E2a: ratios derived from same 15 | **15** (still!) | **0.6846** | **+0.057** |
| E1: unconstrained, all numeric features | ~104 | 0.7589 | +0.132 |
| E2b: unconstrained + ratios + ext-interactions | ~111 | 0.7658 | +0.139 |

The gap is dominated by two sources:

1. **Engineered features we already could compute** but currently don't (5 ratios from existing inputs → +0.057 AUC, zero new questions).
2. **Features we don't collect**, primarily `EXT_SOURCE_1/2/3` (external bureau credit scores) and ~85 secondary application fields (housing statistics, social-circle defaults, document flags, regional indicators). These cannot be self-reported but could come from third-party data integrations.

The product constraint *"applicants must be able to self-report every input"* is what binds AUC to ~0.63. Any improvement strategy must either (a) compute more from existing inputs, (b) collect more from the user, or (c) integrate third-party signals via consented data pulls.

A single fixed questionnaire forces one trade-off across all users. A bank's loan officer (motivated, has the data) and a casual web visitor (impatient, exploratory) have different tolerance for question count. We need a tiered strategy.

---

## Decision

Introduce **three opt-in questionnaire tiers** sharing a common feature-engineering layer:

```
┌──────────────────────────────────────────────────────────────────┐
│              ALWAYS-ON: Derived Features Layer (D)               │
│  Server-side ratios + interactions computed from any input set:  │
│  dti, credit/income, annuity/credit, employed/age, income/family,│
│  ext_source_mean, ext_source_2 × ext_source_3 (when available)   │
└──────────────────────────────────────────────────────────────────┘
                              ▲
                              │ derived from
              ┌───────────────┼───────────────┐
              │               │               │
        ┌─────┴─────┐   ┌─────┴─────┐   ┌────┴──────────┐
        │ Tier 1    │   │ Tier 2    │   │ Tier 3        │
        │ Quick     │   │ Standard+ │   │ Extended      │
        │ ~8 fields │   │ ~20 fields│   │ ~100+ fields  │
        │ < 2 min   │   │ ~5 min    │   │ 15–30 min     │
        │ AUC ~0.60 │   │ AUC ~0.70 │   │ AUC ~0.78     │
        └───────────┘   └───────────┘   └───────────────┘
```

### Tier 1 — Quick (~8 fields, < 2 min)

For exploratory users / mobile / low-friction lead capture. A *reduced* subset of the current 15 that retains the most predictive features only.

- Fields: gender, age, total_income, employment_status, years_employed, credit_amount, loan_annuity, num_family_members
- Excludes: education, housing_type, owns_car/owns_housing, contract_type, num_children, family_status
- Predicted AUC: ~0.59–0.62 (with derived layer applied)
- **Use case:** "Eligibility check" — clearly marked as indicative, not a real underwriting decision

### Tier 2 — Standard+ (~20 fields, ~5 min) — **the new default**

A modest extension of the current 15-field questionnaire. Adds 3–5 high-value questions identified by feature-importance analysis on `application_train` (see issue [#new — feature importance study]).

- Includes all current 15 fields, **plus** candidates:
  - "Months at current address" (proxy for residential stability)
  - "Any rejected loan applications in last 12 months" (self-reported, binary)
  - "Highest delinquency in past 24 months: never / 1–30d / 31–60d / 60d+" (self-reported credit-history bucket)
  - "Open credit cards count" (self-reported)
  - "Months at current employer" (proxy more specific than `years_employed`)
- Predicted AUC: ~0.68–0.71 with derived layer
- **Use case:** the standard application path for most users; replaces the current 15-field form

### Tier 3 — Extended (~100+ fields, 15–30 min)

For loan officers entering data on behalf of customers, or for users who explicitly want maximum-accuracy assessment (e.g., pre-approval for a mortgage).

- All Tier 2 fields **plus** the remainder of the Kaggle `application_train` schema deemed user-collectable: detailed housing stats, asset valuations, full employment history, full credit-card / installment summaries (self-reported approximations).
- Excludes: `EXT_SOURCE_1/2/3` (not knowable by user — see "Future: bureau pull integration" below).
- Predicted AUC: ~0.76–0.78 with derived layer (matches Kaggle median territory).
- **Use case:** loan-officer workflow; high-value or high-risk applicants who consent to detailed disclosure; pre-approval flows.

### Derived Features Layer (D) — always on, all tiers

Server computes the following from *whatever subset of inputs the tier provides*, gracefully degrading when source fields are missing:

| Derived feature | Source fields | Always available |
|---|---|---|
| `dti` | loan_annuity / total_income | Tier 1, 2, 3 |
| `credit_to_income` | credit_amount / total_income | Tier 1, 2, 3 |
| `annuity_to_credit` | loan_annuity / credit_amount | Tier 1, 2, 3 |
| `years_employed_ratio` | years_employed / age | Tier 1, 2, 3 |
| `income_per_family_member` | total_income / num_family_members | Tier 1, 2, 3 |
| `ext_source_mean`, `ext_2_x_3` | EXT_SOURCE_* (consented bureau pull, future) | When integration ships |

Measured impact (Practical 3 §7): ratios alone lift AUC from 0.6272 → 0.6846 (+0.057). They cost zero additional user input.

### Future — Consented Bureau Data Pull (separate from tiers)

`EXT_SOURCE_*` features account for ~0.04–0.06 AUC by themselves. They cannot be self-reported. A consented soft bureau-pull integration (Equifax/Experian/local equivalent) would feed these into the Derived layer without a single new user question, lifting all tiers ~+0.05 AUC. This is **Epic 8 / MLOps & Long-Term Deployment** scope, gated on legal review (GDPR Art. 6 lawful basis, EU AI Act Art. 13 transparency).

---

## Consequences

### Positive

- Stakeholders can pick the form length that fits their user funnel; not bound to one trade-off.
- The derived-layer change ships immediately and is invisible to users (+0.057 AUC, no UI).
- Tier 3 gives us a credible defence narrative: "we have a path to Kaggle-median AUC on the same dataset, gated only on user willingness to spend 20 minutes."
- Each tier can be A/B tested independently — we will know empirically whether longer forms cost conversions or improve quality.

### Negative / risks

- **More UI surface to maintain.** Mode-selector logic, three field sets, three preprocessing branches.
  *Mitigation:* shared `QuestionnaireToFeatures` interface; each tier returns a strict superset of the previous; preprocessing handles missing-tier-fields by imputation.
- **Self-reported credit-history fields (Tier 2 additions) are unverifiable.** Users may lie.
  *Mitigation:* mark these features lower-weight in the model; never as sole basis for rejection; cross-check against bureau pull when integration ships.
- **Three models to maintain** if we train one per tier. Cost grows linearly.
  *Mitigation:* train **one model** on the union (Tier 3 fields), with missing fields imputed at inference for shorter tiers. The same artefact serves all tiers. Tier 1 inference simply has more imputed values.
- **Fairness risk widens with more questions** — more proxies for protected attributes (e.g., months-at-employer correlates with age and immigration status).
  *Mitigation:* fairness audit (issue #51, Sprint 4) runs across all three tier outputs; demographic-parity and equalised-odds thresholds enforced per tier.
- **Conversion data unknown.** We're guessing that longer forms hurt conversion; we have no telemetry yet to confirm.
  *Mitigation:* ship Tier 2 as default with mode-selector visible, capture per-tier completion rates from week 1; revisit policy once we have 4 weeks of telemetry.

### Open questions for stakeholder review

1. Which 3–5 fields go into Tier 2 — confirm via the feature-importance study (issue [E14]).
2. Is the consented bureau-pull integration on the strategic roadmap, and if so what's the legal-review cadence?
3. Default tier on first visit — Tier 1 or Tier 2?
4. Whether Tier 3 needs distinct UI (multi-step wizard) or can be served by a single very long form.
5. Marketing/UX copy for the mode selector — "Why are we asking this?" microcopy per field.

---

## Implementation phasing

| Phase | Sprint | Issues | Scope |
|---|---|---|---|
| **0 — Decision recorded** | This ADR | — | This document; stakeholder review |
| **1 — Derived features in production** | Sprint 4 | [E10] | Ship the 5 ratios server-side; retrain `risk_model.pkl`; no UI change |
| **2 — Feature-importance study** | Sprint 4 | [E14] | Rank `application_train` features for Tier 2 candidates; recommend top 5 |
| **3 — Tier 2 questionnaire + mode selector** | Sprint 5 | [E11], [E13] | Add 3–5 fields to form; build mode-selector UI; default tier = 2 |
| **4 — Tier 1 quick mode** | Sprint 5 | [E12a] | Reduced-field path with explicit "indicative only" copy |
| **5 — Tier 3 extended mode** | Sprint 6 | [E12b] | Loan-officer / opt-in extended form |
| **6 — Consented bureau integration** | Epic 8 (MLOps) | [E15] | Soft credit pull → EXT_SOURCE_* into derived layer |
| **7 — Per-tier fairness audit** | rolling | [#51 + extension] | Demographic-parity check per tier |

---

## References

- Practical 3 report, §6 (Kaggle benchmark) and §7 (model expansion measured results)
- `notebooks/risk_default_analysis.ipynb` (cells E1, E2a, E2b — measured AUC numbers)
- Kaggle Home Credit Default Risk competition: <https://www.kaggle.com/c/home-credit-default-risk>
- Aguiar public kernel (top-100 reference): <https://github.com/js-aguiar/home-credit-default-competition>
- CTGAN paper (Xu et al. NeurIPS 2019): <https://arxiv.org/abs/1907.00503>
