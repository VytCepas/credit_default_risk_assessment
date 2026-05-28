# ADR 0002 — Additional Prediction Surfaces (P-01 … P-10)

**Status:** Implemented (P-01 … P-09); P-10 deferred (EPIC, gated on bureau data)
**Date:** 2026-05-23
**Authors:** Vytautas Čepas
**Reviewers:** Laurynas Žalaga (pending)
**Supersedes:** none
**Superseded by:** none

---

## Context

The Standard+ result page from [ADR 0001](0001_tiered_questionnaire.md) shows
the user a single number: the risk score (0–1000) and its tier. That answers
*"what is my score"* but leaves several recurring user questions on the table:

- *"Why is my score what it is?"*
- *"How can I make it better?"*
- *"Am I unusual?"*
- *"What if I borrowed less?"*
- *"What's the most I could borrow and still be approved?"*
- *"How long does this approval take?"*

These are the same questions the EU AI Act Article 13 transparency
obligations push us toward: a credit-scoring system should explain its
decision in terms the applicant can act on, not just emit a probability.

We had two options:

1. **Defer everything beyond the score** and rely on the embedded SHAP bar chart for "why".
2. **Build a small catalogue of focused result-page surfaces**, each answering one of the recurring questions in its own panel.

Option 1 was simpler but produced a thin result page that didn't justify the
five minutes of form-filling. Option 2 turns the result page into the
product.

## Decision

Ship **nine focused prediction surfaces** as result-page panels, derived
from the same single `predict()` call (plus the precomputed lookup
artefacts). Each surface is implemented as a small pure function in
[`models/insights.py`](../../models/insights.py) — no Streamlit imports —
and called once by `app.py::show_result_page()`.

| ID | Surface | Answers | Owner |
|---|---|---|---|
| **P-01** | Counter-factual recommendations | "What single mutable input, if changed by how much, would drop my risk most?" | Vytautas (#75) |
| **P-02** | Approval probability ± CI | "How confident is the model?" | Laurynas (#76) |
| **P-03** | Cohort percentile | "Among similar applicants, where do I rank?" | Laurynas (#77) |
| **P-04** | Industry & region default-rate benchmark | "What's typical for my organisation / region?" | Vytautas (#78) |
| **P-05** | Loan-affordability sandbox | "What happens if I borrow €X?" | Vytautas (#79) |
| **P-06** | Recommended max loan (binary search) | "What's the largest loan that keeps me under Medium-risk?" | Laurynas (#80) |
| **P-07** | Time-to-improvement projection | "How long until my risk drops to tier T?" | Laurynas (#81) |
| **P-08** | Estimated approval-process time | "Roughly how long does an application take?" | Vytautas (#82) |
| **P-09** | Risk decomposition by feature group | "Which broad category drove my score?" (SHAP, log-odds units) | Vytautas (#83) |
| P-10 | *Time-to-default survival* — **deferred** | "If approved, what's my likely time-to-default distribution?" | Both (#84) — gated on bureau data |

### Implementation rules

- **Pure functions only.** Each surface is a function that takes the predictor
  + the form dict (+ optional precomputed lookup) and returns a JSON-safe dict.
  No `streamlit` imports in `models/insights.py`.
- **Graceful degradation.** Surfaces that depend on precomputed artefacts
  (P-03 cohort lookup, P-04 industry/region) **must** return an empty / stub
  dict when the artefact is missing, never raise. The result page silently
  hides those tabs.
- **SHAP attribution stays in log-odds units.** Do not convert P-09 to a
  pseudo-percentage — log-odds are signed, can be negative, and that signal
  is meaningful to the user (red = increases risk, green = decreases). The
  test for P-09 asserts the SHAP invariant
  (`sum(shap) + base_value ≈ raw_logit`), not a 0–100 sum.
- **Same data, no second model.** Every surface is computed from the existing
  `Top25Predictor` and the same form input. P-10 is the only one that would
  introduce a second model (a survival regressor on the bureau loan history),
  and is therefore deferred.

### Precomputed lookups

| Artefact | Produced by | Used by |
|---|---|---|
| `scripts/results/cohort_distributions.json` (20 age × income cohorts) | `scripts/precompute_insights.py` | P-03 |
| `scripts/results/industry_region_benchmarks.json` (58 industries × 3 region ratings, 8.07 % population baseline) | `scripts/precompute_insights.py` | P-04 |

Both are committed to the repo so the app boots without re-running the
precompute step. The app degrades gracefully if either is missing.

---

## Consequences

### Positive

- **Result page becomes the product.** Justifies the 22-question form length.
- **Each surface is independently testable.** Pure functions → unit tests with no Streamlit fixture noise. Current coverage: 12 unit tests in `tests/test_insights.py` + 2 integration tests in `tests/test_integration.py` + 4 SHAP-invariant tests in `tests/test_shap_validation.py`.
- **Transparent path to EU AI Act Article 13 compliance.** P-01 + P-09 deliver per-decision explanations; P-03 + P-04 deliver "similarly-situated" context; P-06 + P-07 deliver actionable next steps.
- **No second model to maintain.** Every surface composes off the production `Top25Predictor`. The only added artefacts are the two JSON lookups.

### Negative / risks

- **SHAP in log-odds is conceptually heavier** than a 0–100 pie chart. The UI mitigates with red/green coloring and a caption explaining "log-odds units". A future ADR could revisit if user testing shows confusion.
- **Counter-factuals (P-01) can recommend impossible changes** (e.g., "be 5 years older"). Mitigation: only mutable features are perturbed — `age`, `years_employed`, `years_at_address` are mutable in the optimistic-projection sense (you wait); identity columns are excluded.
- **Cohort & industry buckets are precomputed**, so they go stale if the underlying training data drifts. Refresh cadence: rerun `precompute_insights.py` whenever `top25_risk_model.pkl` is retrained.
- **P-05/P-06 sandbox / binary search call `predictor.predict()` repeatedly**, contributing to result-page latency. Each call is ~5 ms locally; binary search caps at ~15 iterations; total budget well under the 3 s NFR (tests/test_perf.py).

### Open questions for stakeholder review

1. Should P-09 ever be re-expressed as a pseudo-percentage view (advanced users keep log-odds, casual users see "Loan factors: 45%")? Defer until UX feedback.
2. P-07's "time to improvement" assumes time-passing-equally for `years_employed` and `years_at_address`. Is it misleading to imply both will tick up together? Possibly — refine after first round of user testing.
3. P-08 currently uses a static lookup keyed on `contract_type` and tier. Should it use Home Credit's *actual* historical approval-time distribution? Yes, when we get bureau data.

---

## Implementation status

| Surface | Issue | Status |
|---|---|---|
| P-01 Counter-factuals | #75 | ✅ Shipped |
| P-02 Approval prob + CI | #76 | ✅ Shipped |
| P-03 Cohort percentile | #77 | ✅ Shipped |
| P-04 Industry/region benchmark | #78 | ✅ Shipped |
| P-05 Affordability sandbox | #79 | ✅ Shipped |
| P-06 Recommended max loan | #80 | ✅ Shipped |
| P-07 Time-to-improvement | #81 | ✅ Shipped |
| P-08 Approval-process time | #82 | ✅ Shipped |
| P-09 Risk decomposition (SHAP) | #83 | ✅ Shipped |
| P-10 Time-to-default survival | #84 | 🗓 Deferred — requires bureau data; tracked as an EPIC |

All shipped surfaces are visible on the Standard+ result page tabs.

---

## References

- [ADR 0001](0001_tiered_questionnaire.md) — Tiered Questionnaire Strategy (the question shape this answers).
- [`models/insights.py`](../../models/insights.py) — implementation.
- [`tests/test_insights.py`](../../tests/test_insights.py), [`tests/test_integration.py`](../../tests/test_integration.py), [`tests/test_shap_validation.py`](../../tests/test_shap_validation.py) — tests.
- [`scripts/precompute_insights.py`](../../scripts/precompute_insights.py) — precomputed lookup generator.
- EU AI Act Regulation (EU) 2024/1689, Article 13 — transparency obligations for high-risk AI systems.
- SHAP — Lundberg & Lee 2017, *A Unified Approach to Interpreting Model Predictions*.
- Practical 3 report §7.5 — the original captured-inline version this ADR replaces.
