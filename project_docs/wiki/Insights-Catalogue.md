# Insights Catalogue (P-01 … P-09)

The Standard+ result page surfaces nine user-facing insights, derived from
[ADR 0002](https://github.com/VytCepas/credit_default_risk_assessment/blob/main/project_docs/practical_3_report.md#75-insights-surfaces--adr-0002-pr-85)
and implemented in [`models/insights.py`](https://github.com/VytCepas/credit_default_risk_assessment/blob/main/models/insights.py).

| ID | Surface | What it answers | Owner |
|---|---|---|---|
| **P-01** | **Counter-factual** | "What single mutable input, if changed by how much, would drop my risk most?" | VC (#75) |
| **P-02** | **Approval probability ± CI** | "How confident is the model in its decision?" | LZ (#76) |
| **P-03** | **Cohort percentile** | "Among similar applicants (age × income bucket), where do I rank?" | LZ (#77) |
| **P-04** | **Industry & region benchmark** | "What's the typical default rate for my organisation type / region?" | VC (#78) |
| **P-05** | **Loan-affordability sandbox** | "What happens to my risk if I borrow €X instead of €Y?" — live slider | VC (#79) |
| **P-06** | **Recommended max loan** | "What's the largest loan that still keeps me under the Medium-risk line?" | LZ (#80) |
| **P-07** | **Time-to-improvement** | "If I make change C, how long until my risk drops to tier T?" | LZ (#81) |
| **P-08** | **Approval-process time** | "Roughly how long does an application take?" | VC (#82) |
| **P-09** | **Risk decomposition by feature group** | "Which broad category drove my score?" — Plotly pie | VC (#83) |
| P-10 | *Time-to-default survival* | (EPIC, gated on bureau data) | VC + LZ (#84) |

## How each surface is computed

| Surface | Approach |
|---|---|
| P-01 Counter-factual | Perturb each *mutable* feature ±1 unit / category, recompute probability, sort descending by absolute Δ |
| P-02 Approval prob CI | Bootstrap N predictions over training-set noise; report mean ± 1.96 × SE |
| P-03 Cohort percentile | Precomputed cohort distribution (age bucket × income bucket × default rate) loaded from `scripts/results/cohort_distributions.json` (20 cohorts) |
| P-04 Industry benchmark | Precomputed lookup from `scripts/results/industry_region_benchmarks.json` (58 industries × 3 region ratings; 8.07 % population baseline) |
| P-05 Affordability sandbox | Streamlit `st.slider` over `credit_amount`; each tick re-runs the predictor (cached) |
| P-06 Recommended max loan | Binary-search `credit_amount` against tier-boundary probability (default 0.30 ≈ Medium ceiling) |
| P-07 Time-to-improvement | Treat `years_employed`, `years_at_address` etc. as monotonic; project months-to-tier-change |
| P-08 Approval-process time | Static lookup driven by `contract_type`; calibrated against Home Credit's published averages |
| P-09 Risk decomposition | SHAP values summed by feature group (Personal / Employment / Loan / Financial / Assets / Residence / Other) |

## Precomputed data shipping with the repo

| Artefact | Contents | Used by |
|---|---|---|
| `scripts/results/cohort_distributions.json` | 20 (age × income) cohorts with default rates | P-03 |
| `scripts/results/industry_region_benchmarks.json` | 58 industries × 3 region ratings, 8.07 % population baseline | P-04 |

Both are produced by `scripts/precompute_insights.py` and committed to the
repo so the app boots without re-running them. The app degrades gracefully
if either file is missing.

## Tests

[`tests/test_insights.py`](https://github.com/VytCepas/credit_default_risk_assessment/blob/main/tests/test_insights.py) — 12 unit tests covering the surfaces above (output bounds, monotonicity, fallback behavior when lookups are absent). All green in CI.

## When to use which

- **"Why this score?"** → P-01 counter-factual + P-09 decomposition
- **"How sure are you?"** → P-02 confidence interval
- **"Am I unusual?"** → P-03 cohort + P-04 industry
- **"What if I asked for less?"** → P-05 sandbox + P-06 recommended max
- **"How can I improve?"** → P-07 time-to-improvement
- **"How long until I hear back?"** → P-08
