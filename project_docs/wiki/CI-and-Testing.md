# CI and Testing

## Pipeline

Source: [`.github/workflows/ci.yml`](https://github.com/VytCepas/credit_default_risk_assessment/blob/main/.github/workflows/ci.yml)

**Triggers**

- `push` on every branch
- `pull_request` targeting `main`

**Phase 1 — Lint (flake8)**

- Runner: `ubuntu-latest`, Python 3.12, pip cache
- Scope: `src/` `models/` `tests/` (NOT `app.py` — see below)
- Config: `--max-line-length=100 --extend-ignore=E203,W503 --exclude=__pycache__,.venv`

**Phase 2 — Test (pytest)**

- Depends on lint (`needs: lint`) — broken syntax never wastes test runtime
- Installs `requirements.txt` + `pytest` + `pytest-cov`
- Runs `pytest tests/ -v --cov=models.top25_predictor --cov=models.insights --cov-report=term-missing --cov-report=xml --cov-fail-under=75`
- **Coverage gate at 75 %** — currently measuring ~85 % over the tested modules. Coverage is scoped to `models.top25_predictor` + `models.insights` because those are the modules under test; including untested modules just to inflate the denominator would game the metric. Add untested modules to the `--cov=` flags when they get tests.
- Coverage report uploaded as artifact `coverage-report` (30-day retention)

## Why `app.py` is outside the lint scope

`app.py` is dense with long Plotly traces and inline HTML/CSS. The 100-char
line limit fights this with no benefit. Keep new code that needs to be linted
in `src/components/` or `models/` and call from `app.py` instead of fattening
the entry point.

## Test inventory (24 tests across 5 files)

### [`tests/test_top25_predictor.py`](https://github.com/VytCepas/credit_default_risk_assessment/blob/main/tests/test_top25_predictor.py) — 5 tests

| What | Verifies |
|---|---|
| Bundle loads | `joblib.load` returns the expected keys; reported AUC > 0.70 |
| Output shape | `predict()` returns the documented dict |
| Output bounds | Probability ∈ [0,1]; score ∈ [0, 1000]; tier ∈ {Low, Medium, High} |
| Differentiation | Low-risk profile scores lower than high-risk |
| Consistency / missing fields | Same input → same output; optional fields can be absent |

### [`tests/test_insights.py`](https://github.com/VytCepas/credit_default_risk_assessment/blob/main/tests/test_insights.py) — 12 tests

| What | Verifies |
|---|---|
| Counter-factuals (P-01) | Returns sorted list; deltas signed correctly |
| Approval prob CI (P-02) | Lower ≤ mean ≤ upper; band ∈ {high, medium, low} |
| Cohort percentile (P-03) | Returns 0–100; falls back gracefully when lookup absent |
| Industry benchmark (P-04) | Returns default rate; falls back gracefully |
| Affordability sandbox (P-05) | Monotonic in credit amount |
| Recommended max loan (P-06) | Binary search converges; result respects tier boundary |
| Time-to-improvement (P-07) | Returns valid months or already-at-target flag |
| Process time (P-08) | Tier × completeness table covers all combinations |
| Risk decomposition (P-09) | SHAP groups are finite log-odds; per-feature sum matches group totals |

### [`tests/test_integration.py`](https://github.com/VytCepas/credit_default_risk_assessment/blob/main/tests/test_integration.py) — 2 tests

| What | Verifies |
|---|---|
| Full pipeline | Form → predict → 7 insights surfaces; mirrors `app.py::show_result_page` without Streamlit |
| Contract lock | `predict()` keys match what `app.py` consumes |

### [`tests/test_shap_validation.py`](https://github.com/VytCepas/credit_default_risk_assessment/blob/main/tests/test_shap_validation.py) — 4 tests

| What | Verifies |
|---|---|
| No NaN / inf | Every SHAP value and base value is finite |
| SHAP invariant | `sum(shap) + base_value ≈ model.predict(raw_score=True)` within 1e-4 |
| Top features known | Top-5 names are all in `predictor.feature_set` |
| Unsupported model fallback | Stub dict returned when `shap.TreeExplainer` raises |

### [`tests/test_perf.py`](https://github.com/VytCepas/credit_default_risk_assessment/blob/main/tests/test_perf.py) — 1 test

| What | Verifies |
|---|---|
| P95 prediction latency | 20 consecutive predictions; P95 well under the 3 s NFR budget (current local ~0.005 s) |

Run locally:

```bash
pytest tests/ -v --cov=models --cov=src --cov-report=term-missing
```

## What CI **does not** do (yet)

- **No `marimo check`.** Tracked as Sprint 5 work (E9-S4).
- **No security/dependency scanning.** Not in scope for an academic project.
- **`behavioral_traits_model` and `behavioral_predictor` are uncovered.** Their absence from the `--cov=` scope means a regression there won't trip the 75 % gate. Add tests + extend `--cov=` flags when working on those modules.

## Recent CI runs

`gh run list --branch main --limit 5` from the repo root.

## Pre-commit advice

```bash
# Lint just what changed
git diff --name-only --cached | grep -E '\.py$' | xargs flake8 --max-line-length=100 --extend-ignore=E203,W503

# Quick test
pytest tests/ -q
```

(Not enforced as a git hook — pick whichever ergonomics fit you.)
