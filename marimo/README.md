# marimo notebooks

[Marimo](https://marimo.io) is a Python notebook with reactive execution
(cells re-run automatically when their inputs change), stored as plain
`.py` files that diff cleanly in git. This folder mirrors `notebooks/`
but in marimo format.

| File | Mirrors | Purpose |
|------|---------|---------|
| `risk_default_analysis.py` | `notebooks/risk_default_analysis.ipynb` | Port of the authoritative Jupyter notebook — full E1–E5 experiment chain. |
| `top25_squeeze.py` | (new — no Jupyter twin yet) | Reactive playground for the production Standard+ model: load the pickle, show feature importance, expose what-if sliders, surface insights (counter-factuals, approval probability, recommended max loan). |

## Running

```bash
# Edit mode (notebook UI in the browser)
.venv/bin/marimo edit marimo/risk_default_analysis.py
.venv/bin/marimo edit marimo/top25_squeeze.py

# Read-only app mode
.venv/bin/marimo run marimo/top25_squeeze.py
```

## Prerequisites

- `pip install -r requirements.txt` (marimo is already pinned there)
- For `top25_squeeze.py` you need:
  - `src/assets/top25_risk_model.pkl` — produced by
    `scripts/squeeze_top25_accuracy.py`
  - `scripts/results/cohort_distributions.json` and
    `industry_region_benchmarks.json` — produced by
    `scripts/precompute_insights.py` (optional; some insights skip
    gracefully if absent)

## Why marimo alongside Jupyter?

- **Reactive execution.** No more "did you run cells in order" stale-state
  bugs — when you change a slider or a constant, every downstream cell
  re-runs automatically.
- **`.py` source format.** Diffs cleanly in PR review (the `.ipynb` is
  JSON with embedded base64 outputs that don't diff well). Easier for
  LLM-assisted edits too.
- **Marimo `run` mode.** A notebook can also be served as a read-only
  interactive app — useful for demos that don't need a full Streamlit
  stack.

Migration tracked as **Epic 9** in the project ADR / roadmap. Long-term
plan is to decommission the `.ipynb` once parity is reached, but for now
both formats live side-by-side.

## Sister directories

- `../notebooks/` — Jupyter `.ipynb` notebooks (authoritative for now)
- `../scripts/` — one-shot experiment and pre-compute scripts
- `../models/` — Python model classes consumed by both
