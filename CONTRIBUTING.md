yo# Contributing

This is an academic term project, but contributions and reviews are welcome.
Follow these conventions so PRs land cleanly through CI.

## Workflow

1. Branch from `main`: `git checkout -b feat/<short-name>` (or `fix/`, `docs/`, `chore/`).
2. Make focused commits — one logical change per commit.
3. Open a PR to `main` with a 1-paragraph "why" + a Test Plan checklist.
4. CI must be green before merge — no `--no-verify`, no skipped phases.
5. Squash-merge from the GitHub UI; the PR title becomes the commit message.

## Commit messages

Conventional-commit prefixes used in the project history:

| Prefix | When |
|---|---|
| `feat:` | New capability, model, or page |
| `fix:` | Bug fix or test correction |
| `docs:` | Docs-only change |
| `chore:` | Tooling, requirements, CI config |
| `refactor:` | Code restructuring with no behavior change |
| `test:` | Test-only addition |

Pair-session commits use the `Co-authored-by:` trailer so the contributor graph
reflects actual collaboration (see Practical 3 §8.3).

## Code style

- **Lint:** `flake8 src/ models/ tests/ --max-line-length=100 --extend-ignore=E203,W503`
- **Formatter:** [Black](https://black.readthedocs.io) (pinned in `requirements.txt`); E203/W503 are ignored to interop with it.
- **`app.py`** is intentionally *outside* the lint scope (long Plotly/markup
  lines) — keep that boundary; do not add new heavy files outside the scope
  to avoid the linter.
- **Type hints** on public functions in `models/` and `src/predictors/`. Notebook
  cells and one-shot scripts are exempt.
- **Docstrings:** module + class + public function. One short line is fine for
  obvious helpers; longer when "why" is non-obvious. Avoid commentary that
  duplicates the code.

## Tests

- New behavior in `models/` or `src/predictors/` → add a test in `tests/`.
- Tests use plain `pytest`; no fixtures framework beyond what comes with it.
- Run locally: `pytest tests/ -v --cov=models --cov=src --cov-report=term-missing`.
- CI uploads `coverage.xml` as an artifact named `coverage-report`.

## ML changes that touch the production model

If you change `scripts/squeeze_top25_accuracy.py` or anything that produces
`src/assets/top25_risk_model.pkl`:

1. Re-run the script and commit the new pickle **and** the new
   `scripts/results/squeeze_summary.json`.
2. Quote the before/after ROC-AUC in the PR description.
3. Re-run `pytest tests/test_top25_predictor.py` — the bundle schema must stay
   stable, or the tests will catch a missing key.

## ADRs

Architecture-shaping decisions go in `project_docs/adr/` using the existing
template (`0001_tiered_questionnaire.md`). Bump the next number; do not edit
old ones — supersede instead.

## Wiki

The source of the GitHub Wiki lives under `project_docs/wiki/`. To update the
live wiki:

```bash
# one-time:
git clone https://github.com/VytCepas/credit_default_risk_assessment.wiki.git wiki-live

# sync:
cp project_docs/wiki/*.md wiki-live/
cd wiki-live && git add -A && git commit -m "docs(wiki): sync from project_docs/wiki" && git push
```

Keep the in-repo source as the canonical edit point so wiki changes go through
PR review like everything else.

## Useful commands

```bash
streamlit run app.py                                                  # dev server
pytest tests/ -v                                                      # tests
flake8 src/ models/ tests/ --max-line-length=100 --extend-ignore=E203,W503
.venv/bin/marimo edit marimo/top25_squeeze.py                         # what-if playground
gh run list --branch main --limit 5                                   # recent CI runs
```
