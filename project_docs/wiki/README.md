# Wiki source

The `.md` files in this folder are the **source of the GitHub Wiki** at
<https://github.com/VytCepas/credit_default_risk_assessment/wiki>.

Editing here (and reviewing via PR) is preferred over editing the live wiki
directly — that way wiki changes get the same code-review treatment as the
rest of the project.

## Pages

| File | Wiki page |
|---|---|
| `Home.md` | Wiki home / index |
| `Project-Overview.md` | Project Overview |
| `Architecture.md` | Architecture |
| `Modeling-Pipeline.md` | Modeling Pipeline |
| `Standard-Plus-Questionnaire.md` | Standard+ Questionnaire |
| `Insights-Catalogue.md` | Insights Catalogue |
| `CI-and-Testing.md` | CI and Testing |
| `Risk-Register.md` | Risk Register |
| `Roadmap.md` | Roadmap |
| `ADR-Index.md` | ADR Index |
| `Glossary.md` | Glossary |

## Page naming

The GitHub Wiki uses the **file name** (without `.md`) as the page slug, and
hyphens become spaces in the page title. Keep new files in `Title-Case-With-Hyphens.md`.

## Syncing to the live wiki

The GitHub Wiki is a separate git repo (`.wiki.git`). To push the contents of
this folder to the live wiki:

```bash
# one-time clone (somewhere outside this repo)
git clone https://github.com/VytCepas/credit_default_risk_assessment.wiki.git wiki-live
cd wiki-live

# sync (re-run any time the source changes)
cp /path/to/project_docs/wiki/*.md .
git add -A
git commit -m "docs(wiki): sync from project_docs/wiki @ <commit-sha>"
git push
```

If you prefer a one-liner from the repo root:

```bash
TMP=$(mktemp -d) && \
  git clone https://github.com/VytCepas/credit_default_risk_assessment.wiki.git "$TMP" && \
  cp project_docs/wiki/*.md "$TMP"/ && \
  cd "$TMP" && git add -A && \
  git commit -m "docs(wiki): sync from project_docs/wiki" && \
  git push && \
  cd - && rm -rf "$TMP"
```

The wiki accepts the same Markdown flavour as the rest of the repo (GFM +
tables), but **does not** render relative `.md` links between repo and wiki
the same way GitHub does — prefer absolute `https://github.com/...` links
when linking from the wiki into the repo.
